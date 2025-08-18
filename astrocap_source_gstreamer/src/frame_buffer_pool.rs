use bytes::{Bytes, BytesMut};
use std::sync::{Arc, Mutex};
use tracing;

/// A memory slot in the frame buffer pool
#[derive(Debug)]
struct MemorySlot {
    /// Bytes slice for this slot (view into the arena)
    buffer: Bytes,
    /// Whether this slot is currently in use
    in_use: bool,
    /// Slot index for debugging
    slot_index: usize,
}

/// Handle that represents ownership of a buffer from the pool
pub struct BufferHandle {
    /// The buffer data as a mutable slice
    buffer: *mut [u8],
    /// Pointer to identify this buffer when returning it
    buffer_ptr: *const u8,
    /// Store the length to avoid dereferencing
    len: usize,
}

impl BufferHandle {
    /// Get a mutable slice to the buffer data
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { &mut *self.buffer }
    }

    /// Get the buffer pointer for returning to the pool
    pub fn as_ptr(&self) -> *const u8 {
        self.buffer_ptr
    }

    /// Get the length of the buffer
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

// Safety: BufferHandle contains a valid pointer that we control
unsafe impl Send for BufferHandle {}

/// Frame buffer pool that manages pre-allocated memory buffers for zero-copy operations
#[derive(Debug)]
pub struct FrameBufferPool {
    /// All memory slots
    slots: Vec<MemorySlot>,
    /// Size of each buffer slot
    buffer_size: usize,
    /// Total number of buffer slots
    buffer_count: usize,
    /// The underlying memory arena
    #[allow(unused)]
    memory_arena: Bytes,
    /// Arena bounds for fast address checking
    arena_start: *const u8,
    arena_end: *const u8,
}

impl FrameBufferPool {
    /// Creates a new frame buffer pool with the specified buffer size and count
    pub fn new(buffer_size: usize, buffer_count: usize) -> Self {
        let total_size = buffer_size * buffer_count;

        // Create zeroed arena
        let arena = BytesMut::zeroed(total_size);
        let memory_arena = arena.freeze();

        // Get arena bounds for fast address checking
        let arena_start = memory_arena.as_ptr();
        let arena_end = unsafe { arena_start.add(total_size) };

        // Create slots as views into the arena
        let mut slots = Vec::with_capacity(buffer_count);
        for i in 0..buffer_count {
            let start = i * buffer_size;
            let end = start + buffer_size;
            let slot_buffer = memory_arena.slice(start..end);

            slots.push(MemorySlot {
                buffer: slot_buffer,
                in_use: false,
                slot_index: i,
            });
        }

        let total_allocated_bytes = total_size;

        tracing::debug!(
            buffer_count,
            buffer_size,
            total_allocated_bytes,
            "FrameBufferPool created successfully"
        );

        Self {
            slots,
            buffer_size,
            buffer_count,
            memory_arena,
            arena_start,
            arena_end,
        }
    }

    /// Gets a buffer handle for an available buffer slot
    /// Returns None if no slots are available
    pub fn get_buffer_handle(&mut self) -> Option<BufferHandle> {
        for slot in &mut self.slots {
            if !slot.in_use {
                slot.in_use = true;
                // Convert Bytes to mutable slice
                // Safety: We control the lifecycle and ensure exclusive access
                let ptr = slot.buffer.as_ptr() as *mut u8;
                let len = slot.buffer.len();
                let slice_ptr = std::ptr::slice_from_raw_parts_mut(ptr, len);

                return Some(BufferHandle {
                    buffer: slice_ptr,
                    buffer_ptr: ptr,
                    len,
                });
            }
        }
        None
    }

    /// Legacy method for backward compatibility - gets a mutable slice to an available buffer slot
    /// Returns None if no slots are available
    /// Note: This method has borrowing limitations - prefer get_buffer_handle() for new code
    pub fn get_buffer_mut(&mut self) -> Option<&mut [u8]> {
        for slot in &mut self.slots {
            if !slot.in_use {
                slot.in_use = true;
                // Convert Bytes to mutable slice
                // Safety: We control the lifecycle and ensure exclusive access
                let ptr = slot.buffer.as_ptr() as *mut u8;
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr, slot.buffer.len()) };
                return Some(slice);
            }
        }
        None
    }

    /// Returns a buffer to the pool by pointer
    /// The pointer must be the same as returned by get_buffer_mut or BufferHandle::as_ptr()
    pub fn return_buffer(&mut self, buffer_ptr: *const u8) -> bool {
        for slot in &mut self.slots {
            if slot.buffer.as_ptr() == buffer_ptr {
                if slot.in_use {
                    slot.in_use = false;
                    // Zero out the buffer for next use
                    let ptr = slot.buffer.as_ptr() as *mut u8;
                    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, slot.buffer.len()) };
                    slice.fill(0);
                    return true;
                }
                break;
            }
        }
        false
    }

    /// Super fast ownership check - just two comparisons!
    pub fn contains_address(&self, ptr: *const u8) -> bool {
        ptr >= self.arena_start && ptr < self.arena_end
    }

    /// Get slot index from memory address
    fn address_to_slot(&self, ptr: *const u8) -> Option<usize> {
        if !self.contains_address(ptr) {
            return None;
        }

        let offset = unsafe { ptr.offset_from(self.arena_start) } as usize;
        let slot_index = offset / self.buffer_size;

        if slot_index < self.buffer_count {
            Some(slot_index)
        } else {
            None
        }
    }

    /// Allocate a buffer handle from the pool
    pub fn alloc(&mut self) -> Option<BufferHandle> {
        self.get_buffer_handle()
    }

    /// Free a buffer handle back to the pool by pointer
    pub fn free(&mut self, buffer_ptr: *const u8) -> bool {
        self.return_buffer(buffer_ptr)
    }

    /// Check if the pool is full (all slots in use)
    pub fn is_full(&self) -> bool {
        self.available_count() == 0
    }

    /// Returns the number of available (unused) buffer slots
    pub fn available_count(&self) -> usize {
        self.slots.iter().filter(|slot| !slot.in_use).count()
    }

    /// Returns the number of buffer slots currently in use
    pub fn in_use_count(&self) -> usize {
        self.slots.iter().filter(|slot| slot.in_use).count()
    }

    /// Returns the size of each buffer slot
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Returns the total number of buffer slots
    pub fn buffer_count(&self) -> usize {
        self.buffer_count
    }

    /// Get a Bytes view of a specific slot (useful for sharing across threads)
    pub fn get_slot_bytes(&self, slot_index: usize) -> Option<Bytes> {
        if slot_index < self.slots.len() {
            Some(self.slots[slot_index].buffer.clone())
        } else {
            None
        }
    }

    /// Check if a specific slot is in use
    pub fn is_slot_in_use(&self, slot_index: usize) -> bool {
        self.slots
            .get(slot_index)
            .map(|slot| slot.in_use)
            .unwrap_or(false)
    }

    /// Mark a specific slot as in use (for allocator integration)
    pub fn mark_slot_in_use(&mut self, slot_index: usize) -> bool {
        if let Some(slot) = self.slots.get_mut(slot_index) {
            if !slot.in_use {
                slot.in_use = true;
                return true;
            }
        }
        false
    }

    /// Mark a specific slot as available (for allocator integration)
    pub fn mark_slot_available(&mut self, slot_index: usize) -> bool {
        if let Some(slot) = self.slots.get_mut(slot_index) {
            if slot.in_use {
                slot.in_use = false;
                // Zero out the buffer
                let ptr = slot.buffer.as_ptr() as *mut u8;
                let slice = unsafe { std::slice::from_raw_parts_mut(ptr, slot.buffer.len()) };
                slice.fill(0);
                return true;
            }
        }
        false
    }
}

// Safety: FrameBufferPool is safe to send between threads
// The Bytes type is already Send + Sync, and we manage slot state appropriately
unsafe impl Send for FrameBufferPool {}
unsafe impl Sync for FrameBufferPool {}

/// Thread-safe wrapper around the frame buffer pool
pub type SharedFrameBufferPool = Arc<Mutex<FrameBufferPool>>;

/// Creates a new shared frame buffer pool
pub fn create_shared_pool(buffer_size: usize, buffer_count: usize) -> SharedFrameBufferPool {
    Arc::new(Mutex::new(FrameBufferPool::new(buffer_size, buffer_count)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = FrameBufferPool::new(1024, 4);
        assert_eq!(pool.buffer_size(), 1024);
        assert_eq!(pool.buffer_count(), 4);
        assert_eq!(pool.available_count(), 4);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_buffer_allocation_and_return() {
        let mut pool = FrameBufferPool::new(1024, 2);

        // Store pointers first to avoid borrowing issues
        let (buffer1_ptr, buffer2_ptr) = {
            // Get first buffer
            let buffer1 = pool.get_buffer_mut().unwrap();
            assert_eq!(buffer1.len(), 1024);
            let ptr1 = buffer1.as_ptr();

            // Get second buffer
            let buffer2 = pool.get_buffer_mut().unwrap();
            assert_eq!(buffer2.len(), 1024);
            let ptr2 = buffer2.as_ptr();

            // No more buffers available
            assert!(pool.get_buffer_mut().is_none());

            (ptr1, ptr2)
        };

        // Now we can check counts without borrowing conflicts
        assert_eq!(pool.available_count(), 0);
        assert_eq!(pool.in_use_count(), 2);

        // Return first buffer
        assert!(pool.return_buffer(buffer1_ptr));
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Return second buffer
        assert!(pool.return_buffer(buffer2_ptr));
        assert_eq!(pool.available_count(), 2);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_buffer_handle_approach() {
        let mut pool = FrameBufferPool::new(1024, 2);

        // Get first buffer handle
        let mut handle1 = pool.get_buffer_handle().unwrap();
        assert_eq!(handle1.len(), 1024);

        // We can check pool status while handle is alive!
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Get second buffer handle
        let mut handle2 = pool.get_buffer_handle().unwrap();
        assert_eq!(handle2.len(), 1024);
        assert_eq!(pool.available_count(), 0);
        assert_eq!(pool.in_use_count(), 2);

        // No more buffers available
        assert!(pool.get_buffer_handle().is_none());

        // We can modify buffer contents
        let slice1 = handle1.as_mut_slice();
        slice1[0] = 42;

        let slice2 = handle2.as_mut_slice();
        slice2[0] = 24;

        // Return handles
        let ptr1 = handle1.as_ptr();
        let ptr2 = handle2.as_ptr();

        assert!(pool.return_buffer(ptr1));
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        assert!(pool.return_buffer(ptr2));
        assert_eq!(pool.available_count(), 2);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_address_checking() {
        let pool = FrameBufferPool::new(1024, 3);

        // Test with slot bytes
        let slot0_bytes = pool.get_slot_bytes(0).unwrap();
        let slot1_bytes = pool.get_slot_bytes(1).unwrap();
        let slot2_bytes = pool.get_slot_bytes(2).unwrap();

        // All slots should be within our arena
        assert!(pool.contains_address(slot0_bytes.as_ptr()));
        assert!(pool.contains_address(slot1_bytes.as_ptr()));
        assert!(pool.contains_address(slot2_bytes.as_ptr()));

        // Check slot index calculation
        assert_eq!(pool.address_to_slot(slot0_bytes.as_ptr()), Some(0));
        assert_eq!(pool.address_to_slot(slot1_bytes.as_ptr()), Some(1));
        assert_eq!(pool.address_to_slot(slot2_bytes.as_ptr()), Some(2));

        // Test with external pointer
        let external_data = vec![0u8; 1024];
        assert!(!pool.contains_address(external_data.as_ptr()));
        assert_eq!(pool.address_to_slot(external_data.as_ptr()), None);
    }

    #[test]
    fn test_slot_management() {
        let mut pool = FrameBufferPool::new(512, 2);

        // Initially no slots in use
        assert!(!pool.is_slot_in_use(0));
        assert!(!pool.is_slot_in_use(1));

        // Mark slot 0 as in use
        assert!(pool.mark_slot_in_use(0));
        assert!(pool.is_slot_in_use(0));
        assert!(!pool.is_slot_in_use(1));
        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Can't mark same slot as in use again
        assert!(!pool.mark_slot_in_use(0));

        // Mark slot 0 as available
        assert!(pool.mark_slot_available(0));
        assert!(!pool.is_slot_in_use(0));
        assert_eq!(pool.available_count(), 2);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_shared_pool() {
        let shared_pool = create_shared_pool(256, 3);

        {
            let pool = shared_pool.lock().unwrap();
            assert_eq!(pool.buffer_size(), 256);
            assert_eq!(pool.buffer_count(), 3);
            assert_eq!(pool.available_count(), 3);
        }

        // Test that we can access from multiple scopes using handles
        let buffer_ptr = {
            let mut pool = shared_pool.lock().unwrap();
            let handle = pool.get_buffer_handle().unwrap();
            let ptr = handle.as_ptr();
            assert_eq!(pool.available_count(), 2);
            ptr
        };

        {
            let pool = shared_pool.lock().unwrap();
            assert_eq!(pool.in_use_count(), 1);
        }

        // Return the buffer
        {
            let mut pool = shared_pool.lock().unwrap();
            assert!(pool.return_buffer(buffer_ptr));
            assert_eq!(pool.in_use_count(), 0);
        }
    }

    #[test]
    fn test_allocate_deallocate_interface() {
        let shared_pool = create_shared_pool(1024, 2);
        let mut pool = shared_pool.lock().unwrap();

        // Test allocation
        let handle = pool.alloc();
        assert!(handle.is_some());

        assert_eq!(pool.available_count(), 1);
        assert_eq!(pool.in_use_count(), 1);

        // Test deallocation
        let handle = handle.unwrap();
        let deallocated = pool.free(handle.as_ptr());
        assert!(deallocated);
        assert_eq!(pool.available_count(), 2);
        assert_eq!(pool.in_use_count(), 0);
    }

    #[test]
    fn test_is_full() {
        let mut pool = FrameBufferPool::new(256, 1);

        assert!(!pool.is_full());
        assert_eq!(pool.available_count(), 1);

        // Allocate the only slot using handle approach
        let _handle = pool.get_buffer_handle().unwrap();
        assert!(pool.is_full());
        assert_eq!(pool.available_count(), 0);
    }

    #[test]
    fn test_buffer_handle_len_and_empty() {
        let mut pool = FrameBufferPool::new(512, 1);

        let handle = pool.get_buffer_handle().unwrap();
        assert_eq!(handle.len(), 512);
        assert!(!handle.is_empty());

        // Test with zero-size pool (edge case)
        let mut zero_pool = FrameBufferPool::new(0, 1);
        let zero_handle = zero_pool.get_buffer_handle().unwrap();
        assert_eq!(zero_handle.len(), 0);
        assert!(zero_handle.is_empty());
    }
}
