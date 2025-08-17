use gst::glib;
use std::sync::{Arc, Mutex, Weak};

/// Data passed to the drop callback for memory cleanup
struct PoolMemoryUserData {
    /// Weak reference to the pool to avoid circular references
    pool: Weak<Mutex<FrameBufferPool>>,
    /// Index of the slot in the pool
    slot_index: usize,
}

/// A memory slot in the frame buffer pool
#[derive(Debug)]
struct MemorySlot {
    /// Raw memory buffer for this slot
    buffer: Vec<u8>,
    /// Whether this slot is currently in use
    in_use: bool,
}

/// A pool that manages pre-allocated memory buffers for zero-copy operation with GStreamer.
///
/// This pool contains fixed-size memory slots that can be allocated
/// to GStreamer elements and then reused without copying when passed to the astrocap pipeline.
#[derive(Debug)]
pub struct FrameBufferPool {
    slots: Vec<MemorySlot>,
    buffer_size: usize,
    buffer_count: usize,
}

impl FrameBufferPool {
    /// Create a new frame buffer pool with the specified configuration
    pub fn new(buffer_size: usize, buffer_count: usize) -> Self {
        tracing::debug!(buffer_size, buffer_count, "Creating FrameBufferPool");

        // Pre-allocate all memory slots with raw buffers
        let mut slots = Vec::with_capacity(buffer_count);
        for i in 0..buffer_count {
            let buffer = vec![0u8; buffer_size];
            tracing::trace!(slot_index = i, buffer_size, "Pre-allocated memory slot");

            slots.push(MemorySlot {
                buffer,
                in_use: false,
            });
        }

        tracing::info!(
            buffer_count,
            buffer_size,
            total_allocated_bytes = buffer_count * buffer_size,
            "FrameBufferPool created successfully"
        );

        Self {
            slots,
            buffer_size,
            buffer_count,
        }
    }

    /// Get the configured buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get the configured buffer count
    pub fn buffer_count(&self) -> usize {
        self.buffer_count
    }

    /// Allocate a memory slot from the pool and create GStreamer memory with custom cleanup
    ///
    /// Returns None if no slots are available
    pub fn allocate(&mut self, pool_ref: &SharedFrameBufferPool) -> Option<gst::Memory> {
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if !slot.in_use {
                slot.in_use = true;

                // Create user data for the drop callback
                let user_data = Box::new(PoolMemoryUserData {
                    pool: Arc::downgrade(pool_ref),
                    slot_index: index,
                });

                // Create GStreamer memory with our custom drop function
                let memory = unsafe {
                    let size = slot.buffer.len();
                    let data = slot.buffer.as_mut_ptr();
                    let user_data_ptr = Box::into_raw(user_data);

                    gst::Memory::from_glib_full(gst::ffi::gst_memory_new_wrapped(
                        0, // flags
                        data as glib::ffi::gpointer,
                        size,
                        0, // offset
                        size,
                        user_data_ptr as glib::ffi::gpointer,
                        Some(Self::drop_pool_memory), // destructor
                    ))
                };

                tracing::trace!(
                    slot_index = index,
                    available_slots = self.available_count(),
                    "Allocated memory slot from pool"
                );

                return Some(memory);
            }
        }

        tracing::warn!(
            buffer_count = self.buffer_count,
            "No available memory slots in pool"
        );
        None
    }

    /// Custom destructor function called when GStreamer memory is freed
    unsafe extern "C" fn drop_pool_memory(user_data: glib::ffi::gpointer) {
        let user_data: Box<PoolMemoryUserData> =
            Box::from_raw(user_data as *mut PoolMemoryUserData);

        if let Some(pool) = user_data.pool.upgrade() {
            if let Ok(mut pool_guard) = pool.lock() {
                if user_data.slot_index < pool_guard.slots.len() {
                    pool_guard.slots[user_data.slot_index].in_use = false;

                    tracing::trace!(
                        slot_index = user_data.slot_index,
                        available_slots = pool_guard.available_count(),
                        "Returned memory slot to pool via drop callback"
                    );
                } else {
                    tracing::warn!(
                        slot_index = user_data.slot_index,
                        pool_size = pool_guard.slots.len(),
                        "Invalid slot index in drop callback"
                    );
                }
            } else {
                tracing::warn!("Failed to lock pool in drop callback");
            }
        } else {
            tracing::debug!("Pool was already dropped when memory destructor called");
        }
    }

    /// Deallocate a memory slot back to the pool (fallback method, not needed with custom destructor)
    ///
    /// This finds the slot by comparing the memory address
    pub fn deallocate(&mut self, memory: &gst::Memory) -> bool {
        let memory_ptr = memory.as_ptr();

        // Find the slot by comparing memory pointers
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if slot.in_use && slot.buffer.as_ptr() == memory_ptr as *const u8 {
                slot.in_use = false;

                tracing::trace!(
                    slot_index = index,
                    available_slots = self.available_count(),
                    "Deallocated memory slot back to pool"
                );

                return true;
            }
        }

        tracing::warn!("Attempted to deallocate memory not from this pool");
        false
    }

    pub fn available_count(&self) -> usize {
        self.slots.iter().filter(|slot| !slot.in_use).count()
    }

    pub fn in_use_count(&self) -> usize {
        self.slots.iter().filter(|slot| slot.in_use).count()
    }

    /// Check if the pool is full (no available slots)
    pub fn is_full(&self) -> bool {
        self.available_count() == 0
    }
}

/// Thread-safe wrapper around the frame buffer pool
pub type SharedFrameBufferPool = Arc<Mutex<FrameBufferPool>>;

/// Helper function to create a shared frame buffer pool
pub fn create_shared_pool(buffer_size: usize, buffer_count: usize) -> SharedFrameBufferPool {
    Arc::new(Mutex::new(FrameBufferPool::new(buffer_size, buffer_count)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        gst::init().unwrap();

        let buffer_size = 1920 * 1080;
        let buffer_count = 4;
        let pool = FrameBufferPool::new(buffer_size, buffer_count);

        assert_eq!(pool.buffer_size(), buffer_size);
        assert_eq!(pool.buffer_count(), buffer_count);
        assert_eq!(pool.available_count(), buffer_count);
        assert_eq!(pool.in_use_count(), 0);
        assert!(!pool.is_full());
    }

    #[test]
    fn test_allocation_and_automatic_deallocation() {
        gst::init().unwrap();

        let buffer_size = 1024;
        let buffer_count = 2;
        let pool = create_shared_pool(buffer_size, buffer_count);

        // Test allocation and automatic cleanup through scope
        {
            let mem1 = {
                let mut pool_guard = pool.lock().unwrap();
                pool_guard.allocate(&pool)
            };

            assert!(mem1.is_some());
            let memory = mem1.unwrap();
            assert_eq!(memory.size(), buffer_size);

            // Pool should reflect allocation
            {
                let pool_guard = pool.lock().unwrap();
                assert_eq!(pool_guard.in_use_count(), 1);
                assert_eq!(pool_guard.available_count(), buffer_count - 1);
            }

            // Memory goes out of scope here and should trigger our custom destructor
        }

        // Give GStreamer a moment to call the destructor
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Check if automatic cleanup worked
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.in_use_count(), 0);
            assert_eq!(pool_guard.available_count(), buffer_count);
        }
    }
}
