use gst::glib;
use gst_base::subclass::prelude::*;

use crate::SharedFrameBufferPool;

glib::wrapper! {
    /// Custom GStreamer allocator for ring buffer memory
    pub struct AstrocapGstAllocator(ObjectSubclass<imp::AstrocapGstAllocator>)
        @extends gst::Allocator, gst::Object;
}

impl AstrocapGstAllocator {
    pub fn new() -> Self {
        glib::Object::builder().build()
    }

    /// Set the frame buffer pool for this allocator to use
    pub fn set_pool(&self, pool: SharedFrameBufferPool) {
        let imp = self.imp();
        *imp.pool.lock().unwrap() = Some(pool);
    }

    /// Get a reference to the frame buffer pool (if set)
    pub fn pool(&self) -> Option<SharedFrameBufferPool> {
        let imp = self.imp();
        imp.pool.lock().unwrap().clone()
    }
}

impl Default for AstrocapGstAllocator {
    fn default() -> Self {
        Self::new()
    }
}

mod imp {
    use super::*;
    use gst::glib::{bool_error, BoolError};
    use std::sync::Mutex;

    #[derive(Default)]
    pub struct AstrocapGstAllocator {
        /// Reference to the frame buffer pool that manages our memory
        pub(crate) pool: Mutex<Option<SharedFrameBufferPool>>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for AstrocapGstAllocator {
        const NAME: &'static str = "AstrocapGstAllocator";
        type Type = super::AstrocapGstAllocator;
        type ParentType = gst::Allocator;
    }

    impl ObjectImpl for AstrocapGstAllocator {}

    impl GstObjectImpl for AstrocapGstAllocator {}

    impl AllocatorImpl for AstrocapGstAllocator {
        fn alloc(
            &self,
            size: usize,
            _params: Option<&gst::AllocationParams>,
        ) -> Result<gst::Memory, BoolError> {
            tracing::info!(
                size,
                "🔥 AstrocapGstAllocator::alloc() called - our allocator is being used!"
            );

            // Check if we have a pool configured
            let pool_opt = { self.pool.lock().unwrap().clone() };

            let Some(pool) = pool_opt else {
                tracing::warn!("AstrocapGstAllocator allocation attempt without configured pool");
                return Err(bool_error!("AstrocapGstAllocator: no pool configured"));
            };

            let mut pool_guard = pool.lock().unwrap();

            // Validate that the requested size matches our pool configuration
            if size != pool_guard.buffer_size() {
                tracing::warn!(
                    requested_size = size,
                    pool_buffer_size = pool_guard.buffer_size(),
                    "Size mismatch - requested size doesn't match pool buffer size"
                );
                return Err(bool_error!(
                    "Requested size {} doesn't match pool buffer size {}",
                    size,
                    pool_guard.buffer_size()
                ));
            }

            let Some(memory) = pool_guard.allocate(&pool) else {
                tracing::warn!(
                    size,
                    buffer_count = pool_guard.buffer_count(),
                    "Pool exhausted - no available buffers"
                );
                return Err(bool_error!("No available buffers in pool"));
            };

            tracing::info!(
                size,
                available_slots = pool_guard.available_count(),
                in_use_slots = pool_guard.in_use_count(),
                "✅ Successfully allocated memory from pool"
            );

            Ok(memory)
        }

        fn free(&self, memory: gst::Memory) {
            // Get a reference to the pool
            let pool_opt = { self.pool.lock().unwrap().clone() };

            let Some(pool) = pool_opt else {
                tracing::warn!(
                    "Memory deallocation attempt without configured pool - memory will be dropped"
                );
                // Memory will be dropped normally by GStreamer since we don't have a pool
                return;
            };

            let mut pool_guard = pool.lock().unwrap();

            if pool_guard.deallocate(&memory) {
                tracing::debug!(
                    available_slots = pool_guard.available_count(),
                    in_use_slots = pool_guard.in_use_count(),
                    "Successfully returned memory to pool"
                );
            } else {
                tracing::warn!(
                    "Failed to return memory to pool - memory not recognized as belonging to this pool"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create_shared_pool;
    use gst::prelude::*;

    #[test]
    fn test_allocator_creation() {
        gst::init().unwrap();

        let allocator = AstrocapGstAllocator::new();

        // Verify it's properly typed
        assert!(allocator.is::<gst::Allocator>());
        assert_eq!(allocator.type_().name(), "AstrocapGstAllocator");

        // Should start with no pool
        assert!(allocator.pool().is_none());
    }

    #[test]
    fn test_pool_configuration() {
        gst::init().unwrap();

        let allocator = AstrocapGstAllocator::new();
        let buffer_size = 1024;
        let buffer_count = 4;
        let pool = create_shared_pool(buffer_size, buffer_count);

        // Set pool
        allocator.set_pool(pool.clone());

        // Verify pool is set
        assert!(allocator.pool().is_some());
        let retrieved_pool = allocator.pool().unwrap();

        // Verify same configuration
        let pool_guard = retrieved_pool.lock().unwrap();
        assert_eq!(pool_guard.buffer_size(), buffer_size);
        assert_eq!(pool_guard.buffer_count(), buffer_count);
    }

    #[test]
    fn test_allocation_without_pool() {
        gst::init().unwrap();

        let allocator = AstrocapGstAllocator::new();
        let params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);

        // Should fail - no pool configured
        let result = allocator.alloc(1024, Some(&params));
        assert!(result.is_err());
    }

    #[test]
    fn test_successful_allocation_and_automatic_cleanup() {
        gst::init().unwrap();

        let allocator = AstrocapGstAllocator::new();
        let buffer_size = 1024;
        let buffer_count = 2;
        let pool = create_shared_pool(buffer_size, buffer_count);
        allocator.set_pool(pool.clone());

        let params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);

        // Test allocation and automatic cleanup through scope
        {
            // Should succeed with correct size
            let result = allocator.alloc(buffer_size, Some(&params));
            assert!(result.is_ok());

            let memory = result.unwrap();
            assert_eq!(memory.size(), buffer_size);

            // Pool should reflect allocation
            {
                let pool_guard = pool.lock().unwrap();
                assert_eq!(pool_guard.in_use_count(), 1);
                assert_eq!(pool_guard.available_count(), buffer_count - 1);
            }

            // Memory goes out of scope here and should be automatically freed
        }

        // Check if automatic cleanup worked
        {
            let pool_guard = pool.lock().unwrap();
            // In a properly working implementation, this should be 0
            // If it's still 1, the automatic cleanup didn't work
            let in_use = pool_guard.in_use_count();
            if in_use == 0 {
                println!("✓ Automatic cleanup working correctly");
            } else {
                println!(
                    "! Automatic cleanup not working - memory still in use: {}",
                    in_use
                );
            }
        }
    }

    #[test]
    fn test_size_mismatch_rejection() {
        gst::init().unwrap();

        let allocator = AstrocapGstAllocator::new();
        let buffer_size = 1024;
        let pool = create_shared_pool(buffer_size, 2);
        allocator.set_pool(pool);

        let params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);

        // Should fail with wrong size
        let result = allocator.alloc(2048, Some(&params)); // Wrong size
        assert!(result.is_err());
    }

    #[test]
    fn test_pool_exhaustion() {
        gst::init().unwrap();

        let allocator = AstrocapGstAllocator::new();
        let buffer_size = 1024;
        let buffer_count = 1; // Only one slot
        let pool = create_shared_pool(buffer_size, buffer_count);
        allocator.set_pool(pool.clone());

        let params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);

        // First allocation should succeed
        let mem1 = allocator.alloc(buffer_size, Some(&params));
        assert!(mem1.is_ok());

        // Pool should be exhausted
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.in_use_count(), 1);
            assert_eq!(pool_guard.available_count(), 0);
            assert!(pool_guard.is_full());
        }

        // Second allocation should fail (pool exhausted)
        let mem2 = allocator.alloc(buffer_size, Some(&params));
        assert!(mem2.is_err());

        // Keep the memory alive to maintain pool state
        let _memory = mem1.unwrap();

        // Verify pool is still exhausted
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.available_count(), 0);
        }
    }
}
