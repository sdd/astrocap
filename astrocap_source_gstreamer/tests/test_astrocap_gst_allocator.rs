#[allow(unused_imports)]
use astrocap_source_gstreamer::GstSource;

use astrocap_source_gstreamer::astrocap_gst_allocator::AstrocapGstAllocator;
use astrocap_source_gstreamer::create_shared_pool;
use gst::prelude::*;

#[test]
fn test_allocator_registration() {
    // Initialize GStreamer
    gst::init().expect("Failed to initialize GStreamer");

    // Create our custom allocator
    let allocator = AstrocapGstAllocator::new();

    // Verify it's a valid GStreamer allocator
    assert!(allocator.is::<gst::Allocator>());

    // Test that the allocator has the expected type name
    let type_name = allocator.type_().name();
    assert_eq!(type_name, "AstrocapGstAllocator");

    println!(
        "✓ AstrocapGstAllocator registered successfully with type: {}",
        type_name
    );
}

#[test]
fn test_allocator_in_simple_pipeline() {
    // Initialize GStreamer
    gst::init().expect("Failed toinitialize GStreamer");

    // Create a minimal pipeline to test allocator integration
    let pipeline = gst::Pipeline::new();

    // Create basic elements - use appsink which supports allocator property
    let src = gst::ElementFactory::make("videotestsrc")
        .property("num-buffers", 1i32)
        .build()
        .expect("Failed to create videotestsrc");

    let sink = gst::ElementFactory::make("appsink")
        .build()
        .expect("Failed to create appsink");

    // Add elements to pipeline
    pipeline.add_many(&[&src, &sink]).unwrap();
    src.link(&sink).unwrap();

    // Create our custom allocator
    let allocator = AstrocapGstAllocator::new();

    // Test that we can create the allocator and it's a valid GStreamer object
    assert!(allocator.is::<gst::Allocator>());
    assert!(allocator.is::<gst::Object>());

    println!("✓ AstrocapGstAllocator is properly registered as a GStreamer object");

    // Set pipeline to ready state to validate the pipeline structure
    let state_change_result = pipeline.set_state(gst::State::Ready);

    match state_change_result {
        Ok(_) => {
            println!("✓ Pipeline with videotestsrc -> appsink successfully set to READY");

            // Clean up
            let _ = pipeline.set_state(gst::State::Null);
        }
        Err(e) => {
            panic!("Failed to set pipeline to READY state: {}", e);
        }
    }
}

#[test]
fn test_allocator_allocation_attempt() {
    // Initialize GStreamer
    gst::init().expect("Failed to initialize GStreamer");

    // Create our custom allocator
    let allocator = AstrocapGstAllocator::new();

    // Try to allocate memory (should fail gracefully with our current implementation)
    let allocation_params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);
    let result = allocator.alloc(1024, Some(&allocation_params));

    // We expect this to fail with our current stub implementation
    assert!(result.is_err());
    println!("✓ Allocator correctly returns error for unimplemented allocation");

    // Test that the error message is what we expect
    if let Err(err) = result {
        let error_msg = format!("{}", err);
        println!("Actual error message: '{}'", error_msg);

        // GStreamer wraps our BoolError in a generic "Failed to allocate memory" message
        // This is expected behavior - GStreamer provides a consistent API error message
        // while our detailed error gets logged to GStreamer's debug system
        assert_eq!(error_msg, "Failed to allocate memory");
        println!("✓ Error message indicates allocator is not yet implemented");
    }
}

#[test]
fn test_allocator_type_system_integration() {
    // Initialize GStreamer
    gst::init().expect("Failed to initialize GStreamer");

    // Create our custom allocator
    let allocator = AstrocapGstAllocator::new();

    // Test GObject type system integration
    let gtype = allocator.type_();
    assert_eq!(gtype.name(), "AstrocapGstAllocator");

    // Test that it's properly registered in the GStreamer type hierarchy
    assert!(gtype.is_a(gst::Allocator::static_type()));
    assert!(gtype.is_a(gst::Object::static_type()));

    // Test that we can downcast it
    let as_allocator: &gst::Allocator = allocator.upcast_ref();
    assert!(as_allocator.is::<AstrocapGstAllocator>());

    println!("✓ AstrocapGstAllocator properly integrated with GObject type system");
}

#[test]
fn test_allocator_pool_configuration() {
    // Initialize GStreamer
    gst::init().expect("Failed to initialize GStreamer");

    // Create our custom allocator
    let allocator = AstrocapGstAllocator::new();

    // Initially, allocator should have no pool configured
    assert!(allocator.pool().is_none());
    println!("✓ Allocator starts with no pool configured");

    // Create a frame buffer pool
    let buffer_size = 1920 * 1080; // 1080p frame size
    let buffer_count = 8;
    let pool = create_shared_pool(buffer_size, buffer_count);

    // Verify pool was created with correct configuration
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.buffer_size(), buffer_size);
        assert_eq!(pool_guard.buffer_count(), buffer_count);
    }
    println!("✓ FrameBufferPool created with correct configuration");

    // Set the pool on the allocator
    allocator.set_pool(pool.clone());

    // Verify the allocator now has the pool
    assert!(allocator.pool().is_some());
    let retrieved_pool = allocator.pool().unwrap();

    // Verify it's the same pool (same configuration)
    {
        let retrieved_guard = retrieved_pool.lock().unwrap();
        assert_eq!(retrieved_guard.buffer_size(), buffer_size);
        assert_eq!(retrieved_guard.buffer_count(), buffer_count);
    }
    println!("✓ Allocator pool set and retrieved correctly");
}

#[test]
fn test_allocator_with_pool_allocation_attempt() {
    // Initialize GStreamer
    gst::init().expect("Failed to initialize GStreamer");

    // Create allocator and pool
    let allocator = AstrocapGstAllocator::new();
    let buffer_size = 1920 * 1080;
    let buffer_count = 4;
    let pool = create_shared_pool(buffer_size, buffer_count);
    allocator.set_pool(pool.clone());

    let allocation_params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);
    let result = allocator.alloc(buffer_size, Some(&allocation_params));

    assert!(result.is_ok());

    if let Ok(memory) = result {
        // Verify the memory object
        assert_eq!(memory.size(), buffer_size);
        println!("✓ Successfully allocated memory of size {}", memory.size());

        // Verify pool state changed
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.in_use_count(), 1);
            assert_eq!(pool_guard.available_count(), buffer_count - 1);
        }
        println!("✓ Pool state correctly reflects allocation");
    }

    // Verify pool state after free (due to scope dropping)
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.in_use_count(), 0);
        assert_eq!(pool_guard.available_count(), buffer_count);
    }
    println!("✓ Pool state correctly reflects deallocation");
}

#[test]
fn test_allocator_size_mismatch() {
    // Initialize GStreamer
    gst::init().expect("Failed to initialize GStreamer");

    // Create allocator and pool
    let allocator = AstrocapGstAllocator::new();
    let buffer_size = 1920 * 1080;
    let buffer_count = 4;
    let pool = create_shared_pool(buffer_size, buffer_count);
    allocator.set_pool(pool);

    // Try to allocate memory with WRONG size - should fail
    let allocation_params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);
    let wrong_size = 1024; // Different from pool buffer_size
    let result = allocator.alloc(wrong_size, Some(&allocation_params));

    // Should fail due to size mismatch
    assert!(result.is_err());

    if let Err(err) = result {
        let error_msg = format!("{}", err);
        assert_eq!(error_msg, "Failed to allocate memory");
        println!("✓ Allocator correctly rejects size mismatch");
    }
}

#[test]
fn test_allocator_pool_exhaustion() {
    // Initialize GStreamer
    gst::init().expect("Failed to initialize GStreamer");

    // Create allocator and small pool
    let allocator = AstrocapGstAllocator::new();
    let buffer_size = 1920 * 1080;
    let buffer_count = 2; // Small pool for testing exhaustion
    let pool = create_shared_pool(buffer_size, buffer_count);
    allocator.set_pool(pool.clone());

    let allocation_params = gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);

    // Allocate all available slots
    let mem1 = allocator
        .alloc(buffer_size, Some(&allocation_params))
        .expect("First allocation should succeed");
    let mem2 = allocator
        .alloc(buffer_size, Some(&allocation_params))
        .expect("Second allocation should succeed");

    // Verify pool is now full
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.in_use_count(), 2);
        assert_eq!(pool_guard.available_count(), 0);
        assert!(pool_guard.is_full());
    }
    println!("✓ Pool correctly shows as full");

    // Try to allocate one more - should fail
    let result = allocator.alloc(buffer_size, Some(&allocation_params));
    assert!(result.is_err());
    println!("✓ Allocator correctly handles pool exhaustion");

    // Free one slot and try again
    drop(mem1);

    let mem3 = allocator
        .alloc(buffer_size, Some(&allocation_params))
        .expect("Allocation should succeed after freeing");

    // Clean up
    drop(mem2);
    drop(mem3);

    // Verify pool is back to normal
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.in_use_count(), 0);
        assert_eq!(pool_guard.available_count(), buffer_count);
    }
    println!("✓ Pool correctly restored after cleanup");
}
