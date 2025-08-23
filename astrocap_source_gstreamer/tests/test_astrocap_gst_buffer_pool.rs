use test_log::test;

#[allow(unused_imports)]
use astrocap_source_gstreamer::GstSource;

use astrocap_source_gstreamer::create_shared_pool;
use astrocap_source_gstreamer::AstrocapGstBufferPool;
use gst::prelude::*;

#[test]
fn test_buffer_pool_in_simple_pipeline() {
    gst::init().expect("Failed to initialize GStreamer");

    let pipeline = gst::Pipeline::new();

    // Create basic elements - use appsink which supports buffer pool property
    let src = gst::ElementFactory::make("videotestsrc")
        .property("num-buffers", 1i32)
        .build()
        .expect("Failed to create videotestsrc");

    let sink = gst::ElementFactory::make("appsink")
        .build()
        .expect("Failed to create appsink");

    pipeline.add_many(&[&src, &sink]).unwrap();
    src.link(&sink).unwrap();

    // Create our custom buffer pool
    let buffer_pool = AstrocapGstBufferPool::new();

    // Test that we can create the buffer pool and it's a valid GStreamer object
    assert!(buffer_pool.is::<gst::BufferPool>());
    assert!(buffer_pool.is::<gst::Object>());

    // Test GObject type system integration
    let gtype = buffer_pool.type_();
    assert_eq!(gtype.name(), "AstrocapGstBufferPool");

    // Test that it's properly registered in the GStreamer type hierarchy
    assert!(gtype.is_a(gst::BufferPool::static_type()));
    assert!(gtype.is_a(gst::Object::static_type()));

    // Test that we can downcast it
    let as_buffer_pool: &gst::BufferPool = buffer_pool.upcast_ref();
    assert!(as_buffer_pool.is::<AstrocapGstBufferPool>());

    // Set pipeline to ready state to validate the pipeline structure
    let state_change_result = pipeline.set_state(gst::State::Ready);

    match state_change_result {
        Ok(_) => {
            // Clean up
            let _ = pipeline.set_state(gst::State::Null);
        }
        Err(e) => {
            panic!("Failed to set pipeline to READY state: {}", e);
        }
    }
}

#[test]
fn test_buffer_pool_pool_configuration() {
    gst::init().expect("Failed to initialize GStreamer");

    // Create our custom buffer pool
    let buffer_pool = AstrocapGstBufferPool::new();

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

    // Set the pool on the buffer pool
    buffer_pool.set_frame_buffer_pool(pool.clone());

    // Configure the GStreamer buffer pool
    let mut config = buffer_pool.config();
    config.set_params(None, buffer_size as u32, 0, buffer_count as u32);
    let result = buffer_pool.set_config(config);
    assert!(result.is_ok());

    // Verify the underlying pool still has correct configuration
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.buffer_size(), buffer_size);
        assert_eq!(pool_guard.buffer_count(), buffer_count);
    }
}

#[test]
fn test_buffer_pool_with_pool_acquisition_attempt() {
    gst::init().expect("Failed to initialize GStreamer");

    // Create buffer pool and pool
    let buffer_pool = AstrocapGstBufferPool::new();
    let buffer_size = 1920 * 1080;
    let buffer_count = 4;
    let pool = create_shared_pool(buffer_size, buffer_count);
    buffer_pool.set_frame_buffer_pool(pool.clone());

    // Configure buffer pool
    let mut config = buffer_pool.config();
    config.set_params(None, buffer_size as u32, 0, buffer_count as u32);
    buffer_pool.set_config(config).unwrap();
    buffer_pool.set_active(true).unwrap();

    let result = buffer_pool.acquire_buffer(None);

    assert!(result.is_ok());

    if let Ok(buffer) = result {
        // Verify the buffer object
        assert_eq!(buffer.size(), buffer_size);
        println!("✓ Successfully acquired buffer of size {}", buffer.size());

        // Verify pool state changed
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.in_use_count(), 1);
            assert_eq!(pool_guard.available_count(), buffer_count - 1);
        }
        println!("✓ Pool state correctly reflects acquisition");
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
fn test_buffer_pool_size_mismatch() {
    // Initialize GStreamer
    gst::init().expect("Failed to initialize GStreamer");

    // Create buffer pool and pool
    let buffer_pool = AstrocapGstBufferPool::new();
    let buffer_size = 1920 * 1080;
    let buffer_count = 4;
    let pool = create_shared_pool(buffer_size, buffer_count);
    buffer_pool.set_frame_buffer_pool(pool);

    // Configure with WRONG size - should be detected during acquisition
    let wrong_size = 1024; // Different from pool buffer_size
    let mut config = buffer_pool.config();
    config.set_params(None, wrong_size, 0, buffer_count as u32);
    let config_result = buffer_pool.set_config(config);
    assert!(config_result.is_err()); // Configuration should fail for now

    /*buffer_pool.set_active(true).unwrap();

    // Try to acquire buffer - should fail due to size mismatch
    let result = buffer_pool.acquire_buffer(None);

    // Should fail due to size mismatch
    assert!(result.is_err());

    if let Err(flow_error) = result {
        // Verify it's the correct error type
        assert_eq!(flow_error, gst::FlowError::Error);
        println!("✓ Buffer pool correctly rejects size mismatch");
    }*/
}

#[test]
fn test_buffer_pool_pool_exhaustion() {
    gst::init().expect("Failed to initialize GStreamer");

    // Create buffer pool and small pool
    let buffer_pool = AstrocapGstBufferPool::new();
    let buffer_size = 1920 * 1080;
    let buffer_count = 2; // Small pool for testing exhaustion
    let pool = create_shared_pool(buffer_size, buffer_count);
    buffer_pool.set_frame_buffer_pool(pool.clone());

    // Configure buffer pool
    let mut config = buffer_pool.config();
    config.set_params(None, buffer_size as u32, 0, buffer_count as u32);
    buffer_pool.set_config(config).unwrap();
    buffer_pool.set_active(true).unwrap();

    // Acquire all available slots
    let buf1 = buffer_pool
        .acquire_buffer(None)
        .expect("First acquisition should succeed");
    let buf2 = buffer_pool
        .acquire_buffer(None)
        .expect("Second acquisition should succeed");

    // Verify pool is now full
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.in_use_count(), 2);
        assert_eq!(pool_guard.available_count(), 0);
        assert!(pool_guard.is_full());
    }
    println!("✓ Pool correctly shows as full");

    // Try to acquire one more - should fail
    let result = buffer_pool.acquire_buffer(None);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), gst::FlowError::Flushing);
    println!("✓ Buffer pool correctly handles pool exhaustion");

    // Free one slot and try again
    drop(buf1);

    let buf3 = buffer_pool
        .acquire_buffer(None)
        .expect("Acquisition should succeed after freeing");

    // Clean up
    drop(buf2);
    drop(buf3);

    // Verify pool is back to normal
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.in_use_count(), 0);
        assert_eq!(pool_guard.available_count(), buffer_count);
    }
    println!("✓ Pool correctly restored after cleanup");
}

#[test]
fn test_buffer_pool_direct_usage_with_caps_negotiation() {
    // Test direct usage of buffer pool with proper caps negotiation
    gst::init().unwrap();

    let buffer_size = 640 * 480; // GRAY8 format
    let buffer_count = 4;
    let pool = create_shared_pool(buffer_size, buffer_count);
    let buffer_pool = AstrocapGstBufferPool::new();
    buffer_pool.set_frame_buffer_pool(pool.clone());

    tracing::info!("Testing direct BufferPool usage with caps negotiation...");

    // Create caps for GRAY8 format
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "GRAY8")
        .field("width", 640i32)
        .field("height", 480i32)
        .field("framerate", gst::Fraction::new(30, 1))
        .build();

    // Configure the buffer pool
    let mut config = buffer_pool.config();
    config.set_params(Some(&caps), buffer_size as u32, 2, 8);

    assert!(
        buffer_pool.set_config(config).is_ok(),
        "Failed to configure buffer pool with caps"
    );

    // Activate the buffer pool
    assert!(
        buffer_pool.set_active(true).is_ok(),
        "Failed to activate buffer pool"
    );

    tracing::info!("BufferPool configured and activated successfully");

    // Test multiple buffer acquisitions and releases
    let mut acquired_buffers = Vec::new();

    // Acquire several buffers
    for i in 0..3 {
        let result = buffer_pool.acquire_buffer(None);
        assert!(result.is_ok(), "Failed to acquire buffer #{}", i + 1);

        let buffer = result.unwrap();
        assert_eq!(buffer.size(), buffer_size);

        tracing::info!("Acquired buffer #{}: size = {}", i + 1, buffer.size());

        // Verify pool state
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.in_use_count(), i + 1);
            assert_eq!(pool_guard.available_count(), buffer_count - (i + 1));
        }

        acquired_buffers.push(buffer);
    }

    tracing::info!("All buffers acquired successfully");

    // Release buffers gradually and verify pool state
    for (i, buffer) in acquired_buffers.into_iter().enumerate() {
        let buffers_remaining = 3 - i;

        // Drop the buffer (should return to pool)
        drop(buffer);

        // Verify pool state after release
        {
            let pool_guard = pool.lock().unwrap();
            assert_eq!(pool_guard.in_use_count(), buffers_remaining - 1);
            assert_eq!(
                pool_guard.available_count(),
                buffer_count - (buffers_remaining - 1)
            );
        }

        tracing::info!(
            "Released buffer #{}, {} buffers remaining",
            i + 1,
            buffers_remaining - 1
        );
    }

    // Final verification - all buffers should be available
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.in_use_count(), 0);
        assert_eq!(pool_guard.available_count(), buffer_count);
    }

    // Deactivate the buffer pool
    assert!(
        buffer_pool.set_active(false).is_ok(),
        "Failed to deactivate buffer pool"
    );

    tracing::info!("✅ Direct BufferPool usage test completed successfully");
}

#[test]
fn test_buffer_pool_exhaustion_and_recovery() {
    // Test buffer pool behavior when exhausted and recovery
    gst::init().unwrap();

    let buffer_size = 1024;
    let buffer_count = 2; // Small pool for testing exhaustion
    let pool = create_shared_pool(buffer_size, buffer_count);
    let buffer_pool = AstrocapGstBufferPool::new();
    buffer_pool.set_frame_buffer_pool(pool.clone());

    tracing::info!("Testing BufferPool exhaustion and recovery...");

    // Configure and activate
    let mut config = buffer_pool.config();
    config.set_params(
        None,
        buffer_size as u32,
        buffer_count as u32,
        buffer_count as u32,
    );
    buffer_pool.set_config(config).unwrap();
    buffer_pool.set_active(true).unwrap();

    // Exhaust the pool
    let buf1 = buffer_pool
        .acquire_buffer(None)
        .expect("First buffer acquisition should succeed");
    let buf2 = buffer_pool
        .acquire_buffer(None)
        .expect("Second buffer acquisition should succeed");

    // Verify pool is exhausted
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.in_use_count(), 2);
        assert_eq!(pool_guard.available_count(), 0);
        assert!(pool_guard.is_full());
    }
    tracing::info!("✓ Pool correctly exhausted");

    // Try to acquire one more - should fail with Flushing
    let result = buffer_pool.acquire_buffer(None);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), gst::FlowError::Flushing);
    tracing::info!("✓ BufferPool correctly rejects acquisition when exhausted");

    // Release one buffer
    drop(buf1);

    // Verify recovery
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.in_use_count(), 1);
        assert_eq!(pool_guard.available_count(), 1);
        assert!(!pool_guard.is_full());
    }

    // Should be able to acquire again
    let buf3 = buffer_pool
        .acquire_buffer(None)
        .expect("Acquisition should succeed after release");
    tracing::info!("✓ BufferPool correctly recovered after buffer release");

    // Clean up
    drop(buf2);
    drop(buf3);

    // Final state verification
    {
        let pool_guard = pool.lock().unwrap();
        assert_eq!(pool_guard.in_use_count(), 0);
        assert_eq!(pool_guard.available_count(), buffer_count);
    }

    buffer_pool.set_active(false).unwrap();
    tracing::info!("✅ BufferPool exhaustion and recovery test completed successfully");
}
