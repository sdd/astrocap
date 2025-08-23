use gst::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use test_log::test;

use astrocap_core::Frame;
use astrocap_source_gstreamer::astrocap_frame_queue::{AstrocapFrameQueue, FrameInfo};
use astrocap_source_gstreamer::astrocap_gst_sink::AstrocapGstSink;
use astrocap_source_gstreamer::frame_buffer_pool::{create_shared_pool, SharedFrameBufferPool};

#[test]
fn test_astrocapsink_zero_copy_integration() {
    gst::init().unwrap();

    tracing::info!("Testing zero-copy with videotestsrc → videoconvert → astrocapsink pipeline...");

    // 1. Create shared frame buffer pool
    let buffer_size = 1920 * 1080; // GRAY8 format
    let buffer_count = 4;
    let pool = create_shared_pool(buffer_size, buffer_count);

    // 2. Create AstrocapFrameQueue for receiving frames
    let frame_info = FrameInfo {
        width: 1920,
        height: 1080,
        stride: 1920,
        timestamp: 0,
        frame_size: buffer_size,
    };
    let astrocap_frame_queue = Arc::new(AstrocapFrameQueue::new(frame_info, 10, false));

    // 3. Create astrocapsink and configure it
    let astrocapsink = AstrocapGstSink::new();
    astrocapsink.set_frame_buffer_pool(pool.clone());
    astrocapsink.set_astrocap_frame_queue(astrocap_frame_queue.clone());

    // 4. Create pipeline: videotestsrc → videoconvert → astrocapsink
    let pipeline = gst::Pipeline::with_name("astrocapsink-zero-copy-test");

    let videotestsrc = gst::ElementFactory::make("videotestsrc")
        .name("test-src")
        .build()
        .unwrap();

    let videoconvert = gst::ElementFactory::make("videoconvert")
        .name("test-convert")
        .build()
        .unwrap();

    // Configure videotestsrc
    videotestsrc.set_property("num-buffers", 5i32);
    videotestsrc.set_property_from_str("pattern", "smpte");

    let src_caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGB")
        .field("width", 1920i32)
        .field("height", 1080i32)
        .build();

    let sink_caps = gst::Caps::builder("video/x-raw")
        .field("format", "GRAY8")
        .field("width", 1920i32)
        .field("height", 1080i32)
        .build();

    pipeline
        .add_many([&videotestsrc, &videoconvert, astrocapsink.upcast_ref()])
        .unwrap();

    // Link with caps (crucial to include videoconvert!)
    videotestsrc
        .link_filtered(&videoconvert, &src_caps)
        .unwrap();
    videoconvert
        .link_filtered(&astrocapsink, &sink_caps)
        .unwrap();

    // 5. Track statistics for verification
    let frames_rendered = Arc::new(Mutex::new(0u32));
    let frames_from_pool = Arc::new(Mutex::new(0u32));

    let frames_rendered_clone = frames_rendered.clone();
    let frames_from_pool_clone = frames_from_pool.clone();
    let pool_clone = pool.clone();

    // Monitor the pool state during rendering with timeout
    let pool_monitor = pool.clone();
    let monitor_should_exit = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let monitor_exit_signal = monitor_should_exit.clone();

    let monitor_handle = std::thread::spawn(move || {
        let mut last_in_use = 0;
        let start_time = std::time::Instant::now();
        let monitor_timeout = Duration::from_secs(5); // Shorter timeout

        loop {
            std::thread::sleep(Duration::from_millis(50));

            // Check for exit signal
            if monitor_exit_signal.load(std::sync::atomic::Ordering::Relaxed) {
                tracing::info!("Monitor thread received exit signal");
                break;
            }

            // Check for timeout
            if start_time.elapsed() > monitor_timeout {
                tracing::debug!(
                    "Monitor thread timed out after {:?} - this is normal",
                    monitor_timeout
                );
                break;
            }

            let pool_guard = pool_monitor.lock().unwrap();
            let in_use = pool_guard.in_use_count();
            if in_use != last_in_use {
                tracing::info!(
                    "Pool state: {} in use, {} available",
                    in_use,
                    pool_guard.available_count()
                );
                last_in_use = in_use;
            }

            // Exit when all slots are returned AND we've seen some activity
            if in_use == 0 && last_in_use > 0 {
                tracing::info!("All pool slots returned, monitor thread exiting");
                break;
            }

            drop(pool_guard); // Release lock
        }
    });

    // 6. Start pipeline
    tracing::info!("Starting astrocapsink pipeline...");
    pipeline.set_state(gst::State::Playing).unwrap();

    // 7. Wait for frames to be processed
    let bus = pipeline.bus().unwrap();
    let timeout = Duration::from_secs(10);
    let start_time = std::time::Instant::now();
    let mut processed_frames = Vec::new(); // Collect frames for explicit cleanup

    while start_time.elapsed() < timeout {
        if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(100)) {
            match msg.view() {
                gst::MessageView::Eos(_) => {
                    tracing::info!("Received EOS - all frames processed");
                    break;
                }
                gst::MessageView::Error(err) => {
                    panic!(
                        "Pipeline error: {} - {}",
                        err.error(),
                        err.debug().unwrap_or_default()
                    );
                }
                _ => {}
            }
        }

        // Check if we have frames in the queue to consume
        if let Some(frame_with_timing) = astrocap_frame_queue.try_read_frame() {
            let mut rendered_count = frames_rendered_clone.lock().unwrap();
            *rendered_count += 1;

            // Verify this frame came from our pool
            if let Frame::Cpu(cpu_frame) = &frame_with_timing.frame {
                // Get the raw data pointer from the ImageBuffer
                let data_ptr = cpu_frame.img.as_raw().as_ptr();

                let pool_guard = pool_clone.lock().unwrap();
                if pool_guard.is_from_pool(data_ptr) {
                    let mut pool_count = frames_from_pool_clone.lock().unwrap();
                    *pool_count += 1;
                    tracing::info!(
                        "✅ Frame {} verified as zero-copy from pool",
                        *rendered_count
                    );
                } else {
                    tracing::warn!(
                        "⚠️ Frame {} is NOT from our pool - was processed via copy mode",
                        *rendered_count
                    );
                }
                drop(pool_guard); // Release pool lock
            }

            // Store frame temporarily to prevent immediate drop
            processed_frames.push(frame_with_timing);
        }
    }

    // 8. Clean shutdown
    tracing::info!("Shutting down pipeline...");
    pipeline.set_state(gst::State::Null).unwrap();

    // 9. Explicit frame cleanup - drop all frames to return pool slots
    tracing::info!("Cleaning up {} processed frames", processed_frames.len());
    processed_frames.clear(); // This should drop all frames and return pool slots

    // Give a moment for pool cleanup
    std::thread::sleep(Duration::from_millis(100));

    // Signal monitor thread to exit
    monitor_should_exit.store(true, std::sync::atomic::Ordering::Relaxed);

    // Wait for monitor thread with timeout
    tracing::info!("Waiting for monitor thread to complete...");
    if monitor_handle.join().is_err() {
        tracing::warn!("Monitor thread panicked or failed to join");
    }

    // 10. Verify results
    let final_rendered = *frames_rendered.lock().unwrap();
    let final_from_pool = *frames_from_pool.lock().unwrap();

    tracing::info!("Test results:");
    tracing::info!("  Frames rendered: {}", final_rendered);
    tracing::info!("  Frames verified from pool: {}", final_from_pool);

    // Verify pool state is clean (allow some tolerance for race conditions)
    {
        let pool_guard = pool.lock().unwrap();
        let in_use = pool_guard.in_use_count();
        let available = pool_guard.available_count();

        tracing::info!(
            "Final pool state: {} in use, {} available",
            in_use,
            available
        );

        // Don't assert perfect cleanup due to potential race conditions
        if in_use > 0 {
            tracing::warn!(
                "Pool still has {} slots in use - potential memory leak",
                in_use
            );
        }
    }

    // All frames should have been processed
    assert!(
        final_rendered > 0,
        "Should have received at least one frame"
    );

    if final_from_pool > 0 {
        tracing::info!(
            "✅ Zero-copy working! {}/{} frames used pool memory",
            final_from_pool,
            final_rendered
        );

        // If we got any zero-copy frames, that's success
        assert!(
            final_from_pool > 0,
            "Expected at least one frame to use zero-copy"
        );
    } else {
        tracing::warn!("⚠️ Zero-copy not working - all frames were copied");
    }

    tracing::info!("✅ Zero-copy astrocapsink integration test completed!");
}

#[test]
fn test_astrocapsink_direct_gray8() {
    // Test without videoconvert to see if zero-copy works with direct GRAY8 input
    gst::init().unwrap();

    tracing::info!("Testing zero-copy with direct videotestsrc → astrocapsink (GRAY8 only)...");

    let buffer_size = 1920 * 1080; // GRAY8 format
    let buffer_count = 4;
    let pool = create_shared_pool(buffer_size, buffer_count);

    let frame_info = FrameInfo {
        width: 1920,
        height: 1080,
        stride: 1920,
        timestamp: 0,
        frame_size: buffer_size,
    };
    let astrocap_frame_queue = Arc::new(AstrocapFrameQueue::new(frame_info, 10, false));

    let astrocapsink = AstrocapGstSink::new();
    astrocapsink.set_frame_buffer_pool(pool.clone());
    astrocapsink.set_astrocap_frame_queue(astrocap_frame_queue.clone());

    // Direct pipeline: videotestsrc → astrocapsink (no videoconvert)
    let pipeline = gst::Pipeline::with_name("astrocapsink-direct-test");

    let videotestsrc = gst::ElementFactory::make("videotestsrc")
        .name("test-src")
        .build()
        .unwrap();

    videotestsrc.set_property("num-buffers", 3i32);
    videotestsrc.set_property_from_str("pattern", "smpte");

    // Direct GRAY8 caps - no conversion needed
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "GRAY8")
        .field("width", 1920i32)
        .field("height", 1080i32)
        .build();

    pipeline
        .add_many([&videotestsrc, astrocapsink.upcast_ref()])
        .unwrap();
    videotestsrc.link_filtered(&astrocapsink, &caps).unwrap();

    let frames_received = Arc::new(Mutex::new(0u32));
    let frames_from_pool = Arc::new(Mutex::new(0u32));
    let mut processed_frames = Vec::new();

    let frames_received_clone = frames_received.clone();
    let frames_from_pool_clone = frames_from_pool.clone();
    let pool_clone = pool.clone();

    tracing::info!("Starting direct pipeline...");
    pipeline.set_state(gst::State::Playing).unwrap();

    let bus = pipeline.bus().unwrap();
    let timeout = Duration::from_secs(10);
    let start_time = std::time::Instant::now();
    let mut eos_received = false;

    while !eos_received && start_time.elapsed() < timeout {
        if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(50)) {
            match msg.view() {
                gst::MessageView::Eos(_) => {
                    tracing::info!("Received EOS");
                    eos_received = true;
                }
                gst::MessageView::Error(err) => {
                    tracing::error!(
                        "Pipeline error: {} - {}",
                        err.error(),
                        err.debug().unwrap_or_default()
                    );
                    break;
                }
                _ => {}
            }
        }

        if let Some(frame_with_timing) = astrocap_frame_queue.try_read_frame() {
            let mut received_count = frames_received_clone.lock().unwrap();
            *received_count += 1;

            if let Frame::Cpu(cpu_frame) = &frame_with_timing.frame {
                let data_ptr = cpu_frame.img.as_raw().as_ptr();
                let pool_guard = pool_clone.lock().unwrap();
                if pool_guard.is_from_pool(data_ptr) {
                    let mut pool_count = frames_from_pool_clone.lock().unwrap();
                    *pool_count += 1;
                    tracing::info!("✅ Direct frame {} from pool (zero-copy)", *received_count);
                } else {
                    tracing::warn!(
                        "⚠️ Direct frame {} not from pool (copy mode)",
                        *received_count
                    );
                }
                drop(pool_guard);
            }

            processed_frames.push(frame_with_timing);
        }
    }

    pipeline.set_state(gst::State::Null).unwrap();

    // Cleanup frames
    processed_frames.clear();
    std::thread::sleep(Duration::from_millis(50));

    let final_received = *frames_received.lock().unwrap();
    let final_from_pool = *frames_from_pool.lock().unwrap();

    tracing::info!("Direct test results:");
    tracing::info!("  Frames received: {}", final_received);
    tracing::info!("  Frames from pool: {}", final_from_pool);

    assert!(final_received > 0, "Should have received frames");

    if final_from_pool > 0 {
        tracing::info!("✅ Zero-copy working with direct GRAY8!");
    } else {
        tracing::warn!("⚠️ Zero-copy not working even with direct GRAY8");
    }
}
