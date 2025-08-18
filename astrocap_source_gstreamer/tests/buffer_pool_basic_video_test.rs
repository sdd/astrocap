use gst::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use test_log::test;

use astrocap_source_gstreamer::{create_shared_pool, AstrocapGstBufferPool};

#[test]
fn test_buffer_pool_video_zero_copy_with_videoconvert() {
    // Test with videoconvert element which should respect buffer pool proposals
    gst::init().unwrap();

    let buffer_size = 320 * 240 * 3;
    let buffer_count = 8;
    let pool = create_shared_pool(buffer_size, buffer_count);
    let buffer_pool = AstrocapGstBufferPool::new();
    buffer_pool.set_pool(pool.clone());

    tracing::info!(
        "Testing zero-copy with videotestsrc ! videoconvert ! appsink pipeline using BufferPool..."
    );

    // Create the pipeline: videotestsrc ! videoconvert ! appsink
    let pipeline = gst::Pipeline::with_name("bufferpool-videoconvert-test");

    let videotestsrc = gst::ElementFactory::make("videotestsrc")
        .name("test-src")
        .build()
        .unwrap();

    let videoconvert = gst::ElementFactory::make("videoconvert")
        .name("test-convert")
        .build()
        .unwrap();

    let appsink = gst_app::AppSink::builder().name("test-sink").build();

    // Configure videotestsrc
    videotestsrc.set_property("num-buffers", 5i32);
    videotestsrc.set_property_from_str("pattern", "smpte");

    let src_caps = gst::Caps::builder("video/x-raw")
        .field("format", "RGB") // Target format (larger)
        .field("width", 320i32)
        .field("height", 240i32)
        .build();

    let sink_caps = gst::Caps::builder("video/x-raw")
        .field("format", "GRAY8") // Source format (smaller)
        .field("width", 320i32)
        .field("height", 240i32)
        .build();

    pipeline
        .add_many([&videotestsrc, &videoconvert, appsink.upcast_ref()])
        .unwrap();

    // Link with caps
    videotestsrc
        .link_filtered(&videoconvert, &src_caps)
        .unwrap();
    videoconvert.link_filtered(&appsink, &sink_caps).unwrap();

    let samples_received = Arc::new(Mutex::new(Vec::new()));
    let buffers_from_our_pool = Arc::new(Mutex::new(0u32));
    let buffers_verified_from_pool = Arc::new(Mutex::new(0u32)); // Add this line
    let allocation_queries_received = Arc::new(Mutex::new(0u32));

    let samples_clone = samples_received.clone();
    let queries_clone = allocation_queries_received.clone();
    let buffer_pool_clone = buffer_pool.clone();
    let caps_clone = sink_caps.clone();
    let pool_clone = pool.clone();
    let buffers_from_our_pool_clone = buffers_from_our_pool.clone();
    let buffers_verified_from_pool_clone = buffers_verified_from_pool.clone();

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .propose_allocation(move |_sink, query| {
                let mut queries_count = queries_clone.lock().unwrap();
                *queries_count += 1;
                tracing::info!("Allocation query #{} received by appsink", *queries_count);

                // Configure our custom buffer pool
                let mut config = buffer_pool_clone.config();
                config.set_params(Some(&caps_clone), buffer_size as u32, 2, 0);

                assert!(
                    buffer_pool_clone.set_config(config).is_ok(),
                    "Failed to configure our custom buffer pool"
                );

                // Add our buffer pool to the allocation query
                query.add_allocation_pool(Some(&buffer_pool_clone), buffer_size as u32, 2, 0);

                tracing::info!("Added our custom BufferPool to allocation query");

                true
            })
            .new_sample(move |sink| {
                if let Ok(sample) = sink.pull_sample() {
                    let buffer = sample.buffer().unwrap();
                    let memory = buffer.peek_memory(0);

                    // Check if buffer contains non-zero data and verify it's from our pool
                    let (contains_non_zero_data, is_from_our_pool) = if let Ok(readable_map) = memory.map_readable() {
                        let data = readable_map.as_slice();
                        let data_ptr = data.as_ptr();
                        let has_data = data.iter().any(|&b| b != 0);

                        let from_pool = {
                            let pool_guard = pool_clone.lock().unwrap();
                            pool_guard.is_from_pool(data_ptr)
                        };

                        (has_data, from_pool)
                    } else {
                        (false, false)
                    };

                    if contains_non_zero_data {
                        let mut our_pool_samples = buffers_from_our_pool_clone.lock().unwrap();
                        *our_pool_samples += 1;
                    }

                    if is_from_our_pool {
                        let mut verified_samples = buffers_verified_from_pool_clone.lock().unwrap();
                        *verified_samples += 1;
                    }


                    let mut samples = samples_clone.lock().unwrap();
                    tracing::info!(
                        "Sample {}: size = {}, contains data = {} (videoconvert output via BufferPool)",
                        samples.len() + 1,
                        buffer.size(),
                        contains_non_zero_data
                    );
                    samples.push(sample);

                    // If we're receiving non-zero data, that's great - our memory is being used!
                    assert!(contains_non_zero_data, "🤔 Received all-zero data - either our memory wasn't used or no data was written");
                    assert!(is_from_our_pool, "🤔 Buffer is not from our pool - memory allocation verification failed");
                    tracing::info!("✅ SUCCESS! Non-zero data found and verified as from our pool!");

                    {
                        let pool_guard = pool_clone.lock().unwrap();
                        tracing::debug!(
                            "Pool state: {} available, {} in use",
                            pool_guard.available_count(),
                            pool_guard.in_use_count()
                        );
                    }
                }
                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );

    // Start pipeline
    tracing::info!("Starting pipeline with videoconvert and custom BufferPool...");
    pipeline.set_state(gst::State::Playing).unwrap();

    // Wait for processing
    let bus = pipeline.bus().unwrap();
    let timeout = Duration::from_secs(10);
    let start_time = std::time::Instant::now();
    let mut eos_received = false;

    while !eos_received && start_time.elapsed() < timeout {
        if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(100)) {
            match msg.view() {
                gst::MessageView::Eos(_) => {
                    tracing::info!("Received EOS");
                    eos_received = true;
                }
                gst::MessageView::Error(err) => {
                    panic!("Pipeline error: {:?}", err.error());
                }
                gst::MessageView::StateChanged(state) => {
                    if let Some(src_element) = state.src() {
                        let src_name = src_element.name();
                        let pipeline_name = pipeline.name();
                        if src_name == pipeline_name {
                            tracing::debug!(
                                "Pipeline state changed: {:?} -> {:?}",
                                state.old(),
                                state.current()
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pipeline.set_state(gst::State::Null).unwrap();

    // Check results
    let samples = samples_received.lock().unwrap();
    let our_pool_samples = *buffers_from_our_pool.lock().unwrap();
    let verified_samples = *buffers_verified_from_pool.lock().unwrap();
    let allocation_queries = *allocation_queries_received.lock().unwrap();

    tracing::info!("Test completed:");
    tracing::info!("  Allocation queries received: {}", allocation_queries);
    tracing::info!("  Samples received: {}", samples.len());
    tracing::info!("  Samples from our buffer pool: {}", our_pool_samples);
    tracing::info!(
        "  Samples verified as from our buffer pool: {}",
        verified_samples
    );

    // Basic functionality checks
    assert!(
        allocation_queries > 0,
        "Should have received at least one allocation query"
    );
    assert_eq!(samples.len(), 5, "Should have received exactly 5 samples");

    // Check that our buffer pool was used for buffer creation
    {
        let pool_guard = pool.lock().unwrap();
        let buffers_allocated = buffer_count - pool_guard.available_count();
        tracing::info!("Buffers allocated from pool: {}", buffers_allocated);
        assert!(
            buffers_allocated > 0,
            "Our buffer pool should have been used for buffer creation"
        );
    }

    assert!(
        our_pool_samples > 0,
        "videoconvert did not use our buffer pool for final buffers"
    );

    assert!(
        verified_samples > 0,
        "videoconvert did not verify that final buffers came from our buffer pool"
    );
}
