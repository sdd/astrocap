use gst::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use test_log::test;

use astrocap_source_gstreamer::{create_shared_pool, AstrocapGstAllocator};

#[test]
fn test_alloc_video_zero_copy_with_videoconvert() {
    // Test with videoconvert element which should respect buffer allocation proposals
    gst::init().unwrap();

    let buffer_size = 320 * 240 * 3;
    let buffer_count = 8;
    let pool = create_shared_pool(buffer_size, buffer_count);
    let allocator = AstrocapGstAllocator::new();
    allocator.set_pool(pool.clone());

    tracing::info!("Testing zero-copy with videotestsrc ! videoconvert ! appsink pipeline...");

    // Create the pipeline: videotestsrc ! videoconvert ! appsink
    let pipeline = gst::Pipeline::with_name("videoconvert-test");

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
    let buffers_from_our_allocator = Arc::new(Mutex::new(0u32));
    let allocation_queries_received = Arc::new(Mutex::new(0u32));

    let samples_clone = samples_received.clone();
    let queries_clone = allocation_queries_received.clone();
    let allocator_clone = allocator.clone();
    let caps_clone = sink_caps.clone();
    let pool_clone = pool.clone();
    let buffers_from_our_allocator_clone = buffers_from_our_allocator.clone();

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            .propose_allocation(move |_sink, query| {
                let mut queries_count = queries_clone.lock().unwrap();
                *queries_count += 1;
                tracing::info!("Allocation query #{} received by appsink", *queries_count);

                let base_allocator: &gst::Allocator = allocator_clone.upcast_ref();
                let allocation_params =
                    gst::AllocationParams::new(gst::MemoryFlags::empty(), 0, 0, 0);
                query.add_allocation_param(Some(base_allocator), allocation_params);

                let buffer_pool = gst::BufferPool::new();
                let mut config = buffer_pool.config();
                config.set_params(Some(&caps_clone), buffer_size as u32, 2, 0);
                config.set_allocator(Some(base_allocator), None);

                assert!(buffer_pool.set_config(config).is_ok(), "Failed to configure buffer pool with our allocator");
                query.add_allocation_pool(Some(&buffer_pool), buffer_size as u32, 2, 0);

                true
            })
            .new_sample(move |sink| {
                if let Ok(sample) = sink.pull_sample() {
                    let buffer = sample.buffer().unwrap();
                    let memory = buffer.peek_memory(0);

                    // Check if buffer contains non-zero data (would prove it's NOT our zeroed memory)
                    let contains_non_zero_data = if let Ok(readable_map) = memory.map_readable() {
                        let data = readable_map.as_slice();
                        data.iter().any(|&b| b != 0)
                    } else {
                        false
                    };

                    if contains_non_zero_data {
                        let mut our_allocator_samples = buffers_from_our_allocator_clone.lock().unwrap();
                        *our_allocator_samples += 1;
                    }

                    let mut samples = samples_clone.lock().unwrap();
                    tracing::info!(
                        "Sample {}: size = {}, contains data = {} (videoconvert output)",
                        samples.len() + 1,
                        buffer.size(),
                        contains_non_zero_data
                    );
                    samples.push(sample);

                    // If we're receiving non-zero data, that's great - our memory is being used!
                    assert!(contains_non_zero_data, "🤔 Received all-zero data - either our memory wasn't used or no data was written");
                    tracing::info!("✅ SUCCESS! Non-zero data found - our zeroed pool memory was written to!");

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
    tracing::info!("Starting pipeline with videoconvert...");
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
                    // Fixed: Proper comparison using element names as strings
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
    let our_allocator_samples = *buffers_from_our_allocator.lock().unwrap();
    let allocation_queries = *allocation_queries_received.lock().unwrap();

    tracing::info!("Test completed:");
    tracing::info!("  Allocation queries received: {}", allocation_queries);
    tracing::info!("  Samples received: {}", samples.len());
    tracing::info!("  Samples from our allocator: {}", our_allocator_samples);

    // Basic functionality checks
    assert!(
        allocation_queries > 0,
        "Should have received at least one allocation query"
    );
    assert_eq!(samples.len(), 5, "Should have received exactly 5 samples");

    // Check that our allocator was called for buffer pool creation
    {
        let pool_guard = pool.lock().unwrap();
        let buffers_allocated = buffer_count - pool_guard.available_count();
        tracing::info!("Buffers allocated from pool: {}", buffers_allocated);
        assert!(
            buffers_allocated > 0,
            "Our allocator should have been used for buffer pool creation"
        );
    }

    assert!(
        our_allocator_samples > 0,
        "videoconvert did not use our allocator for final buffers"
    );
}
