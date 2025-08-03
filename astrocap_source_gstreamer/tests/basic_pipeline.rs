use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use test_log::test;

use astrocap_core::pipeline::{build_pipeline, run_pipeline, PipelineConfig, PipelineContext};

use astrocap_source_gstreamer::GstSource;

#[test]
fn test_build_simplest_pipeline_from_toml() {
    let mut file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    file_path.push("tests");
    file_path.push("test.mkv");
    let file_path = file_path.to_string_lossy();
    // Create a TOML configuration for the test pipeline
    let toml_config = format!(
        r#"
[source]
stage_type = "astrocap_source_gstreamer::GstSource"
file_path = "{file_path}"

[[stages]]
stage_type = "astrocap_core::DummyProcessor"

[sink]
stage_type = "astrocap_core::DummySink"
"#
    );

    // Parse the TOML configuration
    let config: PipelineConfig =
        toml::from_str(&toml_config).expect("Failed to parse TOML configuration");

    // Build the pipeline from the configuration
    let pipeline = build_pipeline(&config);

    // Verify that the pipeline components were created correctly
    assert_eq!(
        format!("{:?}", pipeline),
        "Pipeline { source: \"astrocap_source_gstreamer::GstSource\", stages: [\"astrocap_core::DummyProcessor\"], sink: \"astrocap_core::DummySink\" }"
    );
}

#[test]
fn test_run_pipeline_integration() {
    let mut file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    file_path.push("tests");
    file_path.push("test.mkv");
    let file_path = file_path.to_string_lossy();

    // Create a simple pipeline configuration
    let toml_config = format!(
        r#"
[source]
stage_type = "astrocap_source_gstreamer::GstSource"
file_path = "{file_path}"

[[stages]]
stage_type = "astrocap_core::DummyProcessor"

[sink]
stage_type = "astrocap_core::DummySink"
"#
    );

    // Parse the TOML configuration
    let config: PipelineConfig =
        toml::from_str(&toml_config).expect("Failed to parse TOML configuration");

    // Build the pipeline
    let pipeline = build_pipeline(&config);

    // Create pipeline context
    let pipeline_context = PipelineContext::new();

    // Verify initial state
    assert_eq!(
        format!("{:?}", pipeline),
        "Pipeline { source: \"astrocap_source_gstreamer::GstSource\", stages: [\"astrocap_core::DummyProcessor\"], sink: \"astrocap_core::DummySink\" }"
    );

    // Run the pipeline - this should process all frames from the dummy source
    // The dummy source produces 5 frames, so this should process all of them
    let final_context = run_pipeline(pipeline_context, pipeline);

    println!("Pipeline execution completed successfully!");

    // Assert on the statistics recorded in the pipeline context
    let frames_generated = final_context
        .get("frames_generated")
        .expect("frames_generated counter should exist")
        .downcast_ref::<AtomicUsize>()
        .expect("frames_generated should be AtomicUsize")
        .load(Ordering::SeqCst);

    let frames_processed = final_context
        .get("frames_processed")
        .expect("frames_processed counter should exist")
        .downcast_ref::<AtomicUsize>()
        .expect("frames_processed should be AtomicUsize")
        .load(Ordering::SeqCst);

    let frames_sunk = final_context
        .get("frames_sunk")
        .expect("frames_sunk counter should exist")
        .downcast_ref::<AtomicUsize>()
        .expect("frames_sunk should be AtomicUsize")
        .load(Ordering::SeqCst);

    // Verify expected counts - gst source produces 50 frames (2 sec video)
    assert_eq!(frames_generated, 50, "Expected 50 frames to be generated");
    assert_eq!(frames_processed, 50, "Expected 50 frames to be processed");
    assert_eq!(frames_sunk, 50, "Expected 50 frames to be sunk");

    println!("Statistics verification passed:");
    println!("  Frames generated: {}", frames_generated);
    println!("  Frames processed: {}", frames_processed);
    println!("  Frames sunk: {}", frames_sunk);
}

#[test]
fn test_pseudo_live_drops_frames() {
    let mut file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    file_path.push("tests");
    file_path.push("test.mkv");
    let file_path = file_path.to_string_lossy();

    let toml_config = format!(
        r#"
[source]
stage_type = "astrocap_source_gstreamer::GstSource"
file_path = "{file_path}"
pseudo_live = true

[[stages]]
stage_type = "astrocap_core::DummyProcessor"
processing_delay_ms = 200

[sink]
stage_type = "astrocap_core::DummySink"
"#
    );

    let start_time = Instant::now();
    let config: PipelineConfig =
        toml::from_str(&toml_config).expect("Failed to parse TOML configuration");

    let pipeline = build_pipeline(&config);
    let pipeline_context = PipelineContext::new();

    // Run the pipeline
    let final_context = run_pipeline(pipeline_context, pipeline);
    let duration = start_time.elapsed();

    // Get counters
    let frames_generated = final_context
        .get("frames_generated")
        .unwrap()
        .downcast_ref::<AtomicUsize>()
        .unwrap()
        .load(Ordering::SeqCst);

    let frames_processed = final_context
        .get("frames_processed")
        .unwrap()
        .downcast_ref::<AtomicUsize>()
        .unwrap()
        .load(Ordering::SeqCst);

    let frames_sunk = final_context
        .get("frames_sunk")
        .unwrap()
        .downcast_ref::<AtomicUsize>()
        .unwrap()
        .load(Ordering::SeqCst);

    println!("Pseudo-live pipeline statistics:");
    println!("  Duration: {:?}", duration);
    println!("  Frames generated: {}", frames_generated);
    println!("  Frames processed: {}", frames_processed);
    println!("  Frames sunk: {}", frames_sunk);

    // In pseudo-live mode with 200ms delay, we should drop frames
    // 2s video at 25fps = 50 frames total
    // With 200ms delay, we can only process ~10 frames (2s / 0.2s = 10),
    // plus the 10 frames in the buffer.
    assert_eq!(
        frames_sunk, 20,
        "Expected 20 frames to be processed in pseudo-live mode"
    );
    assert_eq!(frames_processed, 20, "Expected 10 frames to be processed");

    // Should finish roughly in 2 seconds (video duration) since we're dropping frames
    assert!(
        duration.as_secs() <= 5,
        "Pipeline should finish quickly in pseudo-live mode"
    );
}

#[test]
fn test_non_live_processes_all_frames() {
    let mut file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    file_path.push("tests");
    file_path.push("test.mkv");
    let file_path = file_path.to_string_lossy();

    let toml_config = format!(
        r#"
[source]
stage_type = "astrocap_source_gstreamer::GstSource"
file_path = "{file_path}"
pseudo_live = false

[[stages]]
stage_type = "astrocap_core::DummyProcessor"
processing_delay_ms = 200

[sink]
stage_type = "astrocap_core::DummySink"
"#
    );

    let start_time = Instant::now();
    let config: PipelineConfig =
        toml::from_str(&toml_config).expect("Failed to parse TOML configuration");

    let pipeline = build_pipeline(&config);
    let pipeline_context = PipelineContext::new();

    // Run the pipeline
    let final_context = run_pipeline(pipeline_context, pipeline);
    let duration = start_time.elapsed();

    // Get counters
    let frames_generated = final_context
        .get("frames_generated")
        .unwrap()
        .downcast_ref::<AtomicUsize>()
        .unwrap()
        .load(Ordering::SeqCst);

    let frames_processed = final_context
        .get("frames_processed")
        .unwrap()
        .downcast_ref::<AtomicUsize>()
        .unwrap()
        .load(Ordering::SeqCst);

    let frames_sunk = final_context
        .get("frames_sunk")
        .unwrap()
        .downcast_ref::<AtomicUsize>()
        .unwrap()
        .load(Ordering::SeqCst);

    println!("Non-live pipeline statistics:");
    println!("  Duration: {:?}", duration);
    println!("  Frames generated: {}", frames_generated);
    println!("  Frames processed: {}", frames_processed);
    println!("  Frames sunk: {}", frames_sunk);

    // In non-live mode, should process all frames
    // 2s video at 25fps = 50 frames total
    assert_eq!(
        frames_generated, 50,
        "Expected all 50 frames to be generated"
    );
    assert_eq!(
        frames_processed, 50,
        "Expected all 50 frames to be processed"
    );
    assert_eq!(frames_sunk, 50, "Expected all 50 frames to be consumed");

    // Should take roughly 10 seconds (50 frames × 200ms delay)
    let expected_duration = Duration::from_millis(50 * 200); // 10 seconds
    let tolerance = Duration::from_secs(2); // Allow 2 second tolerance

    assert!(
        duration >= expected_duration - tolerance && duration <= expected_duration + tolerance,
        "Pipeline should take roughly 10 seconds (was {:?})",
        duration
    );
}
