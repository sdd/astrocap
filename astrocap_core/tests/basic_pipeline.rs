use astrocap_core::pipeline::{build_pipeline, run_pipeline, PipelineConfig, PipelineContext};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn test_build_simplest_pipeline_from_toml() {
    // Create a TOML configuration for the test pipeline
    let toml_config = r#"
[source]
stage_type = "astrocap_core::DummySource"

[[stages]]
stage_type = "astrocap_core::DummyProcessor"

[sink]
stage_type = "astrocap_core::DummySink"
"#;

    // Parse the TOML configuration
    let config: PipelineConfig =
        toml::from_str(toml_config).expect("Failed to parse TOML configuration");

    // Build the pipeline from the configuration
    let pipeline = build_pipeline(&config);

    // Verify that the pipeline components were created correctly
    assert_eq!(
        format!("{:?}", pipeline),
        "Pipeline { source: \"astrocap_core::DummySource\", stages: [\"astrocap_core::DummyProcessor\"], sink: \"astrocap_core::DummySink\" }"
    );
}

#[test]
fn test_build_pipeline_with_multiple_processors() {
    let toml_config = r#"
[source]
stage_type = "astrocap_core::DummySource"

[[stages]]
stage_type = "astrocap_core::DummyProcessor"

[[stages]]
stage_type = "astrocap_core::DummyProcessor"

[sink]
stage_type = "astrocap_core::DummySink"
"#;

    let config: PipelineConfig =
        toml::from_str(toml_config).expect("Failed to parse TOML configuration");

    let pipeline = build_pipeline(&config);

    assert_eq!(
        format!("{:?}", pipeline),
        "Pipeline { source: \"astrocap_core::DummySource\", stages: [\"astrocap_core::DummyProcessor\", \"astrocap_core::DummyProcessor\"], sink: \"astrocap_core::DummySink\" }"
    );
}

#[test]
fn test_run_pipeline_integration() {
    // Create a simple pipeline configuration
    let toml_config = r#"
[source]
stage_type = "astrocap_core::DummySource"

[[stages]]
stage_type = "astrocap_core::DummyProcessor"

[sink]
stage_type = "astrocap_core::DummySink"
"#;

    // Parse the TOML configuration
    let config: PipelineConfig =
        toml::from_str(toml_config).expect("Failed to parse TOML configuration");

    // Build the pipeline
    let pipeline = build_pipeline(&config);

    // Create pipeline context
    let pipeline_context = PipelineContext::new();

    // Verify initial state
    assert_eq!(
        format!("{:?}", pipeline),
        "Pipeline { source: \"astrocap_core::DummySource\", stages: [\"astrocap_core::DummyProcessor\"], sink: \"astrocap_core::DummySink\" }"
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

    // Verify expected counts - dummy source generates 5 frames
    assert_eq!(frames_generated, 5, "Expected 5 frames to be generated");
    assert_eq!(frames_processed, 5, "Expected 5 frames to be processed");
    assert_eq!(frames_sunk, 5, "Expected 5 frames to be sunk");

    println!("Statistics verification passed:");
    println!("  Frames generated: {}", frames_generated);
    println!("  Frames processed: {}", frames_processed);
    println!("  Frames sunk: {}", frames_sunk);
}

#[test]
fn test_run_pipeline_with_multiple_processors() {
    let toml_config = r#"
[source]
stage_type = "astrocap_core::DummySource"

[[stages]]
stage_type = "astrocap_core::DummyProcessor"

[[stages]]
stage_type = "astrocap_core::DummyProcessor"

[sink]
stage_type = "astrocap_core::DummySink"
"#;

    let config: PipelineConfig =
        toml::from_str(toml_config).expect("Failed to parse TOML configuration");

    let pipeline = build_pipeline(&config);
    let pipeline_context = PipelineContext::new();

    // Run the pipeline
    let final_context = run_pipeline(pipeline_context, pipeline);

    // With 2 processors, frames_processed should be 10 (5 frames × 2 processors)
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

    assert_eq!(frames_generated, 5, "Expected 5 frames to be generated");
    assert_eq!(
        frames_processed, 10,
        "Expected 10 total processing operations (5 frames × 2 processors)"
    );
    assert_eq!(frames_sunk, 5, "Expected 5 frames to be consumed");

    println!("Multi-processor pipeline statistics:");
    println!("  Frames generated: {}", frames_generated);
    println!("  Frames processed: {}", frames_processed);
    println!("  Frames sunk: {}", frames_sunk);
}
