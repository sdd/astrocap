#![allow(unused_imports)]
use std::sync::Arc;

use astrocap_core::pipeline::run_pipeline_with_config_file_path;
use astrocap_core::statistics::PipelineStatistics;
use astrocap_processor_frame_stacker::FrameStackerProcessor;
use astrocap_processor_mask::MaskProcessor;
use astrocap_processor_median_sub::{ImgSubberProcessor, MedianProcessor};
use astrocap_processor_model::ModelState;
use astrocap_processor_point_detector::PointDetectorProcessor;
use astrocap_processor_point_fitter::PointFitterProcessor;
use astrocap_processor_rerun_img_tee::RerunTeeProcessor;
use astrocap_sink_rerun::RerunSink;
use astrocap_source_gstreamer::GstSource;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .with_thread_names(false)
        .compact()
        .init();

    // Create shared statistics
    let stats = Arc::new(PipelineStatistics::new());
    let stats_for_signal = stats.clone();

    // Set up Ctrl+C handler
    ctrlc::set_handler(move || {
        println!("\nShutting down...");
        stats_for_signal.stop();

        // Give a brief moment for cleanup
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Print final statistics
        stats_for_signal.print_final_stats();

        std::process::exit(0);
    })
    .expect("Error setting Ctrl-C handler");

    run_pipeline_with_config_file_path("astrocap.toml", stats.clone());

    // If we reach here, the pipeline completed naturally
    stats.print_final_stats();
}
