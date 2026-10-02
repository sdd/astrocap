#![allow(unused_imports)]

mod config;

use std::sync::Arc;

use astrocap_processor_detection_export::DetectionExportProcessor;
use astrocap_processor_frame_stacker::FrameStackerProcessor;
use astrocap_processor_mask::MaskProcessor;
use astrocap_processor_median_sub::{ImgSubberProcessor, MedianProcessor};
use vyd::pipeline::run_pipeline_with_config_file_path;
use vyd::statistics::PipelineStatistics;
// use astrocap_processor_model::ModelState;
use astrocap_processor_point_detector::PointDetectorProcessor;
use astrocap_processor_point_fitter::PointFitterProcessor;
use astrocap_processor_rerun_img_tee::RerunTeeProcessor;
use astrocap_processor_solvastro::SolvastroProcessor;
use astrocap_processor_tracker::TrackerProcessor;
use astrocap_processor_video_export::VideoExportProcessor;
use astrocap_sink_point_detector_evaluator::PointDetectorEvaluatorSink;
use astrocap_sink_rerun::RerunSink;
use astrocap_sink_tracker_evaluator::TrackerEvaluatorSink;
use astrocap_source_gstreamer::GstSource;
use astrocap_source_recorded_detections::RecordedDetectionsSource;

use crate::config::Config;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .with_thread_names(false)
        .compact()
        .init();

    // Parse command line arguments
    let config = Config::parse_args();

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

    run_pipeline_with_config_file_path(&config.config, stats.clone(), Some("./dumps".into()))?;

    // If we reach here, the pipeline completed naturally
    stats.print_final_stats();
    Ok(())
}
