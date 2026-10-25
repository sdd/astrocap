#![allow(unused_imports)]

mod config;

use astrocap_processor_detection_export::DetectionExportProcessor;
use astrocap_processor_frame_stacker::FrameStackerProcessor;
use astrocap_processor_mask::MaskProcessor;
use astrocap_processor_median_sub::{ImgSubberProcessor, MedianProcessor};
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

    let config = Config::parse_args();
    let builder =
        vyd::PipelineBuilder::from_config_file(&config.config_path)?.with_dump_dir("./dumps")?;

    let pipeline = builder.build();
    let stats = pipeline.statistics();

    // Set up Ctrl+C handler
    let stats_for_signal = stats.clone();
    ctrlc::set_handler(move || {
        println!("\nShutting down...");
        stats_for_signal.stop();
    })
    .expect("Error setting Ctrl-C handler");

    pipeline.run()?;

    stats.print_final_stats();
    Ok(())
}
