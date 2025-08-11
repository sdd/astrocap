use tracing_subscriber::{EnvFilter, fmt};

use astrocap_core::pipeline::run_pipeline_with_config_file_path;

use astrocap_processor_frame_stacker::FrameStackerProcessor;
use astrocap_processor_median_sub::{ImgSubberProcessor, MedianProcessor};
use astrocap_processor_point_detector::PointDetectorProcessor;
use astrocap_processor_point_fitter::PointFitterProcessor;
use astrocap_processor_rerun_img_tee::RerunTeeProcessor;
use astrocap_sink_rerun::RerunSink;
use astrocap_source_gstreamer::GstSource;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_target(false)
        .with_thread_ids(false)
        .with_file(true)
        .with_line_number(true)
        .with_thread_names(true)
        .compact()
        .init();

    run_pipeline_with_config_file_path("astrocap.toml");
}
