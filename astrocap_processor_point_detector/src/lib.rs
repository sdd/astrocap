pub mod config;
pub mod detectors;
pub mod point_detector;
mod recording;

pub use point_detector::PointDetectorProcessor;
vyd::register_vyd_frame_processor!(PointDetectorProcessor);
vyd_telemetry::register_vyd_stage_codec!(
    recording::PointDetectorCodec,
    "astrocap_processor_point_detector::PointDetectorProcessor"
);
