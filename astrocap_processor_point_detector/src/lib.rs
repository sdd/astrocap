pub mod config;
pub mod detectors;
pub mod point_detector;

pub use point_detector::PointDetectorProcessor;
astrocap_core::register_astrocap_frame_processor!(PointDetectorProcessor);
