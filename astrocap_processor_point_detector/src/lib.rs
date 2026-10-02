pub mod config;
pub mod detectors;
pub mod point_detector;

pub use point_detector::PointDetectorProcessor;
vyd::register_vyd_frame_processor!(PointDetectorProcessor);
