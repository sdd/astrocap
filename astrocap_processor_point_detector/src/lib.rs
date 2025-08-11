extern crate core;

use astrocap_core::{register_astrocap_frame_processor, Frame};

pub mod config;
pub mod detectors;
pub mod point_detector;

use crate::point_detector::DetectedPoint;
pub use point_detector::PointDetectorProcessor;

pub trait PointDetector: Send + Sync + 'static {
    fn detect(
        &self,
        frame: &Frame,
        median: Option<&Frame>,
        mask: Option<&Frame>,
    ) -> Vec<DetectedPoint>;
}

register_astrocap_frame_processor!(PointDetectorProcessor);
