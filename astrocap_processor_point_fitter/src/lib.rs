extern crate core;

use astrocap_core::structs::{DetectedPoint, FittedPoint};
use astrocap_core::{frame::CpuFrame, register_astrocap_frame_processor};

pub mod config;
pub mod fitters;
pub mod point_fitter;

pub use point_fitter::PointFitterProcessor;

pub trait PointFitter: Send + Sync + 'static {
    fn fit(&self, frame: &CpuFrame, point: &DetectedPoint) -> FittedPoint;
}

register_astrocap_frame_processor!(PointFitterProcessor);
