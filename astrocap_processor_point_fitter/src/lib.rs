pub mod config;
pub mod fitters;
pub mod point_fitter;

pub use point_fitter::PointFitterProcessor;
astrocap_core::register_astrocap_frame_processor!(PointFitterProcessor);
