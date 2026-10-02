pub mod config;
pub mod fitters;
pub mod point_fitter;

pub use point_fitter::PointFitterProcessor;
vyd::register_vyd_frame_processor!(PointFitterProcessor);
