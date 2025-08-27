pub mod config;
pub mod factory;
pub mod model;
pub mod processor;
pub mod traits;

pub use processor::TrackerProcessor;
astrocap_core::register_astrocap_frame_processor!(TrackerProcessor);
