pub mod config;
pub mod hypothesis_trackers;
pub mod model;
pub mod processor;
pub mod traits;

pub use processor::TrackerProcessor;
vyd::register_vyd_frame_processor!(TrackerProcessor);
