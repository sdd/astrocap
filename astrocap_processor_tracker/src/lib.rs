pub mod config;
pub mod hypothesis_trackers;
pub mod model;
pub mod processor;
mod recording;
pub mod traits;

pub use processor::TrackerProcessor;
vyd::register_vyd_frame_processor!(TrackerProcessor);
vyd_telemetry::register_vyd_stage_codec!(
    recording::TrackerCodec,
    "astrocap_processor_tracker::TrackerProcessor"
);
