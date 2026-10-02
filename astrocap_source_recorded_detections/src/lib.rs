pub mod config;
pub mod source;

pub use source::RecordedDetectionsSource;

vyd::register_vyd_frame_source!(RecordedDetectionsSource);
