pub mod config;
pub mod source;

pub use source::RecordedDetectionsSource;

astrocap_core::register_astrocap_frame_source!(RecordedDetectionsSource);
