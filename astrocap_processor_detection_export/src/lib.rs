pub mod config;
pub mod processor;

pub use processor::DetectionExportProcessor;

astrocap_core::register_astrocap_frame_processor!(DetectionExportProcessor);
