pub mod config;
pub mod processor;

pub use processor::DetectionExportProcessor;

vyd::register_vyd_frame_processor!(DetectionExportProcessor);
