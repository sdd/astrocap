mod config;
mod frame_buffer;
mod gst_pipeline;
pub mod gst_source;
mod ring_buffer_sink;

use astrocap_core::register_astrocap_frame_source;

pub use crate::gst_source::GstSource;

register_astrocap_frame_source!(GstSource);
