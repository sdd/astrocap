pub mod astrocap_gst_allocator;
mod config;
mod frame_buffer;
mod frame_buffer_pool;
mod gst_buffer_timing_meta;
mod gst_pipeline;
pub mod gst_source;
mod ring_buffer_sink;

use astrocap_core::register_astrocap_frame_source;

pub use crate::astrocap_gst_allocator::AstrocapGstAllocator;
pub use crate::frame_buffer_pool::{create_shared_pool, FrameBufferPool, SharedFrameBufferPool};
pub use crate::gst_source::GstSource;

register_astrocap_frame_source!(GstSource);
