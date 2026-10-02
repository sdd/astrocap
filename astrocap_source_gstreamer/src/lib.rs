pub mod astrocap_frame_queue;
pub mod astrocap_gst_allocator;
pub mod astrocap_gst_buffer_pool;
pub mod astrocap_gst_sink;
mod config;
pub mod frame_buffer_pool;
mod gst_buffer_timing_meta;
mod gst_pipeline;
pub mod gst_source;

use vyd::register_vyd_frame_source;

pub use crate::astrocap_gst_allocator::AstrocapGstAllocator;
pub use crate::astrocap_gst_buffer_pool::AstrocapGstBufferPool;
pub use crate::frame_buffer_pool::{create_shared_pool, FrameBufferPool, SharedFrameBufferPool};
pub use crate::gst_source::GstSource;

register_vyd_frame_source!(GstSource);
