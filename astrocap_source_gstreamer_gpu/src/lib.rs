pub use vyd_gstreamer_gpu::*;

use vyd::_register_vyd_stage_impl;

_register_vyd_stage_impl!("astrocap_source_gstreamer_gpu", GstGpuSource, FrameSource);
