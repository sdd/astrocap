pub use vyd_gstreamer::*;

use vyd::_register_vyd_stage_impl;

_register_vyd_stage_impl!("astrocap_source_gstreamer", GstSource, FrameSource);
