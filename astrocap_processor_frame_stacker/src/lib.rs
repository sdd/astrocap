mod processor;

pub use processor::FrameStackerProcessor;

astrocap_core::register_astrocap_frame_processor!(FrameStackerProcessor);
