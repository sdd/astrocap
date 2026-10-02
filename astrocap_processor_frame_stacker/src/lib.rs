mod processor;

pub use processor::FrameStackerProcessor;

vyd::register_vyd_frame_processor!(FrameStackerProcessor);
