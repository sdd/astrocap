mod map_colors;
mod processor;

pub use processor::MaskProcessor;

astrocap_core::register_astrocap_frame_processor!(MaskProcessor);
