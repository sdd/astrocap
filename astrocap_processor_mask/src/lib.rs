mod map_colors;
mod processor;

pub use processor::MaskProcessor;

vyd::register_vyd_frame_processor!(MaskProcessor);
