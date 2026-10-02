use vyd::register_vyd_frame_processor;

mod img_subber_processor;
mod map_colors;
mod median_config;
mod median_filter;
mod median_processor;

pub use img_subber_processor::ImgSubberProcessor;
pub use median_processor::MedianProcessor;

register_vyd_frame_processor!(MedianProcessor);
register_vyd_frame_processor!(ImgSubberProcessor);
