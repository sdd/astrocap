#![cfg_attr(feature = "gpu-wgpu", recursion_limit = "256")]

use vyd::register_vyd_frame_processor;

mod img_subber_processor;
mod map_colors;
mod median_config;
mod median_filter;
#[cfg(feature = "gpu-wgpu")]
mod median_gpu;
mod median_processor;

pub use img_subber_processor::ImgSubberProcessor;
pub use median_processor::MedianProcessor;

register_vyd_frame_processor!(MedianProcessor);
register_vyd_frame_processor!(ImgSubberProcessor);
