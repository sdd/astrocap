use astrocap_core::pipeline::PipelineContext;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessor, FrameProcessorResult};
use image::GrayImage;
use toml::Value;

use crate::median_filter::median_filter;

pub struct MedianProcessor {}

impl MedianProcessor {
    pub fn new(_config: Option<&Value>) -> Result<Self, AstrocapError> {
        Ok(Self {})
    }
}

impl FrameProcessor for MedianProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        _ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        if let Frame::ImgBuf(ref img) = frame_ctx.frame {
            let img_median: GrayImage = median_filter(&img, 30, 30);

            let frame = Frame::ImgBuf(img_median);

            frame_ctx
                .metadata
                .insert("video/median".to_string(), Box::new(frame));
        }

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "median"
    }
}
