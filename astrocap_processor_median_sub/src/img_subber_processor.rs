use astrocap_core::pipeline::PipelineContext;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessor, FrameProcessorResult};
use image::Luma;
use toml::Value;

use crate::map_colors::map_colors;

pub struct ImgSubberProcessor {
    subtractand_key: String,
}

impl ImgSubberProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let subtractand_key = config
            .and_then(|c| c.get("subtractand_key"))
            .and_then(|d| d.as_str())
            .map(|d| d.to_string());

        let Some(subtractand_key) = subtractand_key else {
            return Err(AstrocapError::PluginError(
                "subtractand_key not present in ImgSubber Config".to_string(),
            ));
        };

        Ok(Self { subtractand_key })
    }
}

impl FrameProcessor for ImgSubberProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        tracing::trace!(
            "MedianSubberProcessor received frame with metadata keys: {:?}",
            frame_ctx.metadata.keys().collect::<Vec<_>>()
        );

        let Frame::ImgBuf(ref frame) = frame_ctx.frame else {
            tracing::error!("FrameContext frame is not an ImgBuf!");
            return FrameProcessorResult::Skip;
        };

        let metadata_key = format!("video/{}", self.subtractand_key);
        let Some(subtractand) = frame_ctx.metadata.get(&metadata_key) else {
            tracing::error!("{metadata_key} not present in FrameContext!");
            return FrameProcessorResult::Skip;
        };

        let Some(subtractand) = subtractand.downcast_ref::<Frame>() else {
            tracing::error!("{metadata_key} not downcastable to Frame::Imguf!");
            return FrameProcessorResult::Skip;
        };

        let subtractand = match subtractand {
            Frame::ImgBuf(subtractand) => subtractand,
            _ => {
                tracing::error!("{metadata_key} not an ImgBuf!");
                return FrameProcessorResult::Skip;
            }
        };

        // subtract frame from sub_from
        let subtracted = map_colors(frame, subtractand, |p, q| {
            Luma([(p[0]).saturating_sub(q[0])])
        });

        frame_ctx.frame = Frame::ImgBuf(subtracted);

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "img_subber"
    }
}
