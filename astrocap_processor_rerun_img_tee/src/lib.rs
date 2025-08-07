use astrocap_core::FrameProcessorResult::{Continue, Skip};
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::{
    AstrocapError, Frame, FrameContext, FrameProcessor, FrameProcessorResult,
    register_astrocap_frame_processor,
};
use rerun::RecordingStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use toml::Value;

pub struct RerunTeeProcessor {
    tag: String,
}

impl RerunTeeProcessor {
    pub fn new(_config: Option<&Value>) -> Result<Self, AstrocapError> {
        let tag = _config
            .and_then(|c| c.get("tag"))
            .and_then(|t| t.as_str())
            .unwrap_or("default")
            .to_string();

        Ok(Self { tag })
    }
}

impl FrameProcessor for RerunTeeProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let Some(rec) = ctx.get("rerun") else {
            tracing::warn!("No rerun context");
            return Continue;
        };

        let rec = rec
            .downcast_ref::<RecordingStream>()
            .expect("Rerun context is not a recording stream");

        let Frame::ImgBuf(ref img) = frame_ctx.frame else {
            tracing::warn!("No frame to process");
            return Skip;
        };

        if let Err(err) = rec.log(
            format!("video/{}", self.tag),
            &rerun::Image::from_pixel_format(
                [1920, 1080],
                rerun::PixelFormat::Y8_FullRange,
                img.as_ref(),
            ),
        ) {
            tracing::error!("Failed to log frame: {}", err);
            return Skip;
        }

        Continue
    }

    fn name(&self) -> &str {
        "rerun_tee_processor"
    }
}

register_astrocap_frame_processor!(RerunTeeProcessor);
