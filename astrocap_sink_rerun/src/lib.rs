use astrocap_core::pipeline::PipelineContext;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameSink, register_astrocap_frame_sink};
use rerun::RecordingStream;
use std::any;
use std::any::TypeId;
use std::sync::atomic::{AtomicUsize, Ordering};
use toml::Value;

pub struct RerunSink {
    rec: RecordingStream,
}

impl RerunSink {
    pub fn new(_config: Option<&Value>) -> Result<Self, AstrocapError> {
        let rec = rerun::RecordingStreamBuilder::new("astrocap")
            .connect_grpc()
            .expect("Failed to connect to rerun");

        Ok(Self { rec })
    }
}

impl FrameSink for RerunSink {
    fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext) {
        if !ctx.contains_key("rerun") {
            ctx.insert("rerun".to_string(), Box::new(self.rec.clone()));
        }

        let Frame::ImgBuf(ref img) = frame_ctx.frame else {
            tracing::warn!("No frame to process");
            return;
        };

        if let Err(err) = self.rec.log(
            "video/final",
            &rerun::Image::from_pixel_format(
                [1920, 1080],
                rerun::PixelFormat::Y8_FullRange,
                img.as_ref(),
            ),
        ) {
            tracing::error!("Failed to log frame: {}", err);
        }
    }

    fn name(&self) -> &str {
        "rerun_sink"
    }
}

register_astrocap_frame_sink!(RerunSink);
