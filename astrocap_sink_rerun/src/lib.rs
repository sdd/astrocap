use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::traits::FrameSink;
use astrocap_core::{AstrocapError, Frame, FrameContext, register_astrocap_frame_sink};
use rerun::RecordingStream;
use std::any;
use std::any::TypeId;
use std::sync::atomic::{AtomicUsize, Ordering};
use toml::Value;

pub struct RerunSink;

impl RerunSink {
    pub fn new(_config: Option<&Value>) -> Result<Self, AstrocapError> {
        Ok(Self {})
    }
}

impl FrameSink for RerunSink {
    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        ctx.entry("rerun".to_string()).or_insert_with(|| {
            Box::new(
                rerun::RecordingStreamBuilder::new("astrocap")
                    .connect_grpc()
                    .map(|res| {
                        tracing::info!("Initialized Rerun connection");
                        res
                    })
                    .map_err(|err| {
                        AstrocapError::GeneralPluginError(format!(
                            "Failed to connect to rerun: {}",
                            err
                        ))
                    })
                    .unwrap(),
            )
        });

        Ok(())
    }

    fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext) {
        let rec = ctx.get_as::<RecordingStream>("rerun").unwrap();

        let Ok(pixels) = frame_ctx.frame.get_pixels(None) else {
            tracing::warn!("No frame to process");
            return;
        };

        if let Err(err) = rec.log("video/final", &rerun::Image::from_l8(pixels, [1920, 1080])) {
            tracing::error!("Failed to log frame: {}", err);
        }
    }

    fn name(&self) -> &str {
        "rerun_sink"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

register_astrocap_frame_sink!(RerunSink);
