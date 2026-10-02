use astrocap_core::traits::FrameSink;
use astrocap_core::{AstrocapError, FrameContext};
use rerun::RecordingStream;
use toml::Value;
use vyd::pipeline::PipelineContext;
use vyd::register_vyd_frame_sink;
use vyd::statistics::ProcessingType;

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
        let rec = ctx.try_get_as::<RecordingStream>("rerun").unwrap();

        let Some((width, height)) = frame_ctx.frame.dimensions() else {
            tracing::warn!("No frame dimensions");
            return;
        };

        let Ok(pixels) = frame_ctx.frame.get_pixels(None) else {
            tracing::warn!("No frame to process");
            return;
        };

        if let Err(err) = rec.log(
            "video/final",
            &rerun::Image::from_l8(pixels, [width, height]),
        ) {
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

register_vyd_frame_sink!(RerunSink);
