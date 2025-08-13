use astrocap_core::FrameProcessorResult::{Continue, Skip};
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::{
    AstrocapError, Frame, FrameContext, FrameProcessorResult, register_astrocap_frame_processor,
};
use rerun::RecordingStream;
use rerun::external::arrow::array::Datum;
use std::sync::Arc;
use toml::Value;

pub struct RerunTeeProcessor {
    tag: String,
    key: Option<String>,
}

impl RerunTeeProcessor {
    pub fn new(_config: Option<&Value>) -> Result<Self, AstrocapError> {
        let key = _config
            .and_then(|c| c.get("key"))
            .and_then(|t| t.as_str())
            .map(|t| t.to_string());

        let tag = _config
            .and_then(|c| c.get("tag"))
            .and_then(|t| t.as_str())
            .unwrap_or_else(|| key.as_deref().unwrap_or("default"))
            .to_string();

        Ok(Self { tag, key })
    }
}

impl FrameProcessor for RerunTeeProcessor {
    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        ctx.entry("rerun".to_string()).or_insert_with(|| {
            Box::new(
                rerun::RecordingStreamBuilder::new("astrocap")
                    .connect_grpc()
                    .inspect(|res| {
                        tracing::info!("Initialized Rerun connection");
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

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let rec = ctx.get_as::<RecordingStream>("rerun").unwrap();

        let pixels = match &self.key {
            Some(key) => match frame_ctx.get_as::<Arc<Frame>>(key) {
                Ok(frame) => {
                    let Some(pixels) = frame.as_cpu_image() else {
                        tracing::warn!("No frame to process");
                        return Skip;
                    };

                    pixels
                }
                Err(_) => {
                    return Skip;
                }
            },
            None => {
                let Ok(pixels) = frame_ctx.frame.get_pixels(None) else {
                    tracing::warn!("No frame to process");
                    return Skip;
                };

                pixels
            }
        };

        if let Err(err) = rec.log(
            format!("video/{}", self.tag),
            &rerun::Image::from_l8(pixels, [1920, 1080]),
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
