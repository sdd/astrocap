use astrocap_core::FrameProcessorResult::{Continue, Skip};
use astrocap_core::pipeline_services::RerunRecording;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use std::sync::Arc;
use toml::Value;
use vyd::frame::pool::WaitPolicy;
use vyd::pipeline::PipelineContext;
use vyd::register_vyd_frame_processor;
use vyd::statistics::ProcessingType;

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
        if ctx
            .services()
            .latest::<RerunRecording>()
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
            .is_none()
        {
            let recording = rerun::RecordingStreamBuilder::new("astrocap")
                .connect_grpc()
                .map_err(|error| {
                    AstrocapError::GeneralPluginError(format!(
                        "Failed to connect to rerun: {error}"
                    ))
                })?;
            ctx.services()
                .publish(Arc::new(RerunRecording(recording)))
                .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
            tracing::info!("Initialized Rerun connection");
        }

        Ok(())
    }

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let rec = match ctx.services().latest::<RerunRecording>() {
            Ok(Some(rec)) => rec,
            Ok(None) => {
                tracing::error!("Rerun recording was not initialized");
                return Skip;
            }
            Err(error) => {
                tracing::error!(%error, "Rerun service lookup failed");
                return Skip;
            }
        };

        let (width, height) = frame_ctx.frame.dimensions();

        let pixels = match &self.key {
            Some(key) => match frame_ctx.try_get_as::<Arc<Frame>>(key) {
                Ok(frame) => {
                    let Ok(pixels) = frame.try_read_cpu_luma8(WaitPolicy::Immediate) else {
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
                let Ok(pixels) = frame_ctx.frame.try_read_cpu_luma8(WaitPolicy::Immediate) else {
                    tracing::warn!("No frame to process");
                    return Skip;
                };

                pixels
            }
        };

        if let Err(err) = rec.0.log(
            format!("video/{}", self.tag),
            &rerun::Image::from_l8(pixels, [width, height]),
        ) {
            tracing::error!("Failed to log frame: {}", err);
            return Skip;
        }

        Continue
    }

    fn name(&self) -> &str {
        "rerun_tee_processor"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

register_vyd_frame_processor!(RerunTeeProcessor);
