mod config;
mod frame_exporter;

use astrocap_core::FrameProcessorResult::{Continue, Skip};
use astrocap_core::pipeline_services::IntegratedFrame;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

use std::sync::{Arc, Mutex};
use toml::Value;

use config::Config;
use frame_exporter::FrameExporter;

pub struct VideoExportProcessor {
    config: Config,
    exporter: Option<Arc<Mutex<FrameExporter>>>,
}

impl VideoExportProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: Config = config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!("VideoExportProcessor:: {}", e))
        })?;

        Ok(Self {
            config,
            exporter: None,
        })
    }

    fn get_frame_to_export(
        &self,
        frame_ctx: &FrameContext,
        ctx: &PipelineContext,
    ) -> Result<Frame, AstrocapError> {
        if let Some(key) = &self.config.key {
            if key == "video/integrated" {
                return ctx
                    .services()
                    .latest::<IntegratedFrame>()
                    .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
                    .map(|frame| Frame::Cpu(frame.0.clone()))
                    .ok_or_else(|| {
                        AstrocapError::GeneralPluginError(format!(
                            "video export frame key '{key}' has no published frame"
                        ))
                    })
                    .map_err(|error| {
                        AstrocapError::GeneralPluginError(format!(
                            "video export frame key '{key}': {error}"
                        ))
                    });
            }
            ctx.try_get_as::<Frame>(key).cloned().map_err(|error| {
                AstrocapError::GeneralPluginError(format!(
                    "video export frame key '{key}': {error}"
                ))
            })
        } else {
            Ok(frame_ctx.frame.clone())
        }
    }
}

impl FrameProcessor for VideoExportProcessor {
    fn pipeline_ctx_init(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        let exporter = FrameExporter::new(&self.config.output_path, &self.config)?;
        self.exporter = Some(Arc::new(Mutex::new(exporter)));

        Ok(())
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        let Some(exporter) = self.exporter.take() else {
            return Ok(());
        };
        exporter
            .lock()
            .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?
            .close()
    }

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        match self.try_process(frame_ctx, ctx) {
            Ok(result) => result,
            Err(error) => {
                tracing::error!(%error, "video export failed");
                Skip
            }
        }
    }

    fn try_process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, AstrocapError> {
        // Check if we should start exporting
        if let Some(start_frame_index) = self.config.start_frame_index {
            if frame_ctx.frame_index < start_frame_index {
                return Ok(Continue);
            }

            if frame_ctx.frame_index == start_frame_index {
                tracing::warn!(start_frame_index, "Starting video export");
                // No need to call open() - write_frame() will handle it automatically
            }
        }

        // Check if we should stop exporting
        if let Some(end_frame_index) = self.config.end_frame_index {
            if frame_ctx.frame_index == end_frame_index {
                tracing::warn!(end_frame_index, "Ending video export");
                let exporter = self
                    .exporter
                    .as_ref()
                    .ok_or_else(|| {
                        AstrocapError::GeneralPluginError(
                            "video exporter has not been initialized".into(),
                        )
                    })?
                    .clone();
                exporter
                    .lock()
                    .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?
                    .close()?;
                return Ok(Continue);
            }

            if frame_ctx.frame_index > end_frame_index {
                return Ok(Continue);
            }
        }

        // Export the frame (ensure_open will be called automatically on first frame)
        let exporter = self
            .exporter
            .as_ref()
            .ok_or_else(|| {
                AstrocapError::GeneralPluginError("video exporter has not been initialized".into())
            })?
            .clone();
        let mut exporter = exporter
            .lock()
            .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?;
        let frame_to_export = self.get_frame_to_export(frame_ctx, ctx)?;
        exporter.write_frame(&frame_to_export, ctx)?;

        Ok(Continue)
    }

    fn name(&self) -> &str {
        "selective_video_export_tee_processor"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

vyd::register_vyd_frame_processor!(VideoExportProcessor);

#[cfg(test)]
mod tests {
    use super::*;
    use vyd::frame::CpuFrame;

    #[test]
    fn integrated_frame_key_reads_stacker_output_and_missing_key_errors() {
        let mut config = Config::default();
        config.key = Some("video/integrated".into());
        let processor = VideoExportProcessor {
            config,
            exporter: None,
        };
        let mut ctx = PipelineContext::new();
        let current = FrameContext::new(Frame::from_raw(1, 1, vec![1]).unwrap(), 0);
        assert!(processor.get_frame_to_export(&current, &ctx).is_err());
        ctx.services()
            .publish(Arc::new(IntegratedFrame(
                CpuFrame::from_vec(1, 1, vec![9]).unwrap(),
            )))
            .unwrap();
        let selected = processor.get_frame_to_export(&current, &ctx).unwrap();
        assert_eq!(selected.as_cpu_ref().unwrap(), &[9]);
    }
}
