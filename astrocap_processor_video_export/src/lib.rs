mod config;
mod frame_exporter;

use astrocap_core::FrameProcessorResult::{Continue, Skip};
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};

use std::sync::{Arc, Mutex};
use toml::Value;

use config::Config;
use frame_exporter::FrameExporter;

pub struct VideoExportProcessor {
    config: Config,
    frame_counter: usize,
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
            frame_counter: 0,
        })
    }

    fn get_frame_to_export<'a>(
        &self,
        frame_ctx: &'a FrameContext,
        ctx: &'a PipelineContext,
    ) -> &'a Frame {
        if let Some(key) = &self.config.key {
            // Try to get frame from pipeline context using the specified key
            if let Ok(frame) = ctx.try_get_as::<Frame>(key) {
                frame
            } else {
                tracing::warn!(
                    "Frame key '{}' not found in pipeline context, using current frame",
                    key
                );
                &frame_ctx.frame
            }
        } else {
            &frame_ctx.frame
        }
    }
}

impl FrameProcessor for VideoExportProcessor {
    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        let exporter = FrameExporter::new(&self.config.output_path, &self.config)?;
        ctx.put("exports/video", Arc::new(Mutex::new(exporter)));

        Ok(())
    }

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        // Check if we should start exporting
        if let Some(start_frame_index) = self.config.start_frame_index {
            if frame_ctx.frame_index < start_frame_index {
                return Continue;
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
                let mut exporter = ctx
                    .get_as::<Arc<Mutex<FrameExporter>>>("exports/video")
                    .lock()
                    .unwrap();
                if let Err(e) = exporter.close() {
                    tracing::error!("Failed to close video exporter: {}", e);
                }
            }

            if frame_ctx.frame_index > end_frame_index {
                return Continue;
            }
        }

        // Export the frame (ensure_open will be called automatically on first frame)
        let exporter = ctx.get_as::<Arc<Mutex<FrameExporter>>>("exports/video");
        let mut exporter = exporter.lock().unwrap();
        let frame_to_export = self.get_frame_to_export(frame_ctx, ctx);
        if let Err(e) = exporter.write_frame(frame_to_export) {
            tracing::error!(
                frame_index = frame_ctx.frame_index,
                "Failed to write frame: {}",
                e
            );
        }

        Continue
    }

    fn name(&self) -> &str {
        "selective_video_export_tee_processor"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

astrocap_core::register_astrocap_frame_processor!(VideoExportProcessor);
