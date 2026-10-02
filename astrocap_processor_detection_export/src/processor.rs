use astrocap_core::structs::Detection;
use astrocap_core::{AstrocapError, FrameContext, FrameProcessor, FrameProcessorResult};
use std::fs::File;
use tracing::{debug, error, info};
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

use crate::config::Config;

use serde::{Deserialize, Serialize};

pub struct DetectionExportProcessor {
    config: Config,
    detections: Vec<Vec<Detection>>,
}

impl DetectionExportProcessor {
    pub fn new(config: Option<&toml::Value>) -> Result<Self, AstrocapError> {
        let config = match config {
            Some(config_value) => config_value.clone().try_into().map_err(|e| {
                AstrocapError::PluginInvalidConfigError(format!(
                    "DetectionExportProcessor: {}",
                    e.to_string()
                ))
            })?,
            None => Config::default(),
        };

        Ok(Self {
            config,
            detections: Vec::new(),
        })
    }

    fn export_detections(&self) -> Result<(), AstrocapError> {
        let file = File::create(&self.config.output_path).map_err(|_| {
            AstrocapError::GeneralPluginError(
                "Could not open file for detection export".to_string(),
            )
        })?;
        serde_json::to_writer_pretty(file, &self.detections).map_err(|_| {
            AstrocapError::GeneralPluginError("Could not write detections to file".to_string())
        })?;
        info!(
            "Exported {} frames of detections to {:?}",
            self.detections.len(),
            self.config.output_path
        );
        Ok(())
    }
}

impl FrameProcessor for DetectionExportProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        _ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let frame_detections = if let Ok(detected_points) =
            frame_ctx.try_get_as::<Vec<Detection>>("detected_points")
        {
            detected_points
                .iter()
                .map(|point| Detection::new(point.position.x, point.position.y, point.amplitude))
                .collect()
        } else {
            Vec::new()
        };

        self.detections.push(frame_detections);
        debug!(
            "Exported {} detections from frame {}",
            self.detections.last().unwrap().len(),
            frame_ctx.frame_index
        );

        FrameProcessorResult::Continue
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        if let Err(e) = self.export_detections() {
            error!("Failed to export detections: {}", e);
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "detection_export"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
