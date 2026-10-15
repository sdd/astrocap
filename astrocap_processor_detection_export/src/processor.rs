use astrocap_core::frame_metadata::DetectedPoints;
use astrocap_core::structs::Detection;
use astrocap_core::{AstrocapError, FrameContext, FrameProcessor, FrameProcessorResult};
use std::fs::File;
use std::io::{BufWriter, Write};
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
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, &self.detections).map_err(|_| {
            AstrocapError::GeneralPluginError("Could not write detections to file".to_string())
        })?;
        writer.flush().map_err(|_| {
            AstrocapError::GeneralPluginError("Could not flush detections to file".to_string())
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
        let frame_detections = if let Some(detected_points) =
            frame_ctx.get_typed::<DetectedPoints>()
        {
            detected_points
                .0
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

#[cfg(test)]
mod tests {
    use super::*;
    use astrocap_core::Frame;
    use astrocap_core::frame_metadata::DetectedPoints;

    #[test]
    fn exporter_reads_typed_detector_output() {
        let mut processor = DetectionExportProcessor::new(None).unwrap();
        let mut frame = FrameContext::new(Frame::from_raw(1, 1, vec![0]).unwrap(), 0);
        frame.put_typed(DetectedPoints(vec![Detection::new(2.0, 3.0, 4.0)]));
        let mut context = PipelineContext::new();
        assert!(matches!(
            processor.process(&mut frame, &mut context),
            FrameProcessorResult::Continue
        ));
        assert_eq!(processor.detections.len(), 1);
        assert_eq!(processor.detections[0].len(), 1);
        assert_eq!(processor.detections[0][0].position.x, 2.0);
        assert_eq!(processor.detections[0][0].amplitude, 4.0);
    }
}
