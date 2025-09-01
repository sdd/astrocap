use serde::{Deserialize, Serialize};
use std::fs::File;

use crate::config::Config;
use astrocap_core::frame::CpuFrame;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::structs::DetectedPoint;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameSource};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub amplitude: f32,
}

pub struct RecordedDetectionsSource {
    config: Config,
    frame_index: usize,
    detections: Vec<Vec<Detection>>,
}

impl RecordedDetectionsSource {
    pub fn new(params: Option<&toml::Value>) -> Result<Self, AstrocapError> {
        let config = match params {
            Some(params_value) => params_value.clone().try_into().map_err(|e| {
                AstrocapError::PluginInvalidConfigError(format!(
                    "RecordedDetectionsSource: {}",
                    e.to_string()
                ))
            })?,
            None => Config::default(),
        };

        // Load the detections from the JSON file
        let detections = Self::load_detections(&config.input_path)?;
        tracing::info!(
            "Loaded {} frames of detections from {:?}",
            detections.len(),
            config.input_path
        );

        Ok(Self {
            config,
            frame_index: 0,
            detections,
        })
    }

    fn load_detections(
        input_path: &std::path::PathBuf,
    ) -> Result<Vec<Vec<Detection>>, AstrocapError> {
        let file = File::open(input_path).map_err(|e| {
            AstrocapError::GeneralPluginError(format!(
                "Failed to open detections file {:?}: {}",
                input_path, e
            ))
        })?;

        let detections: Vec<Vec<Detection>> = serde_json::from_reader(file).map_err(|e| {
            AstrocapError::GeneralPluginError(format!(
                "Failed to parse detections from {:?}: {}",
                input_path, e
            ))
        })?;

        Ok(detections)
    }
}

impl FrameSource for RecordedDetectionsSource {
    fn next_frame(&mut self, _ctx: &mut PipelineContext) -> Option<FrameContext> {
        // Check if we've reached the end of our recorded detections
        if self.frame_index >= self.detections.len() {
            return None;
        }

        // Create a dummy frame
        let dummy_frame = Frame::Cpu(
            CpuFrame::from_vec(10, 10, vec![0; 100]).expect("Failed to create dummy frame"),
        );

        // Create frame context
        let mut frame_ctx = FrameContext::new(dummy_frame, self.frame_index);

        // Convert our Detection format to DetectedPoint format (what the pipeline expects)
        let detected_points: Vec<DetectedPoint> = self.detections[self.frame_index]
            .iter()
            .map(|detection| DetectedPoint {
                x: detection.x,
                y: detection.y,
                amplitude: detection.amplitude,
                fitted: None,
            })
            .collect();

        // Inject the detections into the frame context metadata
        // This is the same key that the point detector processor uses
        frame_ctx.put("detected_points", detected_points);

        self.frame_index += 1;

        Some(frame_ctx)
    }

    fn name(&self) -> &str {
        "recorded_detections_source"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
