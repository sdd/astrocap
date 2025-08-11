use crate::config::PointExtractorConfig;
use crate::detectors::peak::PointDetectPeak;
use crate::PointDetector;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::{AstrocapError, FrameContext, FrameProcessor, FrameProcessorResult};
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize)]
pub struct DetectedPoint {
    pub x: u32,
    pub y: u32,
    pub amplitude: u8,
    pub fitted: Option<FittedPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FittedPoint {
    pub x: f32,
    pub y: f32,
    pub amplitude: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub fit_quality: FittedPointQuality,
}

#[derive(Debug, Clone, Serialize)]
pub struct FittedPointQuality {
    pub reduced_chi_squared: f32,
    pub snr: f32,
    pub r_squared: f32,
    pub rms_residual: f32,
    pub score: f32,
}

pub struct PointDetectorProcessor {
    point_detector: Arc<dyn PointDetector>,
}

impl PointDetectorProcessor {
    pub fn new(config: Option<&toml::Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let _config: PointExtractorConfig = config
            .clone()
            .try_into()
            .map_err(|_| AstrocapError::PluginInvalidConfigError)?;

        let point_detector: Arc<dyn PointDetector> = Arc::new(PointDetectPeak {});

        Ok(Self { point_detector })
    }
}

impl FrameProcessor for PointDetectorProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        _ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        if frame_ctx.frame.as_cpu_image().is_none() {
            tracing::error!("Frame is not present");
            return FrameProcessorResult::Skip;
        };

        let detected_points_list = self.point_detector.detect(&frame_ctx.frame, None, None);

        frame_ctx.metadata.insert(
            "detected_points".to_string(),
            Box::new(detected_points_list),
        );

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "point_extractor"
    }
}
