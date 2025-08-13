use crate::config::PointExtractorConfig;
use crate::detectors::adaptive_centroid::PointDetectAdaptiveCentroid;
use crate::detectors::local_maxima::PointDetectLocalMaxima;
use crate::detectors::peak::PointDetectPeak;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::traits::{FrameProcessor, PointDetector};
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use rerun::RecordingStream;
use std::sync::Arc;

pub struct PointDetectorProcessor {
    point_detector: Arc<dyn PointDetector>,
}

impl PointDetectorProcessor {
    pub fn new(config: Option<&toml::Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: PointExtractorConfig = config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!(
                "PointDetectorProcessor: {}",
                e.to_string()
            ))
        })?;

        let point_detector: Arc<dyn PointDetector> = match config {
            PointExtractorConfig::Peak(config) => Arc::new(PointDetectPeak { config }),
            PointExtractorConfig::LocalMaxima(config) => {
                Arc::new(PointDetectLocalMaxima { config })
            }
            PointExtractorConfig::AdaptiveCentroid(config) => {
                Arc::new(PointDetectAdaptiveCentroid { config })
            }
        };

        Ok(Self { point_detector })
    }
}

impl FrameProcessor for PointDetectorProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        if frame_ctx.frame.as_cpu_image().is_none() {
            tracing::error!("Frame is not present");
            return FrameProcessorResult::Skip;
        };

        let mask = ctx.get_as::<Frame>("image/mask").ok();
        let median = frame_ctx
            .get_as::<Arc<Frame>>("video/median")
            .ok()
            .map(|f| f.clone());

        let detected_points_list = self.point_detector.detect(&frame_ctx.frame, median, mask);

        if let Ok(rec) = ctx.get_as::<RecordingStream>("rerun") {
            rec.log(
                "model/detected_points".to_string(),
                &rerun::Points2D::new(
                    detected_points_list
                        .iter()
                        .map(|cand| (cand.x as f32, cand.y as f32)),
                ),
            )
            .unwrap();
        }

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
