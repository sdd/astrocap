use crate::config::PointExtractorConfig;
use crate::detectors::peak::PointDetectPeak;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::traits::{FrameProcessor, PointDetector};
use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use rerun::RecordingStream;
use serde::Serialize;
use std::sync::Arc;
use tracing::log::Log;

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
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        if frame_ctx.frame.as_cpu_image().is_none() {
            tracing::error!("Frame is not present");
            return FrameProcessorResult::Skip;
        };

        let detected_points_list = self.point_detector.detect(&frame_ctx.frame, None, None);

        if let Ok(ref rec) = ctx.get_as::<RecordingStream>("rerun") {
            rec.log(
                format!("model/detected_points"),
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
