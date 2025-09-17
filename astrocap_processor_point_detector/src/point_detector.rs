use crate::config::PointExtractorConfig;
use crate::detectors::adaptive_centroid::PointDetectAdaptiveCentroid;
use crate::detectors::local_maxima::PointDetectLocalMaxima;
use crate::detectors::peak::PointDetectPeak;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::structs::Detection;
use astrocap_core::traits::{FrameProcessor, PointDetector};
use astrocap_core::{AstrocapError, DumpManager, Frame, FrameContext, FrameProcessorResult};
use rerun::RecordingStream;
use std::sync::Arc;

pub struct PointDetectorProcessor {
    point_detector: Box<dyn PointDetector>,
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

        let point_detector: Box<dyn PointDetector> = match config {
            PointExtractorConfig::Peak(config) => Box::new(PointDetectPeak { config }),
            PointExtractorConfig::LocalMaxima(config) => Box::new(PointDetectLocalMaxima {
                config,
                sigma: None,
            }),
            PointExtractorConfig::AdaptiveCentroid(config) => {
                Box::new(PointDetectAdaptiveCentroid { config })
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

        let mask = ctx.try_get_as::<Frame>("image/mask").ok();
        let median = frame_ctx
            .try_get_as::<Arc<Frame>>("video/median")
            .ok()
            .map(|f| f.clone());

        let detected_points_list = self.point_detector.detect(&frame_ctx.frame, median, mask);

        if let Ok(rec) = ctx.try_get_as::<RecordingStream>("rerun") {
            rec.log(
                "model/detected_points".to_string(),
                &rerun::Points2D::new(
                    detected_points_list
                        .iter()
                        .map(|cand| (cand.position.x, cand.position.y)),
                ),
            )
            .unwrap();
        }

        // Get the dump manager from the pipeline context
        let entry = ctx.entry("dump_manager");
        if let std::collections::hash_map::Entry::Occupied(mut entry) = entry {
            let dump_manager = entry
                .get_mut()
                .downcast_mut::<DumpManager>()
                .expect("dump_manager is not a DumpManager");

            dump_manager
                .dumper::<Detection>()
                .lock()
                .unwrap()
                .dumps(detected_points_list.iter(), frame_ctx.frame_index);
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

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
