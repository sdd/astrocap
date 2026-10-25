use crate::config::PointExtractorConfig;
use crate::detectors::adaptive_centroid::PointDetectAdaptiveCentroid;
use crate::detectors::local_maxima::PointDetectLocalMaxima;
use crate::detectors::peak::PointDetectPeak;
use astrocap_core::frame_metadata::{DetectedPoints, MedianFrame};
use astrocap_core::pipeline_services::MaskFrame;
use astrocap_core::pipeline_services::RerunRecording;
use astrocap_core::structs::Detection;
use astrocap_core::traits::{FrameProcessor, PointDetector};
use astrocap_core::{AstrocapError, DumpManager, Frame, FrameContext, FrameProcessorResult};
use std::sync::Arc;
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

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

        let mask = match ctx.services().latest::<MaskFrame>() {
            Ok(mask) => mask,
            Err(error) => {
                tracing::error!(%error, "mask service lookup failed");
                return FrameProcessorResult::Skip;
            }
        };
        // A mask is a persistent pipeline service, not per-frame metadata.
        // A native geometry change cannot rewrite it through the frame hook.
        if let Some(mask) = &mask {
            if mask.0.dimensions() != frame_ctx.frame.dimensions() {
                tracing::error!(
                    mask = ?mask.0.dimensions(),
                    frame = ?frame_ctx.frame.dimensions(),
                    "mask dimensions do not match native output frame"
                );
                return FrameProcessorResult::Skip;
            }
        }
        let median = frame_ctx
            .get_typed::<MedianFrame>()
            .map(|median| median.0.clone())
            .or_else(|| {
                frame_ctx
                    .try_get_as::<Arc<Frame>>("video/median")
                    .ok()
                    .cloned()
            });

        let detected_points_list =
            self.point_detector
                .detect(&frame_ctx.frame, median, mask.as_ref().map(|mask| &mask.0));

        match ctx.services().latest::<RerunRecording>() {
            Ok(Some(rec)) => {
                rec.0
                    .log(
                        "model/detected_points".to_string(),
                        &rerun::Points2D::new(
                            detected_points_list
                                .iter()
                                .map(|cand| (cand.position.x, cand.position.y)),
                        ),
                    )
                    .unwrap();
            }
            Ok(None) => {}
            Err(error) => {
                tracing::error!(%error, "Rerun service lookup failed");
                return FrameProcessorResult::Skip;
            }
        }

        // Get the dump manager from the pipeline context
        let entry = ctx.entry("dump_manager");
        if let std::collections::hash_map::Entry::Occupied(mut entry) = entry {
            let dump_manager = entry
                .get_mut()
                .downcast_mut::<DumpManager>()
                .expect("dump_manager is not a DumpManager");

            let result = dump_manager.dumper::<Detection>().and_then(|dumper| {
                dumper
                    .lock()
                    .map_err(|e| anyhow::anyhow!("dumper lock poisoned: {e}"))?
                    .dumps(detected_points_list.iter(), frame_ctx.frame_index)
            });
            if let Err(error) = result {
                tracing::warn!(?error, "Failed to dump detections");
            }
        }

        frame_ctx.put_typed(DetectedPoints(detected_points_list));

        FrameProcessorResult::Continue
    }

    fn try_process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, AstrocapError> {
        if let Some(mask) = ctx
            .services()
            .latest::<MaskFrame>()
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
        {
            if mask.0.dimensions() != frame_ctx.frame.dimensions() {
                return Err(AstrocapError::GeneralPluginError(format!(
                    "mask dimensions do not match native output frame: {:?} versus {:?}",
                    mask.0.dimensions(),
                    frame_ctx.frame.dimensions()
                )));
            }
        }
        Ok(self.process(frame_ctx, ctx))
    }

    fn name(&self) -> &str {
        "point_extractor"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

#[cfg(test)]
mod native_mask_tests {
    use super::*;
    use astrocap_core::structs::Detection;

    struct NeverCalled;
    impl PointDetector for NeverCalled {
        fn detect(
            &mut self,
            _: &Frame,
            _: Option<Arc<Frame>>,
            _: Option<&Frame>,
        ) -> Vec<Detection> {
            panic!("stale mask reached detector");
        }
    }

    #[test]
    fn stale_pipeline_mask_is_rejected_after_native_scale() {
        let mut context = PipelineContext::new();
        context
            .services()
            .publish(Arc::new(MaskFrame(
                Frame::from_raw(4, 2, vec![255; 8]).unwrap(),
            )))
            .unwrap();
        let mut frame = FrameContext::new(Frame::from_raw(8, 2, vec![9; 16]).unwrap(), 0);
        let mut processor = PointDetectorProcessor {
            point_detector: Box::new(NeverCalled),
        };
        let error = processor
            .try_process(&mut frame, &mut context)
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("mask dimensions do not match native output frame"));
        assert!(frame.get_typed::<DetectedPoints>().is_none());
    }
}
