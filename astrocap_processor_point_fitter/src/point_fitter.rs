use crate::config::PointFitterConfig;
use crate::fitters::nelder_mead::PointFitterGaussianNelderMead;
use astrocap_core::frame_metadata::{DetectedPoints, FittedPoints};
use astrocap_core::pipeline_services::RerunRecording;
use astrocap_core::structs::FittedPoint;
use astrocap_core::traits::{FrameProcessor, PointFitter};
use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use std::sync::Arc;
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

pub struct PointFitterProcessor {
    point_fitter: Arc<dyn PointFitter>,
}

impl PointFitterProcessor {
    pub fn new(config: Option<&toml::Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let _config: PointFitterConfig = config
            .clone()
            .try_into()
            .map_err(|e| AstrocapError::PluginInvalidConfigError(e.to_string()))?;

        let point_fitter: Arc<dyn PointFitter> = Arc::new(PointFitterGaussianNelderMead {});

        Ok(Self { point_fitter })
    }
}

impl FrameProcessor for PointFitterProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let Some(frame) = frame_ctx.frame.as_cpu_frame() else {
            tracing::error!("Frame is not present");
            return FrameProcessorResult::Skip;
        };

        let Some(DetectedPoints(detected_points_list)) = frame_ctx.get_typed::<DetectedPoints>()
        else {
            return FrameProcessorResult::Skip;
        };

        let fitted_points_list: Vec<FittedPoint> = detected_points_list
            .iter()
            .map(|detected_point| self.point_fitter.fit(frame, detected_point))
            .collect();

        match ctx.services().latest::<RerunRecording>() {
            Ok(Some(rec)) => {
                rec.0
                    .log(
                        "model/fitted_points".to_string(),
                        &rerun::Points2D::new(
                            detected_points_list
                                .iter()
                                .map(|cand| (cand.position[0], cand.position[1])),
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

        frame_ctx.put_typed(FittedPoints(fitted_points_list));

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "point_fitter"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
