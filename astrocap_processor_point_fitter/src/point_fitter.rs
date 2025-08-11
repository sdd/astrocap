use crate::config::PointFitterConfig;
use crate::fitters::nelder_mead::PointFitterGaussianNelderMead;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::structs::{DetectedPoint, FittedPoint, FittedPointQuality};
use astrocap_core::traits::{FrameProcessor, PointFitter};
use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use rerun::RecordingStream;
use serde::Serialize;
use std::sync::Arc;

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
            .map_err(|_| AstrocapError::PluginInvalidConfigError)?;

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

        let Ok(detected_points_list) = frame_ctx.get_as::<Vec<DetectedPoint>>("detected_points")
        else {
            return FrameProcessorResult::Skip;
        };

        let fitted_points_list: Vec<FittedPoint> = detected_points_list
            .iter()
            .map(|detected_point| self.point_fitter.fit(frame, detected_point))
            .collect();

        if let Ok(rec) = ctx.get_as::<RecordingStream>("rerun") {
            rec.log(
                format!("model/fitted_points"),
                &rerun::Points2D::new(
                    detected_points_list
                        .iter()
                        .map(|cand| (cand.x as f32, cand.y as f32)),
                ),
            )
            .unwrap();
        }

        frame_ctx.put("fitted_points", fitted_points_list);

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "point_fitter"
    }
}
