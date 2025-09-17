use crate::config::PointFitterConfig;
use crate::fitters::nelder_mead::PointFitterGaussianNelderMead;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::structs::{Detection, FittedPoint};
use astrocap_core::traits::{FrameProcessor, PointFitter};
use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use rerun::RecordingStream;
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
            .map_err(|e| AstrocapError::PluginInvalidConfigError(e.to_string()))?;

        let point_fitter: Arc<dyn PointFitter> = Arc::new(PointFitterGaussianNelderMead {});

        Ok(Self { point_fitter })
    }
}

impl FrameProcessor for PointFitterProcessor {
    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        ctx.put("point_fitter", self.point_fitter.clone());

        Ok(())
    }
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let Some(frame) = frame_ctx.frame.as_cpu_frame() else {
            tracing::error!("Frame is not present");
            return FrameProcessorResult::Skip;
        };

        let Ok(detected_points_list) = frame_ctx.try_get_as::<Vec<Detection>>("detected_points")
        else {
            return FrameProcessorResult::Skip;
        };

        let fitted_points_list: Vec<FittedPoint> = detected_points_list
            .iter()
            .map(|detected_point| self.point_fitter.fit(frame, detected_point))
            .collect();

        if let Ok(rec) = ctx.try_get_as::<RecordingStream>("rerun") {
            rec.log(
                "model/fitted_points".to_string(),
                &rerun::Points2D::new(
                    detected_points_list
                        .iter()
                        .map(|cand| (cand.position[0], cand.position[1])),
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

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
