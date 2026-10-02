use arc_swap::ArcSwapOption;
use rerun::components::ShowLabels;
use rerun::RecordingStream;
use solvastro::dynamic_index::verified_solution::{ItemMatchResult, VerifiedSolution};
use std::sync::Arc;
use toml::Value;

use astrocap_core::traits::{FrameProcessor, StarCandidate};
use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

use crate::config::SolvastroProcessorConfig;
use crate::solvastro_adapter::SolvastroAdapter;

pub struct SolvastroProcessor {
    config: SolvastroProcessorConfig,
    star_candidates: Arc<ArcSwapOption<Vec<StarCandidate>>>,
    solution: Arc<ArcSwapOption<VerifiedSolution>>,
    solvastro_adapter: SolvastroAdapter,
}

impl SolvastroProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            tracing::error!("Missing config");
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: SolvastroProcessorConfig = config.clone().try_into().map_err(|e| {
            tracing::error!(?e, "config error");
            AstrocapError::PluginInvalidConfigError(format!(
                "SolvastroProcessor:: {}",
                e.to_string()
            ))
        })?;

        let star_candidates = Arc::new(ArcSwapOption::from(None));
        let solution = Arc::new(ArcSwapOption::from(None));

        let solvastro_adapter =
            SolvastroAdapter::new(config.clone(), star_candidates.clone(), solution.clone())?;

        Ok(Self {
            config,
            star_candidates,
            solution,
            solvastro_adapter,
        })
    }
}

impl FrameProcessor for SolvastroProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let Some(dims) = frame_ctx.frame.dimensions() else {
            tracing::warn!("Could not get frame dimensions");
            return FrameProcessorResult::Skip;
        };

        let Ok(star_candidates) =
            frame_ctx.try_get_as::<Arc<Vec<StarCandidate>>>("solvastro/star_candidates")
        else {
            tracing::warn!("No candidates");
            return FrameProcessorResult::Continue;
        };

        self.star_candidates.store(Some(star_candidates.clone()));
        self.solvastro_adapter.notify();

        if let Some(solution) = self.solution.load().as_ref() {
            tracing::info!("Found solution");
            frame_ctx.put("solvastro/solution", solution.clone());

            if let Ok(rec) = ctx.try_get_as::<RecordingStream>("rerun") {
                self.log_solution_to_rerun(solution, rec);
            } else {
                tracing::error!("could not get a ref to rerun");
            }
        }

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "solvastro_processor"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

impl SolvastroProcessor {
    fn log_solution_to_rerun(&self, solution: &Arc<VerifiedSolution>, rec: &RecordingStream) {
        let matches: Vec<_> = solution
            .matches
            .iter()
            .filter_map(|m| match m {
                ItemMatchResult::Match {
                    index_x,
                    index_y,
                    index_name,
                    ..
                } => Some((index_x, index_y, index_name)),
                _ => None,
            })
            .collect();

        rec.log(
            "solver/matches".to_string(),
            &rerun::Boxes2D::from_mins_and_sizes(
                std::iter::repeat((10.0, 10.0)).take(matches.len()),
                std::iter::repeat((10.0, 10.0)).take(matches.len()),
            )
            .with_centers(matches.iter().map(|m| (*m.0 as f32, *m.1 as f32)))
            .with_colors(
                matches
                    .iter()
                    .map(|track| rerun::Color::from_rgb(0, 127, 255)),
            )
            .with_labels(matches.iter().map(|m| m.2.clone()))
            .with_show_labels(true),
        )
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to log tracks to rerun: {}", e);
        });
    }
}
