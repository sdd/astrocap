pub mod point_handling;

pub use point_handling::StarCandidate;
use std::collections::HashSet;

use argmin_math::ArgminAdd;
use az::Az;
use std::error::Error;
use std::num::NonZero;
use std::sync::Arc;
use tracing::info;

use crate::config::ModelConfig;

use vyd::frame::CpuFrame;
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;
use astrocap_core::structs::DetectedPoint;
use astrocap_core::traits::{FrameProcessor, PointFitter};
use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use kiddo::{KdTree, SquaredEuclidean};
use nonmax::NonMaxUsize;
use num_traits::float::FloatCore;
use num_traits::real::Real;
use ordered_float::OrderedFloat;
use rerun::RecordingStream;

#[derive(Debug, Clone)]
pub struct FrameState {
    pub detected_points_list: Vec<DetectedPoint>,
    pub detected_points_tree: KdTree<f32, 2>,
}

#[derive(Debug)]
pub struct ModelState {
    model_config: ModelConfig,

    pub current_frame_state: Option<FrameState>,
    pub historical_frame_states: Vec<FrameState>,

    pub star_candidates: Vec<StarCandidate>,
    #[allow(dead_code)]
    star_candidates_tree: KdTree<f32, 2>,
    matched_point_indices: HashSet<usize>,
}

impl ModelState {
    pub fn new(config: Option<&toml::Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let model_config: ModelConfig = config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!("AstrocapProcessorModel: {}", e))
        })?;

        Ok(ModelState {
            model_config,
            current_frame_state: None,
            historical_frame_states: vec![],
            star_candidates: vec![],
            star_candidates_tree: KdTree::new(),
            matched_point_indices: HashSet::new(),
        })
    }
}

/*impl FrameState {
    pub fn from_frame(rec: Option<RecordingStream>) -> Self {
        let mut detected_points_list = PD::detect(frame, median, mask);

        // fit all detected points
        let fitted_points_list: Vec<_> = detected_points_list
            .iter()
            .map(|cand| point_fitter.fit(cand))
            .collect();

        // Assign the fitted points back to the detected points
        for (point, fitted_point) in detected_points_list
            .iter_mut()
            .zip(fitted_points_list.iter())
        {
            point.fitted_point = Some(fitted_point.clone());
        }

        // build a kd tree of all detected points
        let mut detected_points_tree: KdTree<f32, 2> =
            KdTree::with_capacity(detected_points_list.len());

        // track which detected points we're going to remove
        let mut detected_points_removed_index_list: Vec<_> =
            Vec::with_capacity(detected_points_list.len());

        // keep only the brightest point in the vicinity if there are points within
        // POINT_EXCLUSION_DIST of each other
        for (idx, point) in detected_points_list.iter().enumerate() {
            let query = [point.x, point.y];
            let mut near_neighbours = detected_points_tree.nearest_n_within::<SquaredEuclidean>(
                &query,
                POINT_EXCLUSION_DIST,
                NonZero::new(usize::MAX).unwrap(),
                false,
            );

            near_neighbours.sort_unstable_by_key(|nn| {
                OrderedFloat(detected_points_list[nn.item as usize].amplitude)
            });
            if let Some(brightest) = near_neighbours.pop() {
                let brightest_point = &detected_points_list[brightest.item as usize];
                if brightest_point.amplitude < point.amplitude {
                    detected_points_tree
                        .remove(&[brightest_point.x, brightest_point.y], brightest.item);
                    detected_points_removed_index_list.push(brightest.item);
                    detected_points_tree.add(&[point.x, point.y], idx);
                }
            } else {
                detected_points_tree.add(&[point.x, point.y], idx);
            }
            for close_point_result in near_neighbours {
                let point = &detected_points_list[close_point_result.item as usize];
                detected_points_tree.remove(&[point.x, point.y], close_point_result.item);
                detected_points_removed_index_list.push(close_point_result.item);
            }
        }

        // remove the points added to the list
        detected_points_removed_index_list.sort_unstable();
        for &idx in detected_points_removed_index_list.iter().rev() {
            detected_points_list.remove(idx as usize);
        }

        info!(detected_point_qty = detected_points_list.len());

        Self {
            detected_points_list,
            detected_points_tree,
        }
    }
}*/

impl FrameState {
    pub fn new(mut detected_points_list: Vec<DetectedPoint>, point_exclusion_dist: f32) -> Self {
        let mut detected_points_tree: KdTree<f32, 2> =
            KdTree::with_capacity(detected_points_list.len());

        let mut detected_points_removed_index_list: Vec<_> =
            Vec::with_capacity(detected_points_list.len());

        for (idx, point) in detected_points_list.iter().enumerate() {
            let query = [point.x, point.y];
            let mut near_neighbours = detected_points_tree.nearest_n_within::<SquaredEuclidean>(
                &query,
                point_exclusion_dist,
                NonZero::new(usize::MAX).unwrap(),
                false,
            );

            near_neighbours.sort_unstable_by_key(|nn| {
                OrderedFloat(detected_points_list[nn.item as usize].amplitude)
            });
            if let Some(brightest) = near_neighbours.pop() {
                let brightest_point = &detected_points_list[brightest.item as usize];
                if brightest_point.amplitude < point.amplitude {
                    detected_points_tree
                        .remove(&[brightest_point.x, brightest_point.y], brightest.item);
                    detected_points_removed_index_list.push(brightest.item);
                    detected_points_tree.add(&[point.x, point.y], idx as u64);
                }
            } else {
                detected_points_tree.add(&[point.x, point.y], idx as u64);
            }
            for close_point_result in near_neighbours {
                let point = &detected_points_list[close_point_result.item as usize];
                detected_points_tree.remove(&[point.x, point.y], close_point_result.item);
                detected_points_removed_index_list.push(close_point_result.item);
            }
        }

        detected_points_removed_index_list.sort_unstable();
        for &idx in detected_points_removed_index_list.iter().rev() {
            detected_points_list.remove(idx as usize);
        }

        info!(
            detected_point_qty = detected_points_list.len(),
            removed_point_qty = detected_points_removed_index_list.len(),
            tree_size = detected_points_tree.size(),
        );

        Self {
            detected_points_list,
            detected_points_tree,
        }
    }
}

impl FrameProcessor for ModelState {
    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        let tracked_objects: Vec<StarCandidate> = vec![];

        ctx.put("tracked_objects", tracked_objects);

        Ok(())
    }

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        // pull detected_points from frame context
        let Ok(detected_points_list) =
            frame_ctx.try_get_as::<Vec<DetectedPoint>>("detected_points")
        else {
            tracing::error!("Detected points not found");
            return FrameProcessorResult::Skip;
        };

        // create a frame state from detected points
        let new_frame_state = FrameState::new(
            detected_points_list.clone(),
            self.model_config.point_exclusion_radius,
        );

        // Move current frame to history (if it exists)
        if let Some(current) = self.current_frame_state.take() {
            self.historical_frame_states.push(current);
        }

        // Set new current frame
        self.current_frame_state = Some(new_frame_state);

        // pull tracked_objects from pipeline context
        let Ok(tracked_objects) = ctx.try_get_as::<Vec<StarCandidate>>("tracked_objects") else {
            tracing::error!("Tracked objects not found");
            return FrameProcessorResult::Skip;
        };

        // pull rerun from pipeline context
        let rec = ctx.try_get_as::<RecordingStream>("rerun").ok();

        // pull rerun from pipeline context
        let Ok(point_fitter) = ctx.try_get_as::<Arc<dyn PointFitter>>("point_fitter") else {
            tracing::error!("Point fitter not found");
            return FrameProcessorResult::Skip;
        };

        // set them on self
        self.star_candidates = tracked_objects.clone();

        let frame = frame_ctx.frame.as_cpu_frame().unwrap();

        tracing::debug!(
            curr_frame_state_len = self
                .current_frame_state
                .as_ref()
                .unwrap()
                .detected_points_list
                .len(),
            star_candidates_len = self.star_candidates.len(),
        );

        if let Err(err) = self.process_frame(frame, point_fitter.clone(), rec) {
            tracing::error!("Error processing frame: {}", err);
            return FrameProcessorResult::Skip;
        }

        ctx.put("tracked_objects", self.star_candidates.clone());

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "model_v2"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

impl ModelState {
    pub fn process_frame(
        &mut self,
        frame: &CpuFrame,
        point_fitter: Arc<dyn PointFitter>,
        rec: Option<&RecordingStream>,
    ) -> Result<(), Box<dyn Error>> {
        self.update_state_positions(self.model_config.disable_kalman_velocity);

        // Now fit existing points and get match information
        let (matched_point_indices, candidate_matches) =
            self.fit_existing_points(frame, point_fitter.clone());
        self.matched_point_indices = matched_point_indices;

        // Update the detected_point_match_history after frame has been moved to history
        for (candidate_idx, detected_point_idx_opt) in candidate_matches {
            if let Some(cand) = self.star_candidates.get_mut(candidate_idx) {
                if let Some(detected_point_idx) = detected_point_idx_opt {
                    cand.detected_point_match_history
                        .push(Some(NonMaxUsize::new(detected_point_idx).unwrap()));
                } else {
                    cand.detected_point_match_history.push(None);
                }
            }
        }

        // Now fit strong unmatched new points to create new star candidates
        if let Some(ref current_frame_state) = self.current_frame_state {
            self.fit_strong_unmatched_new_points(point_fitter, frame);
        }

        // Clean up poor candidates
        self.clean_up_state();

        // Log to rerun if available
        if let Some(rec) = rec {
            self.log_to_rerun(rec);
        } else {
            info!("No rerun available");
        }

        Ok(())
    }

    fn update_state_positions(&mut self, disable_movement: bool) {
        for candidate in self.star_candidates.iter_mut() {
            if candidate.kalman_initialized {
                candidate.kalman_predict(disable_movement);
            }
        }

        self.star_candidates_tree = KdTree::with_capacity(self.star_candidates.len());

        for (cand_idx, cand) in self.star_candidates.iter_mut().enumerate() {
            self.star_candidates_tree
                .add(&[cand.x, cand.y], cand_idx as u64);
        }
    }

    fn update_state(&mut self) {
        let Some(ref curr_frame_state) = self.current_frame_state else {
            return;
        };

        let mut max_score = f32::MIN;
        let mut min_score = f32::MAX;

        // reset the star candidate tree so that positions are updated
        // before the next frame
        self.star_candidates_tree = KdTree::with_capacity(self.star_candidates.len());

        for (cand_idx, cand) in self.star_candidates.iter_mut().enumerate() {
            cand.age += 1;

            if let Some(last) = cand.detected_point_match_history.last() {
                if let Some(fitted_point_idx) = last {
                    // matched this frame
                    let point = &curr_frame_state.detected_points_list[fitted_point_idx.get()];

                    let score = point.fitted.as_ref().unwrap().score;
                    max_score = FloatCore::max(max_score, score);
                    min_score = FloatCore::min(min_score, score);
                    cand.log_likelihood += score;

                    if (point.amplitude as f32) < self.model_config.amplitude_penalty_threshold {
                        cand.log_likelihood -= self.model_config.amplitude_penalty;
                    }
                } else {
                    // no match this frame
                    cand.log_likelihood -= self.model_config.star_candidate_unmatched_penalty;
                }
            }

            self.star_candidates_tree
                .add(&[cand.x, cand.y], cand_idx as u64);
        }

        info!(min_score, max_score);
    }

    fn clean_up_state(&mut self) {
        let pre_discard_count = self.star_candidates.len();
        self.star_candidates = self
            .star_candidates
            .clone()
            .into_iter()
            .filter(|cand| {
                let unmatched_frames = cand
                    .detected_point_match_history
                    .iter()
                    .rev()
                    .take_while(|&m| m.is_none())
                    .count();

                cand.log_likelihood > self.model_config.star_candidate_discard_threshold
                // && (unmatched_frames < 50)
            })
            .collect();

        info!(
            state_len = self.star_candidates.len(),
            discarded = pre_discard_count - self.star_candidates.len(),
        );
    }

    fn log_to_rerun(&mut self, rec: &RecordingStream) {
        rec.log(
            "model/star_candidates".to_string(),
            &rerun::Points2D::new(
                self.star_candidates
                    .iter()
                    .filter(|cand| {
                        cand.log_likelihood
                            > self.model_config.star_candidate_long_term_match_threshold
                    })
                    .map(|cand| (cand.x.az::<f32>(), cand.y.az::<f32>())),
            )
            .with_colors(
                self.star_candidates
                    .iter()
                    .filter(|cand| {
                        cand.log_likelihood
                            > self.model_config.star_candidate_long_term_match_threshold
                    })
                    .map(|cand| {
                        let log_likelihood = cand.log_likelihood.az::<f32>();
                        let hue = (log_likelihood.min(1000.0) / 1000.0) * 270.0; // 0° = red, 270° = violet
                        let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
                        rerun::Color::from_rgb(r, g, b)
                    }),
            )
            .with_labels(
                self.star_candidates
                    .iter()
                    .filter(|cand| {
                        cand.log_likelihood
                            > self.model_config.star_candidate_long_term_match_threshold
                    })
                    .map(|cand| {
                        let unmatched_frames = cand
                            .detected_point_match_history
                            .iter()
                            .rev()
                            .take_while(|&m| m.is_none())
                            .count();

                        format!(
                            "ID: {}, Age: {}, LL: {}, unmatched: {}",
                            cand.id, cand.age, cand.log_likelihood, unmatched_frames
                        )
                    }),
            ),
        )
        .unwrap()
    }
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h % 360.0; // Wrap hue to [0, 360)
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);

    let c = v * s; // Chroma
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r_prime, g_prime, b_prime) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        300..=359 => (c, 0.0, x),
        _ => (0.0, 0.0, 0.0), // Should never happen due to modulo
    };

    let r = ((r_prime + m) * 255.0).round() as u8;
    let g = ((g_prime + m) * 255.0).round() as u8;
    let b = ((b_prime + m) * 255.0).round() as u8;

    (r, g, b)
}
