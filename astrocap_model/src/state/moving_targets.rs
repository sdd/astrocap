use crate::state::{DetectedPoint, ModelState};
use argmin::core::ArgminFloat;
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::{Az, Cast};
use kiddo::float::kdtree::Axis;
use kiddo::SquaredEuclidean;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use nonmax::NonMaxUsize;
use num_traits::float::FloatCore;
use serde::{Deserialize, Serialize};
use std::iter::Sum;
use tracing::*;

const MAX_FRAME_DELTA: f64 = 100.0;
const MAX_MOVING_ITEM_JITTER: f64 = 20.0;
const MOVING_ITEM_LOOKBACK_WINDOW: usize = 8;

const MOVING_POINT_SCORE_THRESHOLD: f64 = 50f64;

const UNMATCHED_MOVING_TARGET_PENALTY: f64 = 8.0f64;

const MOVING_TARGET_DISCARD_THRESHOLD: f64 = 0f64;

const MIN_MOVEMENT_SPEED: f64 = 1f64;

#[derive(Debug, Serialize, Deserialize)]
pub struct MovingTarget<F: Axis> {
    pub log_odds: F,
    pub age: usize,
    pub latest_score: F,

    detected_point_match_history: Vec<Option<NonMaxUsize>>,

    pub last_frame_delta_x: F,
    pub last_frame_delta_y: F,

    pub x: F,
    pub y: F,
    // TODO:
    //  some kind of polynomial to represent the 2d path
    //  some kind of polynomial to fit the amplitude variation
    //  (eventually): Estimated Keplerian elements, assuming geocentric orbit
}

impl<F: Axis + ArgminFloat + Sum> ModelState<F>
where
    u32: Cast<F>,
    u8: Cast<F>,
    f64: Cast<F>,
    F: Cast<u32>,
    F: Cast<i32>,
    F: Cast<f32>,
    ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>: ArgminAdd<
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
    >,
    ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>: ArgminSub<
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
    >,
    ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>:
        ArgminMul<F, ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>>,
{
    pub(crate) fn update_moving_targets(&mut self) {
        let Some(curr_frame_state) = self.recent_frame_states.last() else {
            warn!("no frame state to update moving targets");
            return;
        };

        let mut discards = vec![];
        for (idx, moving_target) in self.moving_targets.iter_mut().enumerate() {
            moving_target.age += 1;

            // predict position of target in this frame
            moving_target.x += moving_target.last_frame_delta_x;
            moving_target.y += moving_target.last_frame_delta_y;

            let predicted_pos = [moving_target.x, moving_target.y];

            // fit against predicted position
            let best_match = curr_frame_state
                .detected_points_tree
                .nearest_one::<SquaredEuclidean>(&predicted_pos);

            let score = MAX_MOVING_ITEM_JITTER.az::<F>() - best_match.distance;
            moving_target.latest_score = score;

            if score > F::zero() {
                warn!(idx, score = score.az::<f32>(), "moving point matched");
                moving_target.log_odds += score;
            } else {
                warn!(idx, "mover penalized");
                moving_target.log_odds =
                    moving_target.log_odds - UNMATCHED_MOVING_TARGET_PENALTY.az::<F>();

                if moving_target.log_odds < MOVING_TARGET_DISCARD_THRESHOLD.az::<F>() {
                    warn!(idx, "discarded");
                    discards.push(idx);
                }
            }
        }
        for &idx in discards.iter().rev() {
            self.moving_targets.remove(idx);
        }
    }

    pub(crate) fn detect_moving_targets(&mut self) {
        let Some(curr_frame_state) = self.recent_frame_states.last() else {
            warn!("not enough frame states to detect moving targets");
            return;
        };

        let Some(prev_frame_state) = self.recent_frame_states.iter().rev().nth(1) else {
            warn!("not enough prev frame states to detect moving targets");
            return;
        };

        for (idx, detected_point) in curr_frame_state.detected_points_list.iter().enumerate() {
            if self.matched_point_indices.contains(&idx) {
                continue;
            }

            let curr_pos = [detected_point.x.az::<F>(), detected_point.y.az::<F>()];

            let prev_frame_matches = prev_frame_state
                .detected_points_tree
                .nearest_n_within::<SquaredEuclidean>(
                    &curr_pos,
                    MAX_FRAME_DELTA.az::<F>(),
                    usize::MAX,
                    false,
                );

            let mut best_score = <F as FloatCore>::neg_infinity();
            let mut best_delta = [F::zero(), F::zero()];

            let mut curr_frame_pos = [detected_point.x.az::<F>(), detected_point.y.az::<F>()];

            for prev_pos_candidate_res in prev_frame_matches {
                if prev_pos_candidate_res.distance < MIN_MOVEMENT_SPEED.az::<F>() {
                    continue;
                }

                let prev_pos_candidate: &DetectedPoint<F> =
                    &prev_frame_state.detected_points_list[prev_pos_candidate_res.item as usize];

                let frame_delta = [
                    prev_pos_candidate.x.az::<F>() - curr_frame_pos[0],
                    prev_pos_candidate.y.az::<F>() - curr_frame_pos[1],
                ];
                curr_frame_pos = [
                    prev_pos_candidate.x.az::<F>(),
                    prev_pos_candidate.y.az::<F>(),
                ];
                let mut score: F = F::zero();

                for lookback_frame_index in 2..MOVING_ITEM_LOOKBACK_WINDOW {
                    let Some(frame_state) = self
                        .recent_frame_states
                        .iter()
                        .rev()
                        .nth(lookback_frame_index)
                    else {
                        break;
                    };

                    let query_pos = [
                        curr_pos[0] + frame_delta[0] * (lookback_frame_index as u32).az::<F>(),
                        curr_pos[1] + frame_delta[1] * (lookback_frame_index as u32).az::<F>(),
                    ];

                    let frame_match = frame_state
                        .detected_points_tree
                        .nearest_one::<SquaredEuclidean>(&query_pos);

                    if frame_match.distance <= MAX_MOVING_ITEM_JITTER.az::<F>() {
                        score += MAX_MOVING_ITEM_JITTER.az::<F>() - frame_match.distance;
                        warn!(
                            "score: {:.2}, best_score: {:.2}",
                            score.az::<f32>(),
                            best_score.az::<f32>()
                        );

                        // TODO: update delta to delta from first to last frame
                    } else {
                        // info!("no match in frame {}", lookback_frame_index);
                    }
                }

                if score > best_score {
                    best_delta = frame_delta;
                    best_score = score;
                }
            }

            if best_score >= MOVING_POINT_SCORE_THRESHOLD.az::<F>() {
                self.moving_targets.push(MovingTarget {
                    log_odds: best_score,
                    latest_score: best_score,
                    age: 0,
                    detected_point_match_history: vec![],
                    last_frame_delta_x: F::zero() - best_delta[0],
                    last_frame_delta_y: F::zero() - best_delta[1],
                    x: curr_pos[0],
                    y: curr_pos[1],
                });
            }
        }
    }
}
