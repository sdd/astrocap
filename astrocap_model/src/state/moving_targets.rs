use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::{Az, Cast};

use kiddo::{KdTree, SquaredEuclidean};
use nalgebra::Vector2;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use nonmax::NonMaxUsize;
use num_traits::float::FloatCore;
use serde::Serialize;

use tracing::*;

use crate::state::{DetectedPoint, ModelState};
use crate::traits::AstroFloat;

const MAX_FRAME_DELTA: f64 = 100.0;
const MAX_MOVING_ITEM_JITTER: f64 = 20.0;
const MOVING_ITEM_LOOKBACK_WINDOW: usize = 8;

const MOVING_POINT_SCORE_THRESHOLD: f64 = 50f64;

const UNMATCHED_MOVING_TARGET_PENALTY: f64 = 8.0f64;

const MOVING_TARGET_DISCARD_THRESHOLD: f64 = 0f64;

const MIN_MOVEMENT_SPEED: f64 = 1f64;

const MOVING_ITEM_EXCLUSION_DIST2: f64 = 25f64;

const VELOCITY_UPDATE_RATE: f64 = 0.1f64;

#[derive(Debug, Serialize)]
pub struct MovingTarget<F: AstroFloat + 'static> {
    pub log_odds: F,
    pub age: usize,
    pub latest_score: F,

    detected_point_match_history: Vec<Option<NonMaxUsize>>,

    pub position: Vector2<F>,
    pub velocity: Vector2<F>,
    pub pos_history: Vec<Vector2<F>>,
    // TODO:
    //  some kind of polynomial to represent the 2d path
    //  some kind of polynomial to fit the amplitude variation
    //  (eventually): Estimated Keplerian elements, assuming geocentric orbit
}

impl<F: AstroFloat> ModelState<F>
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
            moving_target.position += moving_target.velocity;

            moving_target.pos_history.push(moving_target.position);

            let predicted_pos = moving_target.position;

            // fit against predicted position
            let best_match = curr_frame_state
                .detected_points_tree
                .nearest_one::<SquaredEuclidean>(predicted_pos.as_ref());

            let score = MAX_MOVING_ITEM_JITTER.az::<F>() - best_match.distance;
            moving_target.latest_score = score;

            if score > F::zero() {
                warn!(idx, score = score.az::<f32>(), "moving point matched");
                moving_target.log_odds += score;

                // Update velocity
                let matched_item = &curr_frame_state.detected_points_list[best_match.item as usize];
                let actual_pos = Vector2::new(matched_item.x.az::<F>(), matched_item.y.az::<F>());

                let pos_error = actual_pos - predicted_pos;

                moving_target.velocity += pos_error * VELOCITY_UPDATE_RATE.az::<F>();
                warn!(?pos_error, ?moving_target.velocity);
            } else {
                warn!(idx, "mover penalized");
                moving_target.log_odds -= UNMATCHED_MOVING_TARGET_PENALTY.az::<F>();

                if moving_target.log_odds < MOVING_TARGET_DISCARD_THRESHOLD.az::<F>() {
                    warn!(idx, "discarded");
                    discards.push(idx);
                }
            }
        }
        for &idx in discards.iter().rev() {
            self.moving_targets.remove(idx);
        }

        self.moving_targets_tree = KdTree::new();
        for (idx, mt) in self.moving_targets.iter().enumerate() {
            self.moving_targets_tree
                .add(mt.position.as_ref(), idx as u64);
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
            // don't add any MTs that correspond to points already matched
            // to star candidates
            if self.matched_point_indices.contains(&idx) {
                continue;
            }

            let latest_position =
                Vector2::new(detected_point.x.az::<F>(), detected_point.y.az::<F>());

            let pos_history = vec![latest_position];

            // don't add any MTs that are too close to existing MTs
            if self.moving_targets_tree.size() > 0 {
                let nearest_existing_mt = self
                    .moving_targets_tree
                    .nearest_one::<SquaredEuclidean>(latest_position.as_ref());
                if nearest_existing_mt.distance < MOVING_ITEM_EXCLUSION_DIST2.az::<F>() {
                    continue;
                }
            }

            // don't add any MTs that are too close to existing SCs
            if self.star_candidates_tree.size() > 0 {
                let nearest_existing_mt = self
                    .star_candidates_tree
                    .nearest_one::<SquaredEuclidean>(latest_position.as_ref());
                if nearest_existing_mt.distance < MOVING_ITEM_EXCLUSION_DIST2.az::<F>() {
                    continue;
                }
            }

            let prev_frame_matches = prev_frame_state
                .detected_points_tree
                .nearest_n_within::<SquaredEuclidean>(
                    latest_position.as_ref(),
                    MAX_FRAME_DELTA.az::<F>(),
                    usize::MAX,
                    false,
                );

            let mut best_score = <F as FloatCore>::neg_infinity();
            let mut best_velocity = Vector2::new(F::zero(), F::zero());
            let mut best_pos_history = vec![];

            let mut curr_frame_position = latest_position;

            for prev_pos_candidate_res in prev_frame_matches {
                if prev_pos_candidate_res.distance < MIN_MOVEMENT_SPEED.az::<F>() {
                    continue;
                }

                let prev_pos_candidate: &DetectedPoint<F> =
                    &prev_frame_state.detected_points_list[prev_pos_candidate_res.item as usize];
                let prev_candidate_position = Vector2::new(
                    prev_pos_candidate.x.az::<F>(),
                    prev_pos_candidate.y.az::<F>(),
                );

                let mut frame_delta = curr_frame_position - prev_candidate_position;
                curr_frame_position = prev_candidate_position;

                let mut score: F = F::zero();
                let mut curr_candidate_pos_history = pos_history.clone();

                for lookback_frame_index in 2..MOVING_ITEM_LOOKBACK_WINDOW {
                    let Some(frame_state) = self
                        .recent_frame_states
                        .iter()
                        .rev()
                        .nth(lookback_frame_index)
                    else {
                        break;
                    };

                    let query_pos = latest_position
                        + (frame_delta * (lookback_frame_index as u32).az::<F>().neg());

                    let frame_match = frame_state
                        .detected_points_tree
                        .nearest_one::<SquaredEuclidean>(query_pos.as_ref());

                    let frame_match_point =
                        &frame_state.detected_points_list[frame_match.item as usize];

                    let frame_match_pos = if let Some(fitted_point) =
                        &frame_match_point.fitted_point
                    {
                        Vector2::new(fitted_point.x, fitted_point.y)
                    } else {
                        Vector2::new(frame_match_point.x.az::<F>(), frame_match_point.y.az::<F>())
                    };

                    if frame_match.distance <= MAX_MOVING_ITEM_JITTER.az::<F>() {
                        score += MAX_MOVING_ITEM_JITTER.az::<F>() - frame_match.distance;

                        frame_delta = (latest_position - frame_match_pos)
                            / (lookback_frame_index as u32).az::<F>();
                    }

                    curr_candidate_pos_history.insert(0, frame_match_pos);
                }

                if score > best_score {
                    best_velocity = frame_delta;
                    best_score = score;
                    best_pos_history = curr_candidate_pos_history;
                }
            }

            let best_speed2 =
                (best_velocity[0] * best_velocity[0]) + (best_velocity[1] * best_velocity[1]);
            if best_speed2 < MIN_MOVEMENT_SPEED.az::<F>() {
                continue;
            }

            if best_score >= MOVING_POINT_SCORE_THRESHOLD.az::<F>() {
                self.moving_targets.push(MovingTarget {
                    log_odds: best_score,
                    latest_score: best_score,
                    age: 0,
                    detected_point_match_history: vec![],
                    position: latest_position,
                    velocity: best_velocity,
                    pos_history: best_pos_history,
                });
            }
        }
    }
}
