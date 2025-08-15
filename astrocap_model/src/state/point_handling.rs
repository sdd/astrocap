use crate::state::{FrameState, ModelState};
use crate::traits::{AstroFloat, PointFitter};

use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::{Az, Cast};

use kiddo::SquaredEuclidean;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use nonmax::NonMaxUsize;
use num_traits::float::FloatCore;
use serde::Serialize;
use std::collections::HashSet;
use std::num::NonZero;
use std::sync::Arc;
use tracing::*;

#[derive(Clone, Debug, Serialize)]
pub struct DetectedPoint<F: AstroFloat> {
    pub x: u32,
    pub y: u32,
    pub amplitude: F,
    pub fitted_point: Option<FittedPoint<F>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FittedPoint<F: AstroFloat> {
    pub x: F,
    pub y: F,
    pub amplitude: F,
    pub radius_x: F,
    pub radius_y: F,
    pub score: F, // Keep this for backward compatibility during transition
    pub fit_quality: FittedPointQuality<F>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FittedPointQuality<F: AstroFloat> {
    pub reduced_chi_squared: F,
    pub snr: F,
    pub r_squared: F,
    pub rms_residual: F,
}

#[derive(Clone, Debug, Serialize)]
pub struct StarCandidate<F: AstroFloat> {
    pub x: F,
    pub y: F,
    pub amplitude: F,
    pub radius_x: F,
    pub radius_y: F,
    pub age: usize,
    pub log_likelihood: F,

    pub match_name: Option<String>,

    pub detected_point_match_history: Vec<Option<NonMaxUsize>>,

    // Kalman filter state: [x, y, vx, vy]
    pub kalman_state: [F; 4], // [position_x, position_y, velocity_x, velocity_y]
    pub kalman_covariance: [[F; 4]; 4], // State covariance matrix
    pub kalman_initialized: bool,
}

impl<F: AstroFloat> ModelState<F>
where
    u32: Cast<F>,
    u8: Cast<F>,
    f64: Cast<F>,
    F: Cast<u32>,
    F: Cast<i32>,
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
    pub(crate) fn fit_existing_points(
        &mut self,
        frame: Arc<dyn PointFitter<F>>,
    ) -> (HashSet<usize>, Vec<(usize, Option<usize>)>) {
        let mut match_indexes: HashSet<usize> = HashSet::new();
        let mut candidate_matches: Vec<(usize, Option<usize>)> = Vec::new();

        let Some(ref mut curr_frame_state) = &mut self.current_frame_state else {
            warn!("Not enough previous frame states found when trying to fit existing points");
            return (match_indexes, candidate_matches);
        };

        let mut unmatched_cand_count = 0;
        let mut matched_detected_points: HashSet<usize> = HashSet::new();
        let mut candidates_with_matches = Vec::new();

        for (cand_idx, cand) in self.star_candidates.iter_mut().enumerate() {
            cand.age += 1;

            // Calculate search radius based on track quality
            let recent_matches = cand
                .detected_point_match_history
                .iter()
                .rev()
                .take(10) // Look at last 10 frames
                .collect::<Vec<_>>();

            let match_rate = recent_matches.iter().filter(|m| m.is_some()).count() as f64
                / recent_matches.len() as f64;
            let has_kalman = cand.kalman_initialized;
            let track_confidence = cand.log_likelihood.az::<f64>();

            let search_radius = if has_kalman && match_rate > 0.7 && track_confidence > 20.0 {
                // Strong track - use tight radius
                // Much smaller for established tracks
                3.0f64.az::<F>()
            } else if recent_matches.len() >= 5 && match_rate > 0.4 {
                // Moderate track - use standard radius

                7.0f64.az::<F>()
            } else {
                // Weak/new track - use larger radius but not too large
                let unmatched_frames = cand
                    .detected_point_match_history
                    .iter()
                    .rev()
                    .take_while(|&m| m.is_none())
                    .count();

                let base_radius = self.model_config.max_existing_candidate_match_dist;
                if unmatched_frames > 5 {
                    let expansion_factor = (unmatched_frames as f64 * 0.1).min(1.5); // Reduced cap
                    base_radius * (1.0 + expansion_factor).az::<F>()
                } else {
                    base_radius
                }
            };

            let nearby_detected_points = curr_frame_state
                .detected_points_tree
                .nearest_n_within::<SquaredEuclidean>(
                    &[cand.x, cand.y],
                    search_radius * search_radius,
                    NonZero::new(5).unwrap(),
                    false,
                );

            if nearby_detected_points.is_empty() {
                unmatched_cand_count += 1;
                cand.log_likelihood -= self.model_config.star_candidate_unmatched_penalty;
                candidate_matches.push((cand_idx, None));
                continue;
            }

            let mut best_match: Option<(usize, F)> = None;

            for nearby_point in nearby_detected_points {
                let detected_point_idx = nearby_point.item as usize;

                if matched_detected_points.contains(&detected_point_idx) {
                    continue;
                }

                let detected_point = &mut curr_frame_state.detected_points_list[detected_point_idx];
                if detected_point.fitted_point.is_none() {
                    let fitted_point = frame.fit(detected_point);
                    detected_point.fitted_point = Some(fitted_point);
                }

                let fitted_point = detected_point.fitted_point.as_ref().unwrap();

                if fitted_point.radius_x > self.model_config.max_star_candidate_radius
                    || fitted_point.radius_y > self.model_config.max_star_candidate_radius
                {
                    continue;
                }

                if best_match.is_none() || fitted_point.score > best_match.as_ref().unwrap().1 {
                    best_match = Some((detected_point_idx, fitted_point.score));
                }
            }

            if let Some((best_detected_point_idx, _)) = best_match {
                let detected_point =
                    &curr_frame_state.detected_points_list[best_detected_point_idx];
                let fitted_point = detected_point.fitted_point.as_ref().unwrap();
                let measurement_valid = cand.validate_measurement(fitted_point.x, fitted_point.y);

                candidates_with_matches.push((
                    cand_idx,
                    fitted_point.x,
                    fitted_point.y,
                    measurement_valid,
                ));

                match_indexes.insert(best_detected_point_idx);
                matched_detected_points.insert(best_detected_point_idx);
                candidate_matches.push((cand_idx, Some(best_detected_point_idx)));
            } else {
                unmatched_cand_count += 1;
                cand.log_likelihood -= self.model_config.star_candidate_unmatched_penalty;
                candidate_matches.push((cand_idx, None));
            }
        }

        // Rest of the processing logic stays the same...
        for (cand_idx, x, y, measurement_valid) in candidates_with_matches {
            let history_confidence = self.star_candidates[cand_idx].validate_against_history(
                x,
                y,
                &self.historical_frame_states,
            );

            let cand = &mut self.star_candidates[cand_idx];
            let detected_point_idx = candidate_matches
                .iter()
                .find(|(idx, _)| *idx == cand_idx)
                .and_then(|(_, opt_idx)| *opt_idx)
                .unwrap();

            let detected_point = &curr_frame_state.detected_points_list[detected_point_idx];
            let fitted_point = detected_point.fitted_point.as_ref().unwrap();
            let measurement_noise = 1.0f64.az::<F>();

            if measurement_valid && history_confidence > 0.3f64.az::<F>() {
                if cand.detected_point_match_history.len() >= 2 {
                    cand.kalman_initialized = true;
                }

                cand.kalman_update_weighted(
                    fitted_point.x,
                    fitted_point.y,
                    measurement_noise,
                    history_confidence,
                );

                cand.log_likelihood += self.model_config.star_candidate_strong_match_bonus;
            } else {
                tracing::debug!("Rejecting spurious match for candidate due to poor validation");
                cand.log_likelihood -= self.model_config.star_candidate_unmatched_penalty;
            }

            cand.amplitude = fitted_point.amplitude;
            cand.radius_x = fitted_point.radius_x;
            cand.radius_y = fitted_point.radius_y;

            if fitted_point.score > self.model_config.star_candidate_strong_match_threshold {
                cand.log_likelihood += self.model_config.star_candidate_strong_match_bonus;
            }

            if fitted_point.amplitude < self.model_config.amplitude_penalty_threshold {
                cand.log_likelihood -= self.model_config.amplitude_penalty;
            }
        }

        info!(
            existing_fitted_qty = match_indexes.len(),
            unmatched_cand_count
        );
        (match_indexes, candidate_matches)
    }

    pub(crate) fn fit_strong_unmatched_new_points(
        &mut self,
        point_fitter: Arc<dyn PointFitter<F>>,
        img_w: u32,
        img_h: u32,
    ) {
        let Some(ref mut curr_frame_state) = self.current_frame_state else {
            warn!("Not enough previous frame states found when trying to fit existing points");
            return;
        };

        let initial_len = self.star_candidates.len();

        for (detected_point_index, detected_point) in
            curr_frame_state.detected_points_list.iter_mut().enumerate()
        {
            // skip detected points that have already been matched up to existing
            // star candidates
            if self.matched_point_indices.contains(&detected_point_index) {
                continue;
            }

            // skip points that are too dim
            if detected_point.amplitude
                < self
                    .model_config
                    .detected_point_candidate_amplitude_threshold
            {
                continue;
            }

            // skip points that are too close to existing star candidates
            let close_candidates = self
                .star_candidates_tree
                .nearest_n_within::<SquaredEuclidean>(
                    &[detected_point.x.az::<F>(), detected_point.y.az::<F>()],
                    self.model_config.max_existing_candidate_match_dist
                        * self.model_config.max_existing_candidate_match_dist,
                    NonZero::new(1).unwrap(),
                    false,
                );
            if !close_candidates.is_empty() {
                continue;
            }

            if detected_point.fitted_point.is_none() {
                detected_point.fitted_point = Some(point_fitter.fit(detected_point));
            }

            let fitted_point = &detected_point.fitted_point.clone().unwrap();

            if fitted_point.x >= F::zero()
                && fitted_point.y >= F::zero()
                && fitted_point.x < img_w.az::<F>()
                && fitted_point.y < img_h.az::<F>()
                && fitted_point.score > self.model_config.min_new_star_candidate_score
                && fitted_point.radius_x < self.model_config.max_star_candidate_radius
                && fitted_point.radius_y < self.model_config.max_star_candidate_radius
            {
                let new_cand = StarCandidate::new_with_kalman(
                    fitted_point.x,
                    fitted_point.y,
                    fitted_point.amplitude,
                    fitted_point.radius_x,
                    fitted_point.radius_y,
                    fitted_point.score,
                );

                // new_cand.kalman_update(fitted_point.x, fitted_point.y, 1.0f64.az::<F>());

                self.star_candidates.push(new_cand);
            }
        }

        info!(new_candidate_qty = self.star_candidates.len() - initial_len);
    }
}

impl<F: AstroFloat> StarCandidate<F>
where
    f64: Cast<F>,
{
    fn new_with_kalman(x: F, y: F, amplitude: F, radius_x: F, radius_y: F, score: F) -> Self {
        Self {
            x,
            y,
            amplitude,
            radius_x,
            radius_y,
            age: 0,
            log_likelihood: score,
            match_name: None,
            detected_point_match_history: Vec::new(),
            kalman_state: [x, y, F::zero(), F::zero()], // Initial velocity = 0
            kalman_covariance: Self::initial_covariance(),
            kalman_initialized: false,
        }
    }

    fn initial_covariance() -> [[F; 4]; 4] {
        let position_variance = 2.0f64.az::<F>(); // 2 pixel uncertainty
        let velocity_variance = 50f64.az::<F>(); // 50 pixel/frame uncertainty

        [
            [position_variance, F::default(), F::default(), F::default()],
            [F::default(), position_variance, F::default(), F::default()],
            [F::default(), F::default(), velocity_variance, F::default()],
            [F::default(), F::default(), F::default(), velocity_variance],
        ]
    }

    // Predict step: advance state by one time step
    pub fn kalman_predict(&mut self) {
        let dt = 1.0f64.az::<F>(); // 1 frame

        // Predicted state
        let predicted_state = [
            self.kalman_state[0] + self.kalman_state[2] * dt, // x + vx*dt
            self.kalman_state[1] + self.kalman_state[3] * dt, // y + vy*dt
            self.kalman_state[2],                             // vx unchanged
            self.kalman_state[3],                             // vy unchanged
        ];

        // State transition matrix F
        let f = [
            [1.0f64.az::<F>(), F::default(), dt, F::default()],
            [F::default(), 1.0f64.az::<F>(), F::default(), dt],
            [F::default(), F::default(), 1.0f64.az::<F>(), F::default()],
            [F::default(), F::default(), F::default(), 1.0f64.az::<F>()],
        ];

        // Calculate track quality for adaptive process noise
        let track_maturity = (self.age as f64).min(100.0) / 100.0;
        let recent_matches = self
            .detected_point_match_history
            .iter()
            .rev()
            .take(50)
            .filter(|m| m.is_some())
            .count() as f64
            / 50.0.min(self.detected_point_match_history.len() as f64);

        let track_confidence = (self.log_likelihood.az::<f64>() / 100.0).min(1.0).max(0.0);
        let track_quality =
            (track_maturity * 0.3 + recent_matches * 0.5 + track_confidence * 0.2).min(1.0);

        // Much more conservative process noise, especially for high-quality tracks
        let (position_noise, velocity_noise) = if track_quality > 0.8 {
            // Very established tracks: minimal process noise
            (0.1f64, 0.02f64) // Even smaller than before
        } else if track_quality > 0.5 {
            // Medium-quality tracks: moderate process noise
            (0.3f64, 0.05f64)
        } else {
            // New or poor tracks: higher process noise to allow for uncertainty
            (0.8f64, 0.2f64)
        };

        let position_noise = position_noise.az::<F>();
        let velocity_noise = velocity_noise.az::<F>();

        // Process noise covariance Q
        let q = [
            [position_noise, F::default(), F::default(), F::default()],
            [F::default(), position_noise, F::default(), F::default()],
            [F::default(), F::default(), velocity_noise, F::default()],
            [F::default(), F::default(), F::default(), velocity_noise],
        ];

        // Predict covariance: P = F*P*F^T + Q
        let predicted_covariance = self.matrix_multiply_4x4(&f, &self.kalman_covariance);
        let f_transpose = self.matrix_transpose_4x4(&f);
        let temp = self.matrix_multiply_4x4(&predicted_covariance, &f_transpose);
        let new_covariance = self.matrix_add_4x4(&temp, &q);

        // **KEY FIX**: Cap the maximum uncertainty for established tracks
        // This prevents uncertainty from growing unbounded during prediction-only periods
        let max_position_variance = if track_quality > 0.8 {
            2.0f64.az::<F>() // Very established tracks: max 2 pixel standard deviation
        } else if track_quality > 0.5 {
            4.0f64.az::<F>() // Medium tracks: max 4 pixel standard deviation
        } else {
            8.0f64.az::<F>() // New/poor tracks: max 8 pixel standard deviation
        };

        let max_velocity_variance = if track_quality > 0.8 {
            0.5f64.az::<F>() // Very small velocity uncertainty
        } else if track_quality > 0.5 {
            1.0f64.az::<F>()
        } else {
            2.0f64.az::<F>()
        };

        // Clamp the diagonal elements (variances) to prevent excessive uncertainty growth
        self.kalman_covariance = [
            [
                FloatCore::min(
                    new_covariance[0][0],
                    max_position_variance * max_position_variance,
                ),
                new_covariance[0][1],
                new_covariance[0][2],
                new_covariance[0][3],
            ],
            [
                new_covariance[1][0],
                FloatCore::min(
                    new_covariance[1][1],
                    max_position_variance * max_position_variance,
                ),
                new_covariance[1][2],
                new_covariance[1][3],
            ],
            [
                new_covariance[2][0],
                new_covariance[2][1],
                FloatCore::min(
                    new_covariance[2][2],
                    max_velocity_variance * max_velocity_variance,
                ),
                new_covariance[2][3],
            ],
            [
                new_covariance[3][0],
                new_covariance[3][1],
                new_covariance[3][2],
                FloatCore::min(
                    new_covariance[3][3],
                    max_velocity_variance * max_velocity_variance,
                ),
            ],
        ];

        self.kalman_state = predicted_state;

        // Update position estimates
        self.x = self.kalman_state[0];
        self.y = self.kalman_state[1];
    }

    // Update step: incorporate new measurement
    pub fn kalman_update_weighted(
        &mut self,
        measured_x: F,
        measured_y: F,
        measurement_noise: F,
        confidence: F,
    ) {
        if !self.kalman_initialized {
            // First measurement - just initialize
            self.kalman_state[0] = measured_x;
            self.kalman_state[1] = measured_y;
            return;
        }

        // Calculate track quality metrics
        let track_maturity = (self.age as f64).min(100.0) / 100.0; // 0.0 to 1.0
        let recent_matches = self
            .detected_point_match_history
            .iter()
            .rev()
            .take(50)
            .filter(|m| m.is_some())
            .count() as f64
            / 50.0.min(self.detected_point_match_history.len() as f64);

        let track_confidence = (self.log_likelihood.az::<f64>() / 100.0).min(1.0).max(0.0);

        // Track quality affects how much we trust our prediction vs the measurement
        let track_quality =
            (track_maturity * 0.3 + recent_matches * 0.5 + track_confidence * 0.2).min(1.0);

        // For high-quality tracks, we trust our prediction more and require measurements
        // to be more consistent with our motion model
        let measurement_trust_factor = if track_quality > 0.8 {
            // High-quality track - be more skeptical of measurements that don't fit the motion model
            0.2 + (1.0 - track_quality) * 0.3 // 0.2 to 0.5 range - much lower than before
        } else if track_quality > 0.5 {
            // Medium-quality track
            0.4 + (1.0 - track_quality) * 0.4 // 0.4 to 0.8 range
        } else {
            // Low-quality/new track - trust measurements more
            0.6 + (1.0 - track_quality) * 0.4 // 0.6 to 1.0 range
        };

        // Scale measurement noise based on track quality and measurement confidence
        let base_noise = measurement_noise / FloatCore::max(confidence, 0.1f64.az::<F>());
        let adjusted_noise = base_noise / (measurement_trust_factor.az::<F>());

        // Innovation validation: check if measurement is consistent with motion model
        let predicted_measurement = [self.kalman_state[0], self.kalman_state[1]];
        let raw_innovation = [
            measured_x - predicted_measurement[0],
            measured_y - predicted_measurement[1],
        ];

        // Calculate innovation magnitude
        let innovation_magnitude =
            (raw_innovation[0] * raw_innovation[0] + raw_innovation[1] * raw_innovation[1]).sqrt();

        // Current velocity magnitude
        let current_velocity = (self.kalman_state[2] * self.kalman_state[2]
            + self.kalman_state[3] * self.kalman_state[3])
            .sqrt();
        let position_uncertainty =
            (self.kalman_covariance[0][0] + self.kalman_covariance[1][1]).sqrt();

        // Reasonable movement thresholds based on track quality and current motion
        let max_reasonable_movement = if track_quality > 0.8 {
            // Very established track - allow current velocity + small acceleration + uncertainty
            // Max acceleration of ~0.5 pixels/frame² seems reasonable for stars
            FloatCore::max(
                5.0f64.az::<F>(), // Minimum threshold - even stationary stars can have small motion
                current_velocity * 1.2f64.az::<F>()
                    + 0.5f64.az::<F>()
                    + position_uncertainty * 2.0f64.az::<F>(),
            )
        } else if track_quality > 0.5 {
            // Medium quality track - allow more variation
            FloatCore::max(
                8.0f64.az::<F>(),
                current_velocity * 1.5f64.az::<F>()
                    + 1.0f64.az::<F>()
                    + position_uncertainty * 3.0f64.az::<F>(),
            )
        } else {
            // Low quality or new track - be more permissive
            FloatCore::max(
                15.0f64.az::<F>(),
                current_velocity * 2.0f64.az::<F>()
                    + 3.0f64.az::<F>()
                    + position_uncertainty * 4.0f64.az::<F>(),
            )
        };

        // Only reject truly extreme outliers
        let innovation = if innovation_magnitude > max_reasonable_movement {
            tracing::warn!(
            "Clamping large innovation: magnitude={:.2}, max_reasonable={:.2}, velocity={:.2}, quality={:.2}",
            innovation_magnitude.az::<f64>(),
            max_reasonable_movement.az::<f64>(),
            current_velocity.az::<f64>(),
            track_quality
        );
            // Scale down the innovation rather than rejecting completely
            let scale_factor = max_reasonable_movement / innovation_magnitude;
            [
                raw_innovation[0] * scale_factor,
                raw_innovation[1] * scale_factor,
            ]
        } else {
            raw_innovation
        };

        // Measurement matrix H (we observe position only)
        let _h = [
            [1.0f64.az::<F>(), F::default(), F::default(), F::default()],
            [F::default(), 1.0f64.az::<F>(), F::default(), F::default()],
        ];

        // Measurement noise covariance R
        let _r = [
            [adjusted_noise, F::default()],
            [F::default(), adjusted_noise],
        ];

        // Innovation covariance: S = H*P*H^T + R
        let s = [
            [
                self.kalman_covariance[0][0] + adjusted_noise,
                self.kalman_covariance[0][1],
            ],
            [
                self.kalman_covariance[1][0],
                self.kalman_covariance[1][1] + adjusted_noise,
            ],
        ];

        // Kalman gain: K = P*H^T*S^(-1)
        let s_inv = self.matrix_inverse_2x2(&s);
        let k = self.compute_kalman_gain(&s_inv);

        // Update state: x = x + K*innovation
        for i in 0..4 {
            self.kalman_state[i] += k[i][0] * innovation[0] + k[i][1] * innovation[1];
        }

        // Update covariance: P = (I - K*H)*P
        let i_kh = self.compute_covariance_update(&k);
        self.kalman_covariance = self.matrix_multiply_4x4(&i_kh, &self.kalman_covariance);

        // Update position estimates
        self.x = self.kalman_state[0];
        self.y = self.kalman_state[1];
    }

    // Helper methods for matrix operations (simplified for this use case)
    fn matrix_multiply_4x4(&self, a: &[[F; 4]; 4], b: &[[F; 4]; 4]) -> [[F; 4]; 4] {
        let mut result = [[F::default(); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        result
    }

    fn matrix_transpose_4x4(&self, a: &[[F; 4]; 4]) -> [[F; 4]; 4] {
        let mut result = [[F::default(); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = a[j][i];
            }
        }
        result
    }

    fn matrix_add_4x4(&self, a: &[[F; 4]; 4], b: &[[F; 4]; 4]) -> [[F; 4]; 4] {
        let mut result = [[F::default(); 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        result
    }

    fn matrix_inverse_2x2(&self, a: &[[F; 2]; 2]) -> [[F; 2]; 2] {
        let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
        let inv_det = F::one() / det;
        [
            [a[1][1] * inv_det, -a[0][1] * inv_det],
            [-a[1][0] * inv_det, a[0][0] * inv_det],
        ]
    }

    // Simplified Kalman gain computation for position measurements
    fn compute_kalman_gain(&self, s_inv: &[[F; 2]; 2]) -> [[F; 2]; 4] {
        let mut k = [[F::default(); 2]; 4];
        for i in 0..4 {
            for j in 0..2 {
                k[i][j] = self.kalman_covariance[i][j] * s_inv[j][0]
                    + self.kalman_covariance[i][j] * s_inv[j][1];
            }
        }
        k
    }

    fn compute_covariance_update(&self, k: &[[F; 2]; 4]) -> [[F; 4]; 4] {
        let mut i_kh = [[F::default(); 4]; 4];
        // I - K*H where H = [I 0; 0 I] for position measurements
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    i_kh[i][j] = F::one();
                }
                if j < 2 {
                    i_kh[i][j] -= k[i][j];
                }
            }
        }
        i_kh
    }

    pub fn validate_measurement(&self, measured_x: F, measured_y: F) -> bool {
        if !self.kalman_initialized {
            return true; // Accept all measurements for uninitialized filters
        }

        // Predicted measurement
        let predicted = [self.kalman_state[0], self.kalman_state[1]];

        // Innovation (difference between measurement and prediction)
        let innovation = [measured_x - predicted[0], measured_y - predicted[1]];

        // Innovation covariance (simplified for position-only measurement)
        let measurement_noise = 1.0f64.az::<F>(); // Adjust as needed
        let innovation_cov = [
            [
                self.kalman_covariance[0][0] + measurement_noise,
                self.kalman_covariance[0][1],
            ],
            [
                self.kalman_covariance[1][0],
                self.kalman_covariance[1][1] + measurement_noise,
            ],
        ];

        // Mahalanobis distance: innovation^T * S^(-1) * innovation
        let inv_cov = self.matrix_inverse_2x2(&innovation_cov);
        let mahal_dist_sq = innovation[0]
            * (inv_cov[0][0] * innovation[0] + inv_cov[0][1] * innovation[1])
            + innovation[1] * (inv_cov[1][0] * innovation[0] + inv_cov[1][1] * innovation[1]);

        // Chi-squared test with 2 degrees of freedom
        // 95% confidence: 5.99, 99% confidence: 9.21
        let confidence_threshold = 9.21f64.az::<F>();
        mahal_dist_sq < confidence_threshold
    }

    pub fn validate_against_history(
        &self,
        measured_x: F,
        measured_y: F,
        historical_frame_states: &[FrameState<F>],
    ) -> F {
        // Returns confidence weight [0.0, 1.0]

        let min_history_for_validation = 3;

        // Get the total number of frames we have (historical + current)
        let _total_frames = historical_frame_states.len() + 1; // +1 for current frame

        // The candidate's history should map to the most recent frames
        let history_len = self.detected_point_match_history.len();

        let valid_matches: Vec<_> = self
            .detected_point_match_history
            .iter()
            .enumerate()
            .filter_map(|(history_idx, opt_match)| {
                // Calculate the actual frame index in historical_frame_states
                // The most recent history entry corresponds to the last historical frame
                let frame_idx_in_history = if history_len > historical_frame_states.len() {
                    // More history than we have historical frames - skip early entries
                    let skip_count = history_len - historical_frame_states.len();
                    if history_idx < skip_count {
                        return None; // Skip entries that are too old
                    }
                    history_idx - skip_count
                } else {
                    // History fits within historical frames
                    let start_offset = historical_frame_states.len() - history_len;
                    start_offset + history_idx
                };

                // Skip if this would access the current frame (not in historical_frame_states)
                if frame_idx_in_history >= historical_frame_states.len() {
                    return None;
                }

                opt_match.map(|match_idx| {
                    let frame_state = &historical_frame_states[frame_idx_in_history];
                    let point = &frame_state.detected_points_list[match_idx.get()];
                    let fitted = point.fitted_point.as_ref().unwrap();
                    (history_idx, fitted.x, fitted.y)
                })
            })
            .collect();

        if valid_matches.len() < min_history_for_validation {
            return 1.0f64.az::<F>(); // Full confidence for new tracks
        }

        // Calculate recent velocity trend
        let recent_positions: Vec<_> = valid_matches
            .iter()
            .rev()
            .take(3) // Last 3 positions
            .collect();

        if recent_positions.len() >= 2 {
            // Estimate velocity from recent history
            let (_, x1, y1) = recent_positions[0];
            let (_, x2, y2) = recent_positions[1];
            let estimated_vx = *x1 - *x2;
            let estimated_vy = *y1 - *y2;

            // Predict where we expect the next point
            let predicted_x = *x1 + estimated_vx;
            let predicted_y = *y1 + estimated_vy;

            // Calculate deviation from prediction
            let deviation = ((measured_x - predicted_x) * (measured_x - predicted_x)
                + (measured_y - predicted_y) * (measured_y - predicted_y))
                .sqrt();

            // Convert to confidence weight (sigmoid function)
            let max_allowed_deviation = 5.0f64.az::<F>();

            ((-deviation / max_allowed_deviation).exp())
                / (1.0f64.az::<F>() + (-deviation / max_allowed_deviation).exp())
        } else {
            1.0f64.az::<F>() // Full confidence if insufficient history
        }
    }
}
