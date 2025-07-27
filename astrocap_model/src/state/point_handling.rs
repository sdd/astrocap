use crate::state::ModelState;
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
    pub(crate) fn fit_existing_points(&mut self, frame: Arc<dyn PointFitter<F>>) -> HashSet<usize> {
        let mut match_indexes: HashSet<usize> = HashSet::new();

        let Some(curr_frame_state) = self.recent_frame_states.last_mut() else {
            warn!("Not enough previous frame states found when trying to fit existing points");
            return match_indexes;
        };

        let mut unmatched_cand_count = 0;
        let mut matched_detected_points: HashSet<usize> = HashSet::new();

        for (_cand_idx, cand) in self.star_candidates.iter_mut().enumerate() {
            // Calculate adaptive search radius based on unmatched frame count
            let unmatched_frames = cand
                .detected_point_match_history
                .iter()
                .rev()
                .take_while(|&m| m.is_none())
                .count();

            let search_radius = if unmatched_frames > 5 {
                let base_radius = self.model_config.max_existing_candidate_match_dist;
                let expansion_factor = (unmatched_frames as f64 * 0.1).min(2.0); // Cap expansion
                base_radius * (1.0 + expansion_factor).az::<F>()
            } else {
                self.model_config.max_existing_candidate_match_dist
            };

            // Get up to 5 nearby detected points
            let nearby_detected_points = curr_frame_state
                .detected_points_tree
                .nearest_n_within::<SquaredEuclidean>(
                    &[cand.x, cand.y], // Using Kalman-predicted position
                    search_radius * search_radius,
                    NonZero::new(5).unwrap(),
                    false,
                );

            if nearby_detected_points.is_empty() {
                // No matches found within search radius
                unmatched_cand_count += 1;
                cand.detected_point_match_history.push(None);
                continue;
            }

            // Fit all nearby points and find the best one
            let mut best_match: Option<(usize, F)> = None; // (detected_point_idx, score)

            for nearby_point in nearby_detected_points {
                let detected_point_idx = nearby_point.item as usize;

                // Skip if this detected point was already matched to another candidate
                if matched_detected_points.contains(&detected_point_idx) {
                    continue;
                }

                // Fit the detected point if not already fitted
                let detected_point = &mut curr_frame_state.detected_points_list[detected_point_idx];
                if detected_point.fitted_point.is_none() {
                    let fitted_point = frame.fit(detected_point);
                    detected_point.fitted_point = Some(fitted_point);
                }

                let fitted_point = detected_point.fitted_point.as_ref().unwrap();

                // Reject if the fitted radii are too large (not stellar)
                if fitted_point.radius_x > self.model_config.max_star_candidate_radius
                    || fitted_point.radius_y > self.model_config.max_star_candidate_radius
                {
                    continue;
                }

                // Check if this is the best match so far (highest score)
                if best_match.is_none() || fitted_point.score > best_match.as_ref().unwrap().1 {
                    best_match = Some((detected_point_idx, fitted_point.score));
                }
            }

            if let Some((best_detected_point_idx, _)) = best_match {
                let detected_point =
                    &curr_frame_state.detected_points_list[best_detected_point_idx];
                let fitted_point = detected_point.fitted_point.as_ref().unwrap();

                // Update Kalman filter with the fitted position (more accurate than detected position)
                // if this is the first match then we start the kalman filter updating too now that
                // we have a good initial estimate for velocity
                let measurement_noise = 1.0f64.az::<F>(); // Adjust based on your detector accuracy
                cand.kalman_initialized = true;
                cand.kalman_update(fitted_point.x, fitted_point.y, measurement_noise);

                // Update candidate properties with fitted values
                cand.amplitude = fitted_point.amplitude;
                cand.radius_x = fitted_point.radius_x;
                cand.radius_y = fitted_point.radius_y;

                // Add strong match bonus if score indicates high quality
                if fitted_point.score > self.model_config.star_candidate_strong_match_threshold {
                    cand.log_likelihood += self.model_config.star_candidate_strong_match_bonus;
                }

                // Record the match
                match_indexes.insert(best_detected_point_idx);
                matched_detected_points.insert(best_detected_point_idx);
                cand.detected_point_match_history
                    .push(Some(NonMaxUsize::new(best_detected_point_idx).unwrap()));
            } else {
                // No valid matches found (all were either taken or failed stellar criteria)
                unmatched_cand_count += 1;
                cand.detected_point_match_history.push(None);
            }
        }

        info!(
            existing_fitted_qty = match_indexes.len(),
            unmatched_cand_count
        );

        match_indexes
    }

    pub(crate) fn fit_strong_unmatched_new_points(
        &mut self,
        point_fitter: Arc<dyn PointFitter<F>>,
        img_w: u32,
        img_h: u32,
    ) {
        let Some(curr_frame_state) = self.recent_frame_states.last_mut() else {
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
                let mut new_cand = StarCandidate::new_with_kalman(
                    fitted_point.x,
                    fitted_point.y,
                    fitted_point.amplitude,
                    fitted_point.radius_x,
                    fitted_point.radius_y,
                    fitted_point.score,
                );

                new_cand.kalman_update(fitted_point.x, fitted_point.y, 1.0f64.az::<F>());

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
        let mut candidate = Self {
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
        };
        candidate
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
        // State transition: x(k+1) = x(k) + vx(k), y(k+1) = y(k) + vy(k)
        // Velocity assumed constant: vx(k+1) = vx(k), vy(k+1) = vy(k)
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

        // Process noise covariance Q (small amount of random acceleration)
        let process_noise = 0.1f64.az::<F>();
        let q = [
            [process_noise, F::default(), F::default(), F::default()],
            [F::default(), process_noise, F::default(), F::default()],
            [
                F::default(),
                F::default(),
                process_noise * 0.1f64.az::<F>(),
                F::default(),
            ],
            [
                F::default(),
                F::default(),
                F::default(),
                process_noise * 0.1f64.az::<F>(),
            ],
        ];

        // Predict covariance: P = F*P*F^T + Q
        let predicted_covariance = self.matrix_multiply_4x4(&f, &self.kalman_covariance);
        let f_transpose = self.matrix_transpose_4x4(&f);
        let temp = self.matrix_multiply_4x4(&predicted_covariance, &f_transpose);
        self.kalman_covariance = self.matrix_add_4x4(&temp, &q);

        self.kalman_state = predicted_state;

        // Update position estimates
        self.x = self.kalman_state[0];
        self.y = self.kalman_state[1];
    }

    // Update step: incorporate new measurement
    pub fn kalman_update(&mut self, measured_x: F, measured_y: F, measurement_noise: F) {
        if !self.kalman_initialized {
            // First measurement - just initialize
            self.kalman_state[0] = measured_x;
            self.kalman_state[1] = measured_y;
            return;
        }

        // Measurement matrix H (we observe position only)
        let h = [
            [1.0f64.az::<F>(), F::default(), F::default(), F::default()],
            [F::default(), 1.0f64.az::<F>(), F::default(), F::default()],
        ];

        // Measurement noise covariance R
        let r = [
            [measurement_noise, F::default()],
            [F::default(), measurement_noise],
        ];

        // Innovation: z - H*x
        let predicted_measurement = [self.kalman_state[0], self.kalman_state[1]];
        let innovation = [
            measured_x - predicted_measurement[0],
            measured_y - predicted_measurement[1],
        ];

        // Innovation covariance: S = H*P*H^T + R
        // Simplified for position-only measurement
        let s = [
            [
                self.kalman_covariance[0][0] + measurement_noise,
                self.kalman_covariance[0][1],
            ],
            [
                self.kalman_covariance[1][0],
                self.kalman_covariance[1][1] + measurement_noise,
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
}
