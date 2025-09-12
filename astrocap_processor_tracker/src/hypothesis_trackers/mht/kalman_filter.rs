use crate::model::Detection;
use nalgebra::{Matrix2, Matrix2x4, Matrix4, Matrix4x2, Vector2, Vector4};

use crate::hypothesis_trackers::mht::LogProbs;
use std::sync::LazyLock;

// State transition matrix F. Assumes constant frame rate
#[rustfmt::skip]
pub static F: LazyLock<Matrix4<f32>> = LazyLock::new(|| Matrix4::from_row_slice(&[
    1.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]));

// Static measurement matrix H: projects state [x, y, vx, vy] into [x, y]
#[rustfmt::skip]
pub static H: LazyLock<Matrix2x4<f32>> = LazyLock::new(|| Matrix2x4::new(
    1.0, 0.0, 0.0, 0.0, 
    0.0, 1.0, 0.0, 0.0
));

/// Process noise template (Q_base).
/// This is scaled each frame by q_scale.
/// Form is cononical "constant-velocity white acceleration" block structure.
#[rustfmt::skip]
#[inline]
fn q_cv(dt: f32, q: f32) -> Matrix4<f32> {
    use nalgebra::Matrix4;
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let dt4 = dt2 * dt2;
    Matrix4::from_row_slice(&[
        0.25 * dt4, 0.0,        0.5 * dt3, 0.0,
        0.0,        0.25 * dt4, 0.0,       0.5 * dt3,
        0.5 * dt3,  0.0,        dt2,       0.0,
        0.0,        0.5 * dt3,  0.0,       dt2,
    ]) * q
}

/// baseline acceleration variance, tuned experimentally
const BASELINE_ACCEL_VARIANCE: f32 = 1e-3;

/// Structure holding KF state + covariance + adaptive noise scales
#[derive(Clone, Debug)]
pub struct KalmanFilter {
    pub state: Vector4<f32>,      // [x, y, vx, vy]
    pub covariance: Matrix4<f32>, // 4×4 covariance of the state estimate

    // Measurement covariance (R). We adapt this online.
    pub r: Matrix2<f32>,

    // Scale applied to process noise template (Q_base).
    pub q_scale: f32,

    // Exponential moving average of NIS (tracks consistency over time).
    pub nis_ewma: f32,

    // adaptation parameters
    pub q_min: f32,
    pub q_max: f32,
    pub q_eta: f32,
    pub r_min: f32,
    pub r_max: f32,
    pub r_beta: f32,
}

impl KalmanFilter {
    pub fn new_from_detection(detection: &Detection) -> Self {
        Self::new_with_velocity(Vector4::new(
            detection.position.x,
            detection.position.y,
            0.0,
            0.0,
        ))
    }

    pub fn new_with_velocity(state: Vector4<f32>) -> Self {
        Self {
            state,
            covariance: Matrix4::identity(),
            r: Matrix2::from_diagonal(&Vector2::new(0.25, 0.25)),
            // q_scale: 1.0,
            nis_ewma: 0.0,

            q_scale: 1e-4, // small but nonzero, stars move slowly
            q_min: 1e-6,   // never let Q vanish completely
            q_max: 1e-2,   // allow modest growth if stars are lost for long
            q_eta: 0.01,   // 1% adaptation step per update

            r_min: 0.25,  // sqrt variance = 0.5 pixels
            r_max: 4.0,   // sqrt variance = 4 pixels
            r_beta: 0.02, // 2% smoothing toward empirical variance
        }
    }

    pub fn new_with_covariance(state: Vector4<f32>, covariance: Matrix4<f32>) -> Self {
        Self {
            state,
            covariance,
            r: Matrix2::from_diagonal(&Vector2::new(0.25, 0.25)),

            // q_scale: 1.0,
            nis_ewma: 0.0,

            q_scale: 1e-4, // small but nonzero, stars move slowly
            q_min: 1e-6,   // never let Q vanish completely
            q_max: 1e-2,   // allow modest growth if stars are lost for long
            q_eta: 0.01,   // 1% adaptation step per update

            r_min: 0.25,  // sqrt variance = 0.5 pixels
            r_max: 4.0,   // sqrt variance = 4 pixels
            r_beta: 0.02, // 2% smoothing toward empirical variance
        }
    }

    pub fn x(&self) -> f32 {
        self.state.x
    }

    pub fn y(&self) -> f32 {
        self.state.y
    }

    pub fn covariance(&self) -> &Matrix4<f32> {
        &self.covariance
    }

    pub fn r(&self) -> &Matrix2<f32> {
        &self.r
    }

    pub fn predict(&self, dt: f32) -> Self {
        // Process noise Q (scaled by self.q_scale).
        let q = q_cv(dt, self.q_scale * BASELINE_ACCEL_VARIANCE);
        // let q = q_cv(1.0, /* tune me */ 1e-5); // start around 1e-3, adjust later

        // Predict state
        let state = *F * self.state;

        // Predict covariance
        let covariance = *F * self.covariance * F.transpose() + q;

        Self {
            state,
            covariance,

            r: self.r,
            q_scale: self.q_scale,
            nis_ewma: self.nis_ewma,

            q_min: self.q_min,
            q_max: self.q_max,
            q_eta: self.q_eta,
            r_min: self.r_min,
            r_max: self.r_max,
            r_beta: self.r_beta,
        }
    }

    /// Update step (measurement incorporated).
    /// Uses Joseph form for numerical stability.
    pub fn update(&mut self, detection: &Detection, measurement_cov: &Matrix2<f32>) -> bool {
        let Some(k) = self.kalman_gain(measurement_cov) else {
            return false; // S not invertible
        };

        // Measurement vector z = [x_meas, y_meas]
        let z = Vector2::new(detection.position.x, detection.position.y);

        // Innovation residual (difference between measurement and prediction)
        let r = z - *H * self.state;

        // State update
        self.state = self.state + k * r;

        // Joseph form covariance update:
        // P ← (I - KH) P (I - KH)ᵀ + K R Kᵀ
        let i4 = Matrix4::identity();
        let i_minus_kh = i4 - k * *H;
        self.covariance = i_minus_kh * self.covariance * i_minus_kh.transpose()
            + k * *measurement_cov * k.transpose();

        // Defensive re-symmetrisation (avoid small asymmetries from FP noise).
        self.covariance = 0.5 * (self.covariance + self.covariance.transpose());
        true
    }

    /// Kalman gain computation.
    /// We factor this out so update() can be written in Joseph form.
    pub fn kalman_gain(&self, measurement_cov: &Matrix2<f32>) -> Option<Matrix4x2<f32>> {
        let s = *H * self.covariance * H.transpose() + *measurement_cov;
        let s_inv = safe_invert_2x2(s)?;
        let k = self.covariance * H.transpose() * s_inv;
        Some(k)
    }

    /// Innovation calculation (residual, covariance S, and NIS).
    /// Called before gating / association.
    pub fn innovation(&self, detection: &Detection) -> (Vector2<f32>, Matrix2<f32>, f32) {
        // Measurement vector
        let z = Vector2::new(detection.position.x, detection.position.y);

        // Predicted measurement
        let z_pred = *H * self.state;

        // Innovation residual
        let r = z - z_pred;

        // Innovation covariance
        let mut s = *H * self.covariance * H.transpose() + self.r;

        // Symmetrise and add jitter to ensure PD
        s = s.symmetric_part() + Matrix2::identity() * 1e-6;

        // Attempt Cholesky factorisation
        if let Some(chol) = s.cholesky() {
            let y = chol.solve(&r);
            let nis = y.dot(&y);
            (r, s, nis)
        } else {
            // Fallback: not PD, use pseudo-inverse
            tracing::warn!("Innovation covariance not PD: {:?}", s);
            let s_inv = s.try_inverse().unwrap_or(Matrix2::identity());
            let nis = (r.transpose() * s_inv * r)[0];
            (r, s, nis)
        }
    }

    // /// Association log-likelihood
    // pub fn assoc_log_likelihood(&self, detection: &Detection, pd: f32) -> f32 {
    //     let (r, s) = self.innovation(detection);
    //     let s_inv = s.try_inverse().unwrap_or_else(Matrix2::identity);
    //
    //     let mahalanobis = r.transpose() * s_inv * r;
    //     let det_s = s.determinant().max(1e-6);
    //
    //     let log_gaussian =
    //         -0.5 * (mahalanobis[(0, 0)] + (2.0 * std::f32::consts::PI).ln() + det_s.ln());
    //
    //     pd.ln() + log_gaussian
    // }

    // /// Association log-likelihood (relative form, ignores normalisation constant).
    // /// (This can't be used if you want convertability to log-odds)
    // pub fn assoc_log_likelihood(&self, detection: &Detection, pd: f32) -> f32 {
    //     let (r, s) = self.innovation(detection);
    //     let s_inv = s.try_inverse().unwrap_or_else(Matrix2::identity);
    //
    //     let mahalanobis = r.transpose() * s_inv * r;
    //
    //     // Relative log-likelihood: log(pd) - 0.5 * Mahalanobis distance
    //     pd.ln() - 0.5 * mahalanobis[(0, 0)]
    // }

    /// Association confidence calculation.
    /// Returns log-likelihood (LL) and log-odds (LO).
    pub fn assoc_conf(
        &self,
        detection: &Detection,
        pd: f32,
        clutter_rate: f32,
        measurement_cov: &Matrix2<f32>,
    ) -> (LogProbs, f32, Vector2<f32>, Matrix2<f32>) {
        // Measurement vector
        let z = Vector2::new(detection.position.x, detection.position.y);

        // Predicted measurement
        let z_pred = *H * self.state;

        // Residual
        let residual = z - z_pred;

        // Innovation covariance S
        // S = H P Hᵀ + R
        let s = *H * self.covariance * H.transpose() + *measurement_cov;

        // invert safely
        let s_inv = match safe_invert_2x2(s) {
            Some(inv) => inv,
            None => {
                // return a huge NIS so the caller gates it out
                return (
                    LogProbs {
                        ll: f32::NEG_INFINITY,
                        lo: f32::NEG_INFINITY,
                    },
                    f32::INFINITY,
                    residual,
                    s,
                );
            }
        };

        // NIS = rᵀ S⁻¹ r
        let nis = (residual.transpose() * s_inv * residual)[0];

        // proper Gaussian LL (normalised)
        let det_s = (s[(0, 0)] * s[(1, 1)] - s[(0, 1)] * s[(1, 0)]).max(1e-12);
        let log_gauss = -0.5 * (nis + (2.0 * std::f32::consts::PI).ln() + det_s.ln());

        // Clamp Pd to avoid log(0)
        let pd = pd.clamp(1e-6, 1.0 - 1e-6);

        // Log-likelihood increment
        let delta_ll = pd.ln() + log_gauss;

        // Log-odds increment
        // Option A (true clutter density): subtract ln c
        let delta_lo = pd.ln() - 0.5 * nis - clutter_rate.ln();
        // Option B (gate-integrated clutter): subtract ln(c*A) instead of ln c
        // let gate_area = std::f32::consts::PI * GATE_RADIUS * GATE_RADIUS;
        // let delta_lo = pd.ln() - 0.5 * nis - (clutter_rate * gate_area).ln();

        (
            LogProbs {
                ll: delta_ll,
                lo: delta_lo,
            },
            nis,
            residual,
            s,
        )
    }

    /// Adapt R (measurement covariance) using residuals.
    pub fn adapt_r(&mut self, residual: &Vector2<f32>, s_pred: &Matrix2<f32>) {
        // Gain factor
        let beta = 0.05;

        // Outer product of residual: r rᵀ
        let r_update = residual * residual.transpose();

        // EWMA update of measurement covariance
        self.r = (1.0 - beta) * self.r + beta * (r_update + s_pred);

        // Clamp diagonals to keep them sensible
        for i in 0..2 {
            self.r[(i, i)] = self.r[(i, i)].clamp(0.01, 16.0);
        }
    }

    /// Adapt Q (process noise scale) based on mean NIS.
    pub fn adapt_q_from_mean(&mut self, mean_nis: f32) {
        // Target NIS ~ 2 (for 2D measurements).
        let target = 2.0;

        // EWMA of NIS to smooth fluctuations
        let alpha = 0.05;
        self.nis_ewma = (1.0 - alpha) * self.nis_ewma + alpha * mean_nis;

        // Adjustment factor: push q_scale up if NIS < target, down if > target
        let ratio = (self.nis_ewma / target).clamp(0.5, 2.0);
        self.q_scale *= ratio;

        // Clamp q_scale to avoid collapse or explosion
        self.q_scale = self.q_scale.clamp(1e-8, 1e-2);
    }

    /// Very gentle multiplicative inflation of Q when we missed a detection.
    /// This keeps tracks "alive" over long gaps (e.g. dim stars vanishing for 50 frames).
    pub fn adapt_q_on_miss(&mut self) {
        // Small inflation factor
        let inflation = 1.02; // 2% per miss
        self.q_scale = (self.q_scale * inflation).min(self.q_max);
    }

    /// Adapt Q online based on NIS vs. expected measurement dimension.
    /// If NIS > dim, process noise is too small → increase q_scale.
    /// If NIS < dim, process noise too large → decrease q_scale.
    pub fn adapt_q_from_nis(&mut self, nis: f32, meas_dim: f32) {
        if !nis.is_finite() || nis <= 0.0 {
            return; // ignore garbage
        }

        let err_ratio = (nis / meas_dim).clamp(0.25, 4.0);
        let step = self.q_eta; // e.g. 0.01
        self.q_scale *= 1.0 + step * (err_ratio - 1.0);

        // clamp to configured safe range
        self.q_scale = self.q_scale.clamp(self.q_min, self.q_max);
    }

    /// Adapt R diagonals toward the empirical variance of residuals.
    /// Uses exponential smoothing to avoid jitter.
    pub fn adapt_r_from_pair(&mut self, residual: &Vector2<f32>, _s_pred: &Matrix2<f32>) {
        let beta = self.r_beta; // e.g. 0.02

        // Residual variance estimate (just square each component)
        let rx = residual.x * residual.x;
        let ry = residual.y * residual.y;

        let mut r_new = self.r;

        r_new[(0, 0)] = (1.0 - beta) * r_new[(0, 0)] + beta * rx;
        r_new[(1, 1)] = (1.0 - beta) * r_new[(1, 1)] + beta * ry;

        // Clamp to safe bounds (squared values, since cov is variance)
        r_new[(0, 0)] = r_new[(0, 0)].clamp(self.r_min.powi(2), self.r_max.powi(2));
        r_new[(1, 1)] = r_new[(1, 1)].clamp(self.r_min.powi(2), self.r_max.powi(2));

        self.r = r_new;
    }
}

#[inline]
fn safe_invert_2x2(mut s: Matrix2<f32>) -> Option<Matrix2<f32>> {
    // enforce symmetry + jitter
    s = 0.5 * (s + s.transpose());
    s[(0, 0)] += 1e-6;
    s[(1, 1)] += 1e-6;

    let det = s[(0, 0)] * s[(1, 1)] - s[(0, 1)] * s[(1, 0)];
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }
    let inv = Matrix2::new(
        s[(1, 1)] / det,
        -s[(0, 1)] / det,
        -s[(1, 0)] / det,
        s[(0, 0)] / det,
    );
    if !inv[(0, 0)].is_finite() {
        return None;
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kf_predict_moves_forward() {
        use nalgebra::{Matrix4, Vector4};

        // Initial state: (x=0, y=0, vx=1, vy=2)
        let init_state = Vector4::new(0.0, 0.0, 1.0, 2.0);
        let kf = KalmanFilter::new_with_covariance(init_state, Matrix4::identity());

        let predicted = kf.predict();

        // Expect position advanced by velocity: x=1, y=2
        assert!((predicted.x() - 1.0).abs() < 1e-5);
        assert!((predicted.y() - 2.0).abs() < 1e-5);

        // Velocity should be unchanged
        assert!((predicted.state[2] - 1.0).abs() < 1e-5);
        assert!((predicted.state[3] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_kf_update_pulls_toward_detection() {
        use crate::model::Detection;
        use nalgebra::{Matrix4, Vector4};

        // Start at (0,0) with zero velocity
        let init_state = Vector4::new(0.0, 0.0, 0.0, 0.0);
        let mut kf = KalmanFilter::new_with_covariance(init_state, Matrix4::identity());

        // Measurement at (10,10)
        let detection = Detection {
            id: 1,
            position: Vector2::new(10.0, 10.0),
            amplitude: 100.0,
        };

        kf.update(&detection);

        // After update, x and y should have shifted toward 10
        assert!(kf.x() > 0.0 && kf.x() < 10.0);
        assert!(kf.y() > 0.0 && kf.y() < 10.0);
    }
}
