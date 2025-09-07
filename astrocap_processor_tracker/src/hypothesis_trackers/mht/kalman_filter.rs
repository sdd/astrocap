use crate::model::Detection;
use nalgebra::{Matrix2, Matrix2x4, Matrix4, Matrix4x2, Vector2, Vector4};

use crate::hypothesis_trackers::mht::LogProbs;
use std::sync::LazyLock;

#[rustfmt::skip]
pub static F: LazyLock<Matrix4<f32>> = LazyLock::new(|| Matrix4::from_row_slice(&[
    1.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]));

#[rustfmt::skip]
pub static H: LazyLock<Matrix2x4<f32>> = LazyLock::new(|| Matrix2x4::new(
    1.0, 0.0, 0.0, 0.0, 
    0.0, 1.0, 0.0, 0.0
));

// TODO: make this configurable
pub static R: LazyLock<Matrix2<f32>> = LazyLock::new(|| Matrix2::identity() * 4.0);

// TODO: online learning of this parameter
// Tuning guide:
// If tracks lag real motion → increase q (e.g. ×3).
// If tracks jitter despite clean detections → decrease q (e.g. ÷3).
//
// The "NIS" value logged by the KF update step can be used to tune this parameter.
// Run your tracker for a while.
// Look at average NIS and spread.
// If mean NIS >> 2 → bump up q.
// If mean NIS << 2 → shrink q.
// #[rustfmt::skip]
// pub static Q: LazyLock<Matrix4<f32>> = LazyLock::new(|| {
//     let q = 0.0000001;
//     Matrix4::from_row_slice(&[
//         1.0, 0.0, 1.0, 0.0,
//         0.0, 1.0, 0.0, 1.0,
//         0.0, 0.0, 1.0, 0.0,
//         0.0, 0.0, 0.0, 1.0,
//     ]) * q
// });

#[inline]
fn q_cv(dt: f32, q: f32) -> Matrix4<f32> {
    use nalgebra::Matrix4;
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let dt4 = dt2 * dt2;
    Matrix4::from_row_slice(&[
        0.25 * dt4,
        0.0,
        0.5 * dt3,
        0.0,
        0.0,
        0.25 * dt4,
        0.0,
        0.5 * dt3,
        0.5 * dt3,
        0.0,
        dt2,
        0.0,
        0.0,
        0.5 * dt3,
        0.0,
        dt2,
    ]) * q
}

#[derive(Clone, Debug)]
pub struct KalmanFilter {
    state: Vector4<f32>,
    covariance: Matrix4<f32>,
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
        }
    }

    pub fn new_with_covariance(state: Vector4<f32>, covariance: Matrix4<f32>) -> Self {
        Self { state, covariance }
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

    pub fn predict(&self) -> Self {
        let q = q_cv(1.0, /* tune me */ 1e-3); // start around 1e-3, adjust later

        let state_pred = *F * self.state;
        let p_pred = *F * self.covariance * F.transpose() + q;

        Self {
            state: state_pred,
            covariance: p_pred,
        }
    }

    pub fn update(&mut self, detection: &Detection) {
        // K = PHᵀ (HPHᵀ + R)⁻¹
        // Kalman gain = cov * meas_trans * inv(meas * cov * meas_trans + meas_noise_cov)

        let k = self.kalman_gain();

        // X̂ₙ₊₁ = X̂ₙ + Kₙ (Ẑₙ − H x̂ₙ)
        // new state = old state + kalman gain * (measurement - observation matrix * old state)

        let z_n = Vector2::new(detection.position.x, detection.position.y);

        let new_state = self.state + k * (z_n - *H * self.state);

        self.state = new_state;

        // Pₙ₊₁ = (I - KH) Pₙ (I - KH)ᵀ + KRKᵀ
        // new_covariance = (identity - kalman_gain * meas) * covariance
        //                * (identity - kalman_gain * meas).transpose()
        //                + kalman_gain * meas_noise_cov * kalman_gain.transpose()

        let i_minus_k_h = Matrix4::identity() - k * *H;

        let new_covariance =
            i_minus_k_h * self.covariance * i_minus_k_h.transpose() + (k * *R * k.transpose());

        self.covariance = new_covariance;
    }

    pub fn kalman_gain(&self) -> Matrix4x2<f32> {
        // K = PHᵀ (HPHᵀ + R)⁻¹
        // Kalman gain = cov * meas_trans * inv(meas * cov * meas_trans + meas_noise_cov)

        let k = self.covariance
            * H.transpose()
            * (*H * self.covariance * H.transpose() + *R)
                .try_inverse()
                .unwrap();

        k
    }

    /// Compute innovation residual and covariance
    pub fn innovation(&self, detection: &Detection) -> (Vector2<f32>, Matrix2<f32>) {
        let z = Vector2::new(detection.position.x, detection.position.y);
        let z_pred = &*H * self.state;
        let residual = z - z_pred;

        // AKA innovation_covariance
        let s = &*H * self.covariance * H.transpose() + *R;
        let s_inv = s.try_inverse().unwrap_or_else(Matrix2::identity);

        let nis = (residual.transpose() * s_inv * residual)[0];
        println!("NIS = {:.3}", nis);

        (residual, s)
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

    pub fn assoc_conf(
        &self,
        detection: &Detection,
        pd: f32,
        clutter_rate: f32, // c (per-cell intensity)
    ) -> LogProbs {
        let (r, s) = self.innovation(detection);
        let s_inv = s.try_inverse().unwrap_or_else(Matrix2::identity);

        let mahalanobis = (r.transpose() * s_inv * r)[0];
        let det_s = s.determinant().max(1e-6);

        // full Gaussian LL
        let delta_ll =
            pd.ln() - 0.5 * mahalanobis - 0.5 * ((2.0 * std::f32::consts::PI).ln() + det_s.ln());

        // LO = LL − ln(clutter density)
        let delta_lo = delta_ll - clutter_rate.ln();

        LogProbs {
            ll: delta_ll,
            lo: delta_lo,
        }
    }
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
