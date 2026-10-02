use nalgebra::{
    Matrix2, Matrix2x4, Matrix3, Matrix3x5, Matrix4, Matrix5, Matrix5x3, Vector3, Vector5,
};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::hypothesis_trackers::mht::config::MhtConfig;
use crate::hypothesis_trackers::mht::LogProbs;
use astrocap_core::{structs::Detection, traits::Dumpable};
use std::sync::LazyLock;

static NEXT_KF_ID: AtomicU64 = AtomicU64::new(1);

// State transition matrix F. Assumes constant frame rate
#[rustfmt::skip]
pub static F: LazyLock<Matrix4<f32>> = LazyLock::new(|| Matrix4::from_row_slice(&[
    1.0, 0.0, 1.0, 0.0,
    0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]));

#[rustfmt::skip]
pub static F5: LazyLock<Matrix5<f32>> = LazyLock::new(|| Matrix5::from_row_slice(&[
    1.0, 0.0, 1.0, 0.0, 0.0, // x ← x + vx
    0.0, 1.0, 0.0, 1.0, 0.0, // y ← y + vy
    0.0, 0.0, 1.0, 0.0, 0.0, // vx
    0.0, 0.0, 0.0, 1.0, 0.0, // vy
    0.0, 0.0, 0.0, 0.0, 1.0, // a (random walk)
]));

// Static measurement matrix H: projects state [x, y, vx, vy] into [x, y]
#[rustfmt::skip]
pub static H: LazyLock<Matrix2x4<f32>> = LazyLock::new(|| Matrix2x4::new(
    1.0, 0.0, 0.0, 0.0, 
    0.0, 1.0, 0.0, 0.0
));

#[rustfmt::skip]
pub static H5: LazyLock<Matrix3x5<f32>> = LazyLock::new(|| Matrix3x5::new(
    1.0, 0.0, 0.0, 0.0, 0.0, // measure x
    0.0, 1.0, 0.0, 0.0, 0.0, // measure y
    0.0, 0.0, 0.0, 0.0, 1.0, // measure a
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

// Process noise for 5D: CV on (x,y,vx,vy) + random-walk on a
#[inline]
fn q_cv5(dt: f32, q_pos: f32, q_amp_var_per_frame: f32) -> Matrix5<f32> {
    let q_xy = q_cv(dt, q_pos); // 4×4 CV block
    let mut q5 = Matrix5::<f32>::zeros();
    q5.fixed_slice_mut::<4, 4>(0, 0).copy_from(&q_xy);

    // Random-walk on amplitude: Q_aa = dt * (variance per frame)
    q5[(4, 4)] = dt * q_amp_var_per_frame;
    q5
}

// Baselines (tweak in config later if you like)
const BASELINE_ACCEL_VARIANCE: f32 = 1e-3; // for x,y CV
const BASELINE_AMP_DRIFT_VAR: f32 = 3.0; // σ≈3 amplitude units per frame → var=9

/// Structure holding KF state + covariance + adaptive noise scales
#[derive(Clone, Debug)]
pub struct KalmanFilter {
    pub id: u64,
    pub state: Vector5<f32>,      // [x, y, vx, vy, a]
    pub covariance: Matrix5<f32>, // 4×4 covariance of the state estimate

    // Measurement covariance (R). We adapt this online.
    pub r: Matrix3<f32>,

    // Scale applied to process noise template (Q_base).
    pub q_pos_scale: f32,
    pub q_amp_scale: f32,

    // Exponential moving average of NIS (tracks consistency over time).
    pub nis_ewma: f32,

    // adaptation parameters
    pub q_pos_min: f32,
    pub q_pos_max: f32,
    pub q_pos_eta: f32,

    pub q_amp_min: f32,
    pub q_amp_max: f32,
    pub q_amp_eta: f32,

    pub r_pos_min: f32,
    pub r_pos_max: f32,
    pub r_pos_beta: f32,

    // amplitude-specific adaptation
    pub r_amp_min: f32,  // std floor for r a
    pub r_amp_max: f32,  // std ceil for r a
    pub r_amp_beta: f32, // smoothing for a (usually > r_beta)
}

#[derive(serde::Deserialize, serde::Serialize)]
pub struct KalmanFilterRow {
    pub id: u64,
    pub run_id: u64,
    pub frame_index: usize,

    pub state: [f32; 5],
    pub covariance: [f32; 25],
    pub r: [f32; 9],
    pub q_pos_scale: f32,
    pub q_amp_scale: f32,
    pub nis_ewma: f32,
}

impl Dumpable for KalmanFilter {
    type Row = KalmanFilterRow;

    const TABLE_NAME: &'static str = "kalman_filter";
    const VERSION: u32 = 1;
    fn to_row(&self, run_id: u64, frame_index: usize) -> Self::Row {
        KalmanFilterRow {
            run_id,
            frame_index,
            id: self.id,
            state: self.state.as_slice().try_into().unwrap(),
            covariance: self.covariance.as_slice().try_into().unwrap(),
            r: self.r.as_slice().try_into().unwrap(),
            q_pos_scale: self.q_pos_scale,
            q_amp_scale: self.q_amp_scale,
            nis_ewma: self.nis_ewma,
        }
    }
}

impl KalmanFilter {
    pub fn new_from_detection(detection: &Detection) -> Self {
        Self::new_with_velocity(Vector5::new(
            detection.position.x,
            detection.position.y,
            0.0,
            0.0,
            detection.amplitude,
        ))
    }

    pub fn new_with_velocity(state: Vector5<f32>) -> Self {
        let r_x_init = 0.25f32;
        let r_y_init = 0.25f32;
        let r_a_init = 20.0f32;

        Self {
            id: NEXT_KF_ID.fetch_add(1, Ordering::Relaxed),
            state,
            covariance: Matrix5::identity(),

            // r: Matrix2::from_diagonal(&Vector2::new(0.25, 0.25)),
            r: Matrix3::from_diagonal(&Vector3::new(
                r_x_init.powi(2),
                r_y_init.powi(2),
                r_a_init.powi(2),
            )),

            // q_scale: 1.0,
            nis_ewma: 0.0,

            q_pos_scale: 1e-4, // small but nonzero, stars move slowly
            q_amp_scale: 1e-3,

            q_pos_min: 1e-6, // never let Q vanish completely
            q_pos_max: 1e-2, // allow modest growth if stars are lost for long
            q_pos_eta: 0.01, // 1% adaptation step per update

            q_amp_min: 1.0e-5,
            q_amp_max: 1.0,
            q_amp_eta: 0.02,

            r_pos_min: 0.25,  // sqrt variance = 0.5 pixels
            r_pos_max: 4.0,   // sqrt variance = 4 pixels
            r_pos_beta: 0.02, // 2% smoothing toward empirical variance

            r_amp_min: 5.0,
            r_amp_max: 70.0,
            r_amp_beta: 0.04,
        }
    }

    pub fn new_with_covariance(state: Vector5<f32>, covariance: Matrix5<f32>) -> Self {
        let r_x_init = 0.25f32;
        let r_y_init = 0.25f32;
        let r_a_init = 20.0f32;

        Self {
            id: NEXT_KF_ID.fetch_add(1, Ordering::Relaxed),
            state,
            covariance,

            // r: Matrix2::from_diagonal(&Vector2::new(0.25, 0.25)),
            r: Matrix3::from_diagonal(&Vector3::new(
                r_x_init.powi(2),
                r_y_init.powi(2),
                r_a_init.powi(2),
            )),

            // q_scale: 1.0,
            nis_ewma: 0.0,

            q_pos_scale: 1e-4, // small but nonzero, stars move slowly
            q_pos_min: 1e-6,   // never let Q vanish completely
            q_pos_max: 1e-2,   // allow modest growth if stars are lost for long
            q_pos_eta: 0.01,   // 1% adaptation step per update

            q_amp_scale: 1e-3,
            q_amp_min: 1.0e-5,
            q_amp_max: 1.0,
            q_amp_eta: 0.02,

            r_pos_min: 0.25,  // sqrt variance = 0.5 pixels
            r_pos_max: 4.0,   // sqrt variance = 4 pixels
            r_pos_beta: 0.02, // 2% smoothing toward empirical variance

            r_amp_min: 5.0,
            r_amp_max: 70.0,
            r_amp_beta: 0.04,
        }
    }

    pub fn apply_measurement_from_config(&mut self, cfg: &MhtConfig) {
        // init R from std → variance
        self.r = Matrix3::from_diagonal(&Vector3::new(
            cfg.r_x * cfg.r_x,
            cfg.r_y * cfg.r_y,
            cfg.r_a * cfg.r_a,
        ));

        // store per-dimension clamp bounds and gains (stds)
        self.q_pos_min = cfg.q_pos_min;
        self.q_pos_max = cfg.q_pos_max;

        self.r_pos_min = cfg.r_pos_min;
        self.r_pos_max = cfg.r_pos_max;
        self.r_pos_beta = cfg.r_pos_beta;

        self.r_amp_min = cfg.r_amp_min;
        self.r_amp_max = cfg.r_amp_max;
        self.r_amp_beta = cfg.r_amp_beta;
    }

    pub fn x(&self) -> f32 {
        self.state.x
    }

    pub fn y(&self) -> f32 {
        self.state.y
    }

    pub fn a(&self) -> f32 {
        self.state[4]
    }

    pub fn covariance(&self) -> &Matrix5<f32> {
        &self.covariance
    }

    pub fn r(&self) -> &Matrix3<f32> {
        &self.r
    }

    pub fn predict(&self, dt: f32) -> Self {
        // Process noise Q (scaled by self.q_scale).
        let q = q_cv5(
            dt,
            self.q_pos_scale * BASELINE_ACCEL_VARIANCE,
            self.q_amp_scale * BASELINE_AMP_DRIFT_VAR,
        );
        // let q = q_cv(1.0, /* tune me */ 1e-5); // start around 1e-3, adjust later

        // Predict state
        let state = *F5 * self.state;

        // Predict covariance
        let covariance = *F5 * self.covariance * F5.transpose() + q;

        Self {
            id: NEXT_KF_ID.fetch_add(1, Ordering::Relaxed),
            state,
            covariance,

            r: self.r,
            q_pos_scale: self.q_pos_scale,
            nis_ewma: self.nis_ewma,

            q_pos_min: self.q_pos_min,
            q_pos_max: self.q_pos_max,
            q_pos_eta: self.q_pos_eta,

            q_amp_scale: self.q_amp_scale,
            q_amp_min: self.q_amp_min,
            q_amp_max: self.q_amp_max,
            q_amp_eta: self.q_amp_eta,

            r_pos_min: self.r_pos_min,
            r_pos_max: self.r_pos_max,
            r_pos_beta: self.r_pos_beta,

            r_amp_min: self.r_amp_min,
            r_amp_max: self.r_amp_max,
            r_amp_beta: self.r_amp_beta,
        }
    }

    /// Update step (measurement incorporated).
    /// Uses Joseph form for numerical stability.
    pub fn update(&mut self, detection: &Detection) -> bool {
        let Some(k) = self.kalman_gain() else {
            return false; // S not invertible
        };

        // Measurement vector z = [x_meas, y_meas]
        let z = Vector3::new(
            detection.position.x,
            detection.position.y,
            detection.amplitude,
        );

        // Innovation residual (difference between measurement and prediction)
        let r = z - *H5 * self.state;

        // State update
        self.state = self.state + k * r;

        // Joseph form covariance update:
        // P ← (I - KH) P (I - KH)ᵀ + K R Kᵀ
        let i5 = Matrix5::identity();
        let i_minus_kh = i5 - k * *H5;
        self.covariance =
            i_minus_kh * self.covariance * i_minus_kh.transpose() + k * self.r * k.transpose();

        // Defensive re-symmetrisation (avoid small asymmetries from FP noise).
        self.covariance = 0.5 * (self.covariance + self.covariance.transpose());
        true
    }

    /// Kalman gain computation.
    /// We factor this out so update() can be written in Joseph form.
    pub fn kalman_gain(&self) -> Option<Matrix5x3<f32>> {
        let s = *H5 * self.covariance * H5.transpose() + self.r;
        let s_inv = safe_invert_3x3(s)?;
        let k = self.covariance * H5.transpose() * s_inv;
        Some(k)
    }

    /// Innovation calculation (residual, covariance S, and NIS).
    /// Called before gating / association.
    pub fn innovation(&self, detection: &Detection) -> (Vector3<f32>, Matrix3<f32>, f32) {
        // Measurement vector
        let z = Vector3::new(
            detection.position.x,
            detection.position.y,
            detection.amplitude,
        );

        // Predicted measurement
        let z_pred = *H5 * self.state;

        // Innovation residual
        let r = z - z_pred;

        // Innovation covariance
        let mut s = *H5 * self.covariance * H5.transpose() + self.r;

        // Symmetrise and add jitter to ensure PD
        s = s.symmetric_part() + Matrix3::identity() * 1e-6;

        // Attempt Cholesky factorisation
        if let Some(chol) = s.cholesky() {
            let y = chol.solve(&r);
            let nis = y.dot(&y);
            (r, s, nis)
        } else {
            // Fallback: not PD, use pseudo-inverse
            tracing::warn!("Innovation covariance not PD: {:?}", s);
            let s_inv = s.try_inverse().unwrap_or(Matrix3::identity());
            let nis = (r.transpose() * s_inv * r)[0];
            (r, s, nis)
        }
    }

    /// Association confidence calculation.
    /// Returns log-likelihood (LL) and log-odds (LO).
    pub fn assoc_conf(
        &self,
        detection: &Detection,
        pd: f32,
        clutter_rate: f32,
        parent_id: u64,
        parent_cum_lo: f32,
    ) -> (LogProbs, f32, Vector3<f32>, Matrix3<f32>) {
        // Measurement vector
        let z = Vector3::new(
            detection.position.x,
            detection.position.y,
            detection.amplitude,
        );

        // Predicted measurement
        let z_pred = *H5 * self.state;

        // Residual
        let residual = z - z_pred;

        // Innovation covariance S
        // S = H P Hᵀ + R
        let s = *H5 * self.covariance * H5.transpose() + self.r;

        // invert safely
        let s_inv = match safe_invert_3x3(s) {
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

        // 2D case:
        // let det_s = (s[(0, 0)] * s[(1, 1)] - s[(0, 1)] * s[(1, 0)]).max(1e-12);

        // 3D case:
        let det_s = s.determinant().max(1e-12);

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

        // Self::maybe_log_nis_breakdown(
        //     parent_id,
        //     parent_cum_lo,
        //     detection.amplitude,
        //     z_pred[2],
        //     &self.r,
        //     &residual,
        //     &s,
        //     nis,
        //     6.0,    // cutoff for x/y NIS
        //     6.0,    // cutoff for amplitude NIS
        //     1000.0, // only log once track is well-established
        // );

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

    /// Logs a NIS breakdown for high-confidence tracks if any component is concerning.
    /// Helps diagnose "fizzers" (bright stars spawning froth).
    #[inline]
    fn maybe_log_nis_breakdown(
        track_id: u64,
        cum_lo: f32,
        detection_amp: f32,
        pred_amp: f32,
        r: &Matrix3<f32>,
        residual: &Vector3<f32>,
        s: &Matrix3<f32>,
        nis: f32,
        nis_cutoff_xy: f32,
        nis_cutoff_a: f32,
        cum_lo_cutoff: f32,
    ) {
        if cum_lo < cum_lo_cutoff {
            return; // only care about strong/confident tracks
        }

        // Normalised residuals (per-dimension NIS contributions)
        let nis_x = residual.x.powi(2) / s[(0, 0)].max(1e-6);
        let nis_y = residual.y.powi(2) / s[(1, 1)].max(1e-6);
        let nis_a = residual.z.powi(2) / s[(2, 2)].max(1e-6);

        if nis_x > nis_cutoff_xy || nis_y > nis_cutoff_xy || nis_a > nis_cutoff_a {
            tracing::warn!(
                track_id,
                cum_lo,
                det_amp = detection_amp,
                pred_amp,
                r_x = %r[(0,0)].sqrt(),
                r_y = %r[(1,1)].sqrt(),
                r_a = %r[(2,2)].sqrt(),
                nis_total = nis,
                nis_x,
                nis_y,
                nis_a,
                "Confident track with concerning NIS component"
            );
        }
    }

    /// Adapt Q (process noise scale) based on mean NIS.
    pub fn adapt_q_from_mean(&mut self, mean_nis: f32) {
        // Target NIS ~ 2 for 2D measurements, 3 for 3D.
        let target = 3.0;

        // EWMA of NIS to smooth fluctuations
        let alpha = 0.05;
        self.nis_ewma = (1.0 - alpha) * self.nis_ewma + alpha * mean_nis;

        // Adjustment factor: push q_scale up if NIS < target, down if > target
        let ratio = (self.nis_ewma / target).clamp(0.5, 2.0);
        self.q_pos_scale *= ratio;

        // Clamp q_scale to avoid collapse or explosion
        self.q_pos_scale = self.q_pos_scale.clamp(1e-8, 1e-2);

        if (self.q_pos_scale - self.q_pos_min).abs() < 1e-12 {
            tracing::debug!(q_scale = self.q_pos_scale, "Q scale at min");
        }
        if (self.q_pos_scale - self.q_pos_max).abs() < 1e-12 {
            tracing::debug!(q_scale = self.q_pos_scale, "Q scale at max");
        }
    }

    /// Very gentle multiplicative inflation of Q when we missed a detection.
    /// This keeps tracks "alive" over long gaps (e.g. dim stars vanishing for 50 frames).
    pub fn adapt_q_on_miss(&mut self) {
        // Small inflation factor
        let inflation = 1.02; // 2% per miss
        self.q_pos_scale = (self.q_pos_scale * inflation).min(self.q_pos_max);

        if (self.q_pos_scale - self.q_pos_min).abs() < 1e-12 {
            tracing::debug!(q_scale = self.q_pos_scale, "Q scale at min");
        }
        if (self.q_pos_scale - self.q_pos_max).abs() < 1e-12 {
            tracing::debug!(q_scale = self.q_pos_scale, "Q scale at max");
        }
    }

    /// Adapt Q online based on NIS vs. expected measurement dimension.
    /// If NIS > dim, process noise is too small → increase q_scale.
    /// If NIS < dim, process noise too large → decrease q_scale.
    pub fn adapt_q_from_nis(&mut self, nis: f32, meas_dim: f32) {
        if !nis.is_finite() || nis <= 0.0 {
            return; // ignore garbage
        }

        let err_ratio = (nis / meas_dim).clamp(0.25, 4.0);
        let step = self.q_pos_eta; // e.g. 0.01
        self.q_pos_scale *= 1.0 + step * (err_ratio - 1.0);

        // clamp to configured safe range
        self.q_pos_scale = self.q_pos_scale.clamp(self.q_pos_min, self.q_pos_max);

        if (self.q_pos_scale - self.q_pos_min).abs() < 1e-12 {
            tracing::debug!(q_scale = self.q_pos_scale, "Q pos scale at min");
        }
        if (self.q_pos_scale - self.q_pos_max).abs() < 1e-12 {
            tracing::debug!(q_scale = self.q_pos_scale, "Q pos scale at max");
        }
    }

    pub fn adapt_q_amp_from_residual(&mut self, nis_a: f32) {
        if !nis_a.is_finite() || nis_a <= 0.0 {
            return;
        }
        let target = 1.0; // 1 DOF (amplitude)
        let err_ratio = (nis_a / target).clamp(0.25, 4.0);
        self.q_amp_scale *= 1.0 + self.q_amp_eta * (err_ratio - 1.0);
        self.q_amp_scale = self.q_amp_scale.clamp(self.q_amp_min, self.q_amp_max);

        if (self.q_amp_scale - self.q_amp_min).abs() < 1e-12 {
            tracing::debug!(q_scale = self.q_amp_scale, "Q amp scale at min");
        }
        if (self.q_amp_scale - self.q_amp_max).abs() < 1e-12 {
            tracing::debug!(q_scale = self.q_amp_scale, "Q amp scale at max");
        }
    }

    /// Adapt R from a single residual (x,y,a) with per-dimension gains and clamps.
    /// `s_pred` is optional “guardrail” (innovation covariance) — we’ll only use its diagonals.
    pub fn adapt_r_from_pair(&mut self, residual: &Vector3<f32>, s_pred: &Matrix3<f32>) {
        // per-dimension gains
        let beta_xy = self.r_pos_beta; // e.g. 0.02
        let beta_a = self.r_amp_beta.max(self.r_pos_beta); // e.g. ~0.04

        // residual variances (one-sample estimates)
        let rx = residual.x * residual.x;
        let ry = residual.y * residual.y;
        let ra = residual.z * residual.z;

        // optional guardrail from S (diagonal only)
        let sx = s_pred[(0, 0)].max(0.0);
        let sy = s_pred[(1, 1)].max(0.0);
        let sa = s_pred[(2, 2)].max(0.0);

        let mut r_new = self.r;

        // smooth toward (residual^2 + small guardrail)
        r_new[(0, 0)] = (1.0 - beta_xy) * r_new[(0, 0)] + beta_xy * (rx + 0.1 * sx);
        r_new[(1, 1)] = (1.0 - beta_xy) * r_new[(1, 1)] + beta_xy * (ry + 0.1 * sy);
        r_new[(2, 2)] = (1.0 - beta_a) * r_new[(2, 2)] + beta_a * (ra + 0.1 * sa);

        // clamp in VARIANCE domain
        let r_min_var_xy = self.r_pos_min * self.r_pos_min;
        let r_max_var_xy = self.r_pos_max * self.r_pos_max;
        let r_min_var_a = self.r_amp_min * self.r_amp_min;
        let r_max_var_a = self.r_amp_max * self.r_amp_max;

        r_new[(0, 0)] = r_new[(0, 0)].clamp(r_min_var_xy, r_max_var_xy);
        r_new[(1, 1)] = r_new[(1, 1)].clamp(r_min_var_xy, r_max_var_xy);
        r_new[(2, 2)] = r_new[(2, 2)].clamp(r_min_var_a, r_max_var_a);

        // (optional) zero tiny off-diagonals to keep R diagonal
        r_new[(0, 1)] = 0.0;
        r_new[(1, 0)] = 0.0;
        r_new[(0, 2)] = 0.0;
        r_new[(2, 0)] = 0.0;
        r_new[(1, 2)] = 0.0;
        r_new[(2, 1)] = 0.0;

        self.r = r_new;

        // X floor/ceil checks
        let r_x = self.r[(0, 0)].sqrt();
        if (r_x - self.r_pos_min).abs() < 1e-3 {
            tracing::debug!(r_x, "R_x at floor");
        }
        if (r_x - self.r_pos_max).abs() < 1e-3 {
            tracing::debug!(r_x, "R_x at ceil");
        }

        // Y floor/ceil checks
        let r_y = self.r[(1, 1)].sqrt();
        if (r_y - self.r_pos_min).abs() < 1e-3 {
            tracing::debug!(r_y, "R_y at floor");
        }
        if (r_y - self.r_pos_max).abs() < 1e-3 {
            tracing::debug!(r_y, "R_y at ceil");
        }

        // Amplitude floor/ceil checks
        let r_a = self.r[(2, 2)].sqrt();
        if (r_a - self.r_amp_min).abs() < 1e-3 {
            tracing::debug!(r_a, "R_a at floor");
        }
        if (r_a - self.r_amp_max).abs() < 1e-3 {
            tracing::debug!(r_a, "R_a at ceil");
        }
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

#[inline]
fn safe_invert_3x3(mut s: Matrix3<f32>) -> Option<Matrix3<f32>> {
    // Enforce symmetry + small jitter to diagonals
    s = 0.5 * (s + s.transpose());
    for i in 0..3 {
        s[(i, i)] += 1e-6;
    }

    // Determinant
    let det = s.determinant();
    if !det.is_finite() || det.abs() < 1e-12 {
        return None;
    }

    // Inverse
    let inv = s.try_inverse()?;
    if !inv.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kf_predict_moves_forward() {
        // Initial state: (x=0, y=0, vx=1, vy=2, a=5)
        let init_state = Vector5::new(0.0, 0.0, 1.0, 2.0, 5.0);
        let kf = KalmanFilter::new_with_covariance(init_state, Matrix5::identity());

        let predicted = kf.predict(1.0);

        // Expect position advanced by velocity: x=1, y=2
        assert!((predicted.x() - 1.0).abs() < 1e-5);
        assert!((predicted.y() - 2.0).abs() < 1e-5);

        // Velocity should be unchanged
        assert!((predicted.state[2] - 1.0).abs() < 1e-5);
        assert!((predicted.state[3] - 2.0).abs() < 1e-5);

        // Amplitude should be unchanged
        assert!((predicted.state[4] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_kf_update_pulls_toward_detection() {
        use astrocap_core::structs::Detection;
        use nalgebra::Vector2;

        // Start at (0,0) with zero velocity and amplitude 0
        let init_state = Vector5::new(0.0, 0.0, 0.0, 0.0, 0.0);
        let mut kf = KalmanFilter::new_with_covariance(init_state, Matrix5::identity());

        // Measurement at (10,10) with amplitude 100
        let detection = Detection {
            id: 1,
            position: Vector2::new(10.0, 10.0),
            amplitude: 100.0,
        };

        kf.update(&detection);

        // After update, x and y should have shifted toward 10
        assert!(kf.x() > 0.0 && kf.x() < 10.0);
        assert!(kf.y() > 0.0 && kf.y() < 10.0);

        // Amplitude should have shifted toward 100
        assert!(kf.a() > 0.0 && kf.a() < 100.0);
    }
}
