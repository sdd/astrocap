use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct MhtConfig {
    pub association_gate_radius: f32,

    pub track_birth_rate: f32,

    pub clutter_rate: f32,

    pub threshold: f32,

    pub sigma: f32,

    pub max_track_group_count: usize,
    pub per_root_max_leaf_count: usize,
    pub per_root_min_leaf_log_odds: f32,
    pub min_track_log_odds: f32,

    pub birth_tau: f32,
    pub gating_chi2: f32,

    pub r_x: f32,
    pub r_y: f32,

    // NEW: coast-aware noise adaptation
    pub q_base: f32,          // tiny jitter when tracking cleanly
    pub q_coast_floor: f32,   // minimum diffusion while coasting (lets gates grow)
    pub q_ramp_frames: usize, // frames to ramp from q_base -> q_coast_floor when missing

    pub r_floor: f32,         // min std px (bright stars lock precisely)
    pub r_ceil: f32,          // max std px (bad centroid periods)
    pub r_coast_start: usize, // start inflating R after this many consecutive misses
    pub r_coast_growth: f32,  // multiplicative growth per extra miss (e.g. 0.03 = +3%)
}

impl Default for MhtConfig {
    fn default() -> Self {
        Self {
            association_gate_radius: 6.0,

            /*
               * rough birth rate per frame = 0.01

               * num_independent_cells = pixel_count / PSF Area
                                         = (1920 * 1080) / (3 * 3)
                                         = 4.07e5
               * birth rate density = 0.01 / 4.07e5
                                    = 2e-8 (0.00000002), or -7.61 as log
            */
            track_birth_rate: 0.0000002f32,

            /*
               * Rough estimate of false detections per frame
                    assuming FHD and threshold = 70: Λ = 50

               * Convert to per-cell clutter intensity for association formulas:

                    num_independent_cells = pixel_count / PSF Area
                                          = (1920 * 1080) / (3 * 3)
                                          = 4.07e5

                    λ_cell = Λ / num_independent_cells
                           = 50 / 4.07e5
                           = 1.23e-4 (-9 as log)
            */
            // clutter_rate: 0.00000123f32,
            clutter_rate: 0.000123f32,

            threshold: 70.0,
            sigma: 4.45,

            max_track_group_count: 150,
            per_root_max_leaf_count: 5,
            per_root_min_leaf_log_odds: -15.0,
            min_track_log_odds: -10.0,

            // margin for gating births of new tracks. Track
            // will be born only if birth_ll - clutter_ll > birth_tau
            birth_tau: -10.0,

            // 95% in 2D
            gating_chi2: 5.99,

            // Kalman filter R
            // measurement variance / noise, in pixels
            r_x: 1.0,
            r_y: 1.0,

            // CV stars @ 25–50fps: very small jitter while locked,
            // but enough diffusion to reacquire after long gaps.
            q_base: 1e-7,
            q_coast_floor: 6e-4, // ~5px std after ~50 coasts if v is very certain
            q_ramp_frames: 10,

            r_floor: 0.5,
            r_ceil: 2.5,
            r_coast_start: 5,
            r_coast_growth: 0.03,
        }
    }
}
