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

    pub r_x: f32,
    pub r_y: f32,
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
            track_birth_rate: 0.00000002f32,

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
            clutter_rate: 0.00000123f32,

            threshold: 70.0,
            sigma: 4.45,

            max_track_group_count: 150,
            per_root_max_leaf_count: 10,
            per_root_min_leaf_log_odds: -15.0,
            min_track_log_odds: -10.0,

            // margin for gating births of new tracks. Track
            // will be born only if birth_ll - clutter_ll > birth_tau
            birth_tau: -10.0,

            // Kalman filter R
            // measurement variance / noise, in pixels
            r_x: 1.0,
            r_y: 1.0,
        }
    }
}
