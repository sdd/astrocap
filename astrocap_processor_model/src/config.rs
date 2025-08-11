#[derive(Debug)]
pub struct ModelConfig {
    pub min_reqd_qty_to_attempt_solve: usize,
    pub max_existing_candidate_match_dist: f32,
    pub min_new_star_candidate_score: f32,
    pub max_star_candidate_radius: f32,

    pub star_candidate_strong_match_bonus: f32,
    pub star_candidate_strong_match_threshold: f32,
    pub star_candidate_long_term_match_threshold: f32,

    pub star_candidate_unmatched_penalty: f32,
    pub amplitude_penalty_threshold: f32,
    pub amplitude_penalty: f32,
    pub star_candidate_discard_threshold: f32,
    pub detected_point_candidate_amplitude_threshold: f32,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            min_reqd_qty_to_attempt_solve: 6,
            max_existing_candidate_match_dist: 15f32,
            min_new_star_candidate_score: 0f32,
            max_star_candidate_radius: 4.5f32,

            star_candidate_strong_match_bonus: 20f32,
            star_candidate_strong_match_threshold: 5.0f32,
            star_candidate_long_term_match_threshold: 1000.0f32,

            // star_candidate_unmatched_penalty: 2.5f32,
            star_candidate_unmatched_penalty: 5f32,
            amplitude_penalty_threshold: 8.0f32,
            amplitude_penalty: 0.0f32,

            // star_candidate_discard_threshold: -10.0f32,
            star_candidate_discard_threshold: 0f32,
            detected_point_candidate_amplitude_threshold: 20.0f32,
        }
    }
}
