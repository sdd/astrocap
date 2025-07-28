use az::{Az, Cast};

#[derive(Debug)]
pub struct ModelConfig<F> {
    pub min_reqd_qty_to_attempt_solve: usize,
    pub max_existing_candidate_match_dist: F,
    pub min_new_star_candidate_score: F,
    pub max_star_candidate_radius: F,

    pub star_candidate_strong_match_bonus: F,
    pub star_candidate_strong_match_threshold: F,
    pub star_candidate_long_term_match_threshold: F,

    pub star_candidate_unmatched_penalty: F,
    pub amplitude_penalty_threshold: F,
    pub amplitude_penalty: F,
    pub star_candidate_discard_threshold: F,
    pub detected_point_candidate_amplitude_threshold: F,
}

impl<F> Default for ModelConfig<F>
where
    f64: Cast<F>,
{
    fn default() -> Self {
        Self {
            min_reqd_qty_to_attempt_solve: 6,
            max_existing_candidate_match_dist: 15f64.az::<F>(),
            min_new_star_candidate_score: 0f64.az::<F>(),
            max_star_candidate_radius: 4.5f64.az::<F>(),

            star_candidate_strong_match_bonus: 20f64.az::<F>(),
            star_candidate_strong_match_threshold: 5.0f64.az::<F>(),
            star_candidate_long_term_match_threshold: 1000.0f64.az::<F>(),

            // star_candidate_unmatched_penalty: 2.5f64.az::<F>(),
            star_candidate_unmatched_penalty: 5f64.az::<F>(),
            amplitude_penalty_threshold: 8.0f64.az::<F>(),
            amplitude_penalty: 0.0f64.az::<F>(),

            // star_candidate_discard_threshold: (-10.0f64).az::<F>(),
            star_candidate_discard_threshold: 0f64.az::<F>(),
            detected_point_candidate_amplitude_threshold: 20.0f64.az::<F>(),
        }
    }
}
