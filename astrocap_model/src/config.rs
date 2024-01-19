use az::{Az, Cast};

#[derive(Debug)]
pub struct ModelConfig<F> {
    pub min_reqd_qty_to_attempt_solve: usize,
    pub max_existing_candidate_match_dist: F,
    pub min_new_star_candidate_score: F,
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
            max_existing_candidate_match_dist: 7.5f64.az::<F>(),
            min_new_star_candidate_score: 0f64.az::<F>(),
            star_candidate_unmatched_penalty: 5.0f64.az::<F>(),
            amplitude_penalty_threshold: 8.0f64.az::<F>(),
            amplitude_penalty: 15.0f64.az::<F>(),
            star_candidate_discard_threshold: (-10.0f64).az::<F>(),
            detected_point_candidate_amplitude_threshold: 30.0f64.az::<F>(),
        }
    }
}
