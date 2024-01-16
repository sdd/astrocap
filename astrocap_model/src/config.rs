use az::{Az, Cast};

#[derive(Debug)]
pub struct ModelConfig<F> {
    pub min_reqd_qty_to_attempt_solve: usize,
    pub max_existing_candidate_match_dist: F,
}

impl<F> Default for ModelConfig<F>
where
    f64: Cast<F> {
    fn default() -> Self {
        Self {
            min_reqd_qty_to_attempt_solve: 6,
            max_existing_candidate_match_dist: 7.5f64.az::<F>(),
        }
    }
}
