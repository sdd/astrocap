#[derive(Debug)]
pub struct ModelConfig {
    pub min_reqd_qty_to_attempt_solve: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            min_reqd_qty_to_attempt_solve: 6,
        }
    }
}
