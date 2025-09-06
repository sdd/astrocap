use crate::hypothesis_trackers::mht::MultiHypothesisTracker;
use crate::hypothesis_trackers::sht::SingleHypothesisTracker;
use crate::traits::HypothesisTracker;
use astrocap_core::AstrocapError;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub(crate) struct Config {
    pub(crate) tracker_strategy: TrackerStrategy,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tracker_strategy: TrackerStrategy::SingleHypothesis,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub enum TrackerStrategy {
    SingleHypothesis,
    MultiHypothesis,
}

impl TrackerStrategy {
    pub fn create_tracker(
        &self,
        raw_config: Option<&toml::Value>,
    ) -> Result<Box<dyn HypothesisTracker>, AstrocapError> {
        match self {
            TrackerStrategy::SingleHypothesis => {
                Ok(Box::new(SingleHypothesisTracker::new(raw_config)?))
            }
            TrackerStrategy::MultiHypothesis => {
                Ok(Box::new(MultiHypothesisTracker::new(raw_config)?))
            }
        }
    }
}
