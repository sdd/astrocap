use crate::model::{Detection, Track};
use crate::traits::{Configurable, ConfigurableConfig, Initiator};
use astrocap_core::AstrocapError;
use ordered_float::OrderedFloat;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    threshold: f32,
    radius: f32,
    initial_confidence: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            threshold: 50.0,
            radius: 5.0,
            initial_confidence: 0.5,
        }
    }
}

impl ConfigurableConfig for Config {}

pub struct ThresholdInitiator {
    config: Config,
}

impl Configurable for ThresholdInitiator {
    type Config = Config;

    fn from_config(config: Self::Config) -> Result<Box<Self>, AstrocapError> {
        Ok(Box::new(Self { config }))
    }
}

impl Initiator for ThresholdInitiator {
    fn initiate(&mut self, detections: &[&Detection]) -> Vec<Track> {
        // reject points with amplitude < threshold
        let mut filtered_detections = detections
            .iter()
            .filter(|d| d.amplitude > 0.0)
            .collect::<Vec<_>>();

        // sort by descending amplitude
        filtered_detections.sort_unstable_by_key(|d| -OrderedFloat(d.amplitude));

        // TODO:
        // reject points within specified radius of brighter candidate (or existing track?)

        filtered_detections
            .iter()
            .map(|d| Track::from_detection(d, self.config.initial_confidence))
            .collect()
    }
}
