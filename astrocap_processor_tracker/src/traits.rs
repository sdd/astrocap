use astrocap_core::AstrocapError;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use toml::Value;

use crate::model::{Detection, Track};

/// maps track indices to associated detection index
pub type Associations = HashMap<usize, usize>;

pub trait ConfigurableConfig:
    Default + for<'de> Deserialize<'de> + Send + Sync + Sized + 'static
{
    fn from_toml_value(value: &Value) -> Result<Self, AstrocapError> {
        value
            .clone()
            .try_into()
            .map_err(|e: toml::de::Error| AstrocapError::PluginInvalidConfigError(e.to_string()))
    }
}

pub trait Configurable: Send + Sync + 'static {
    type Config: ConfigurableConfig;

    fn from_config(cfg: Self::Config) -> Result<Box<Self>, AstrocapError>;
}

/// Initializes new tracks from unassociated detections
pub trait Initiator: Send + Sync + 'static {
    fn initiate(&mut self, detections: &[&Detection]) -> Vec<Track>;
}

/// Terminates tracks based on various criteria
pub trait Terminator: Send + Sync + 'static {
    fn should_terminate(&self, track: &Track) -> bool;
}

/// Handles data association between tracks and detections
pub trait Associator: Send + Sync + 'static {
    fn associate(
        &self,
        detections: &[Detection],
        tracks: &[Track],
    ) -> (Associations, HashSet<usize>);
}

/// Updates track state with new measurements
pub trait Updater: Send + Sync + 'static {
    fn update(&self, track: &mut Track, detection: &Detection);
}

/// Predicts track state forward in time
pub trait Predictor: Send + Sync + 'static {
    fn predict(&self, track: &mut Track);
}
