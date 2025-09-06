use crate::model::Track;
use crate::traits::{Configurable, ConfigurableConfig, Terminator};
use astrocap_core::AstrocapError;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    threshold: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self { threshold: 0.05 }
    }
}

impl ConfigurableConfig for Config {}

pub struct SimpleTerminator {
    config: Config,
}

impl Configurable for SimpleTerminator {
    type Config = Config;

    fn from_config(config: Self::Config) -> Result<Box<Self>, AstrocapError> {
        Ok(Box::new(Self { config }))
    }
}

impl Terminator for SimpleTerminator {
    fn should_terminate(&self, track: &Track) -> bool {
        track.confidence < self.config.threshold
    }
}
