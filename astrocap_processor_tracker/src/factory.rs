use astrocap_core::AstrocapError;

use crate::config::Config;
use crate::traits::{Associator, Initiator, Predictor, Terminator, Updater};

type Result<T> = std::result::Result<T, AstrocapError>;

pub(crate) struct Factory {}

impl Factory {
    pub(crate) fn create_initiator(config: &Config) -> Result<Box<dyn Initiator>> {
        todo!();
    }

    pub(crate) fn create_terminator(config: &Config) -> Result<Box<dyn Terminator>> {
        todo!();
    }

    pub(crate) fn create_associator(config: &Config) -> Result<Box<dyn Associator>> {
        todo!();
    }

    pub(crate) fn create_updater(config: &Config) -> Result<Box<dyn Updater>> {
        todo!();
    }

    pub(crate) fn create_predictor(config: &Config) -> Result<Box<dyn Predictor>> {
        todo!();
    }
}
