use super::elements::{
    associators::nearest::NearestAssociator, initiators::threshold::ThresholdInitiator,
    predictors::kalman::Kalman as KalmanPredictor, terminators::simple::SimpleTerminator,
    updaters::kalman::Kalman as KalmanUpdater,
};
use crate::traits::{
    Associator, Configurable, ConfigurableConfig, Initiator, Predictor, Terminator, Updater,
};
use astrocap_core::AstrocapError;
use serde::Deserialize;
use toml::Value;

type Result<T> = std::result::Result<T, AstrocapError>;

#[derive(Default, Deserialize)]
#[serde(default)]
struct ComponentConfig {
    initiator: InitiatorType,
    terminator: TerminatorType,
    associator: AssociatorType,
    updater: UpdaterType,
    predictor: PredictorType,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum InitiatorType {
    #[default]
    Threshold,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum TerminatorType {
    #[default]
    Simple,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum AssociatorType {
    #[default]
    Nearest,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum UpdaterType {
    #[default]
    Kalman,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "lowercase")]
enum PredictorType {
    #[default]
    Kalman,
}

pub(crate) struct SingleHypothesisTrackerElementFactory<'a> {
    raw: &'a Value,
    config: ComponentConfig,
}

impl<'a> SingleHypothesisTrackerElementFactory<'a> {
    pub(crate) fn new(raw: &'a Value) -> Result<Self> {
        Ok(Self {
            raw,
            config: raw.clone().try_into()?,
        })
    }

    pub(crate) fn create_initiator(&self) -> Result<Box<dyn Initiator>> {
        match self.config.initiator {
            InitiatorType::Threshold => {
                let config =
                    <ThresholdInitiator as Configurable>::Config::from_toml_value(self.raw)?;
                Ok(ThresholdInitiator::from_config(config)?)
            }
        }
    }

    pub(crate) fn create_terminator(&self) -> Result<Box<dyn Terminator>> {
        match self.config.terminator {
            TerminatorType::Simple => {
                let config = <SimpleTerminator as Configurable>::Config::from_toml_value(self.raw)?;
                Ok(SimpleTerminator::from_config(config)?)
            }
        }
    }

    pub(crate) fn create_associator(&self) -> Result<Box<dyn Associator>> {
        match self.config.associator {
            AssociatorType::Nearest => {
                let config =
                    <NearestAssociator as Configurable>::Config::from_toml_value(self.raw)?;
                Ok(NearestAssociator::from_config(config)?)
            }
        }
    }

    pub(crate) fn create_updater(&self) -> Result<Box<dyn Updater>> {
        match self.config.updater {
            UpdaterType::Kalman => {
                let config = <KalmanUpdater as Configurable>::Config::from_toml_value(self.raw)?;
                Ok(KalmanUpdater::from_config(config)?)
            }
        }
    }

    pub(crate) fn create_predictor(&self) -> Result<Box<dyn Predictor>> {
        match self.config.predictor {
            PredictorType::Kalman => {
                let config = <KalmanPredictor as Configurable>::Config::from_toml_value(self.raw)?;
                Ok(KalmanPredictor::from_config(config)?)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use toml;

    #[test]
    fn test_create_default_components() {
        let config = toml::Value::Table(toml::map::Map::new());

        let factory = SingleHypothesisTrackerElementFactory::new(&config).unwrap();

        // All components should be creatable with empty config (using defaults)
        assert!(factory.create_initiator().is_ok());
        assert!(factory.create_terminator().is_ok());
        assert!(factory.create_associator().is_ok());
        assert!(factory.create_updater().is_ok());
        assert!(factory.create_predictor().is_ok());
    }

    #[test]
    fn test_create_configured_components() {
        // Create a config with explicit types
        let config_toml = r#"
            initiator = "threshold"
            terminator = "simple"
            associator = "nearest"
            updater = "kalman"
            predictor = "kalman"
            
            # Component-specific configuration would go here
            threshold = 100.0
            max_age = 10
            max_distance = 5.0
        "#;

        let config: Value = toml::from_str(config_toml).unwrap();

        let factory = SingleHypothesisTrackerElementFactory::new(&config).unwrap();

        // All components should be creatable with explicit config
        assert!(factory.create_initiator().is_ok());
        assert!(factory.create_terminator().is_ok());
        assert!(factory.create_associator().is_ok());
        assert!(factory.create_updater().is_ok());
        assert!(factory.create_predictor().is_ok());
    }

    #[test]
    fn test_invalid_component_type() {
        let config_toml = r#"
            initiator = "nonexistent"
        "#;

        let config: Value = toml::from_str(config_toml).unwrap();

        // Should fail with unknown type
        assert!(SingleHypothesisTrackerElementFactory::new(&config).is_err());
    }
}
