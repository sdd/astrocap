use astrocap_core::AstrocapError;
use toml::Value;

use crate::traits::{
    Associator, Configurable, ConfigurableConfig, Initiator, Predictor, Terminator, Updater,
};

// Import all available implementations
use super::elements::{
    associators::nearest::NearestAssociator, initiators::threshold::ThresholdInitiator,
    predictors::kalman::Kalman as KalmanPredictor, terminators::simple::SimpleTerminator,
    updaters::kalman::Kalman as KalmanUpdater,
};

type Result<T> = std::result::Result<T, AstrocapError>;

pub(crate) struct SingleHypothesisTrackerElementFactory {}

impl SingleHypothesisTrackerElementFactory {
    /// Create an initiator from configuration
    ///
    /// Expected configuration format:
    /// ```toml
    /// initiator = "threshold"  # component type
    /// # ... other configuration
    /// ```
    pub(crate) fn create_initiator(config: &Value) -> Result<Box<dyn Initiator>> {
        // Get the initiator type from config
        let initiator_type = config
            .get("initiator")
            .and_then(|v| v.as_str())
            .unwrap_or("threshold"); // Default to threshold

        match initiator_type {
            "threshold" => {
                let component_config =
                    <ThresholdInitiator as Configurable>::Config::from_toml_value(config)?;
                let initiator = ThresholdInitiator::from_config(component_config)?;
                Ok(initiator as Box<dyn Initiator>)
            }
            _ => Err(AstrocapError::PluginInvalidConfigError(format!(
                "Unknown initiator type: {}",
                initiator_type
            ))),
        }
    }

    /// Create a terminator from configuration
    ///
    /// Expected configuration format:
    /// ```toml
    /// terminator = "simple"  # component type
    /// # ... other configuration
    /// ```
    pub(crate) fn create_terminator(config: &Value) -> Result<Box<dyn Terminator>> {
        // Get the terminator type from config
        let terminator_type = config
            .get("terminator")
            .and_then(|v| v.as_str())
            .unwrap_or("simple"); // Default to simple

        match terminator_type {
            "simple" => {
                let component_config =
                    <SimpleTerminator as Configurable>::Config::from_toml_value(config)?;
                let terminator = SimpleTerminator::from_config(component_config)?;
                Ok(terminator as Box<dyn Terminator>)
            }
            _ => Err(AstrocapError::PluginInvalidConfigError(format!(
                "Unknown terminator type: {}",
                terminator_type
            ))),
        }
    }

    /// Create an associator from configuration
    ///
    /// Expected configuration format:
    /// ```toml
    /// associator = "nearest"  # component type
    /// # ... other configuration
    /// ```
    pub(crate) fn create_associator(config: &Value) -> Result<Box<dyn Associator>> {
        // Get the associator type from config
        let associator_type = config
            .get("associator")
            .and_then(|v| v.as_str())
            .unwrap_or("nearest"); // Default to nearest

        match associator_type {
            "nearest" => {
                let component_config =
                    <NearestAssociator as Configurable>::Config::from_toml_value(config)?;
                let associator = NearestAssociator::from_config(component_config)?;
                Ok(associator as Box<dyn Associator>)
            }
            _ => Err(AstrocapError::PluginInvalidConfigError(format!(
                "Unknown associator type: {}",
                associator_type
            ))),
        }
    }

    /// Create an updater from configuration
    ///
    /// Expected configuration format:
    /// ```toml
    /// updater = "kalman"  # component type
    /// # ... other configuration
    /// ```
    pub(crate) fn create_updater(config: &Value) -> Result<Box<dyn Updater>> {
        // Get the updater type from config
        let updater_type = config
            .get("updater")
            .and_then(|v| v.as_str())
            .unwrap_or("kalman"); // Default to kalman

        match updater_type {
            "kalman" => {
                let component_config =
                    <KalmanUpdater as Configurable>::Config::from_toml_value(config)?;
                let updater = KalmanUpdater::from_config(component_config)?;
                Ok(updater as Box<dyn Updater>)
            }
            _ => Err(AstrocapError::PluginInvalidConfigError(format!(
                "Unknown updater type: {}",
                updater_type
            ))),
        }
    }

    /// Create a predictor from configuration
    ///
    /// Expected configuration format:
    /// ```toml
    /// predictor = "kalman"  # component type
    /// # ... other configuration
    /// ```
    pub(crate) fn create_predictor(config: &Value) -> Result<Box<dyn Predictor>> {
        // Get the predictor type from config
        let predictor_type = config
            .get("predictor")
            .and_then(|v| v.as_str())
            .unwrap_or("kalman"); // Default to kalman

        match predictor_type {
            "kalman" => {
                let component_config =
                    <KalmanPredictor as Configurable>::Config::from_toml_value(config)?;
                let predictor = KalmanPredictor::from_config(component_config)?;
                Ok(predictor as Box<dyn Predictor>)
            }
            _ => Err(AstrocapError::PluginInvalidConfigError(format!(
                "Unknown predictor type: {}",
                predictor_type
            ))),
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

        // All components should be creatable with empty config (using defaults)
        assert!(SingleHypothesisTrackerElementFactory::create_initiator(&config).is_ok());
        assert!(SingleHypothesisTrackerElementFactory::create_terminator(&config).is_ok());
        assert!(SingleHypothesisTrackerElementFactory::create_associator(&config).is_ok());
        assert!(SingleHypothesisTrackerElementFactory::create_updater(&config).is_ok());
        assert!(SingleHypothesisTrackerElementFactory::create_predictor(&config).is_ok());
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

        // All components should be creatable with explicit config
        assert!(SingleHypothesisTrackerElementFactory::create_initiator(&config).is_ok());
        assert!(SingleHypothesisTrackerElementFactory::create_terminator(&config).is_ok());
        assert!(SingleHypothesisTrackerElementFactory::create_associator(&config).is_ok());
        assert!(SingleHypothesisTrackerElementFactory::create_updater(&config).is_ok());
        assert!(SingleHypothesisTrackerElementFactory::create_predictor(&config).is_ok());
    }

    #[test]
    fn test_invalid_component_type() {
        let config_toml = r#"
            initiator = "nonexistent"
        "#;

        let config: Value = toml::from_str(config_toml).unwrap();

        // Should fail with unknown type
        assert!(SingleHypothesisTrackerElementFactory::create_initiator(&config).is_err());
    }
}
