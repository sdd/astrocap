use astrocap_core::AstrocapError;
use serde::Deserialize;
use std::path::PathBuf;
use toml::Value;

#[derive(Debug, Deserialize)]
pub struct Config {
    /// Path to the JSON annotations file
    pub annotations_file: PathBuf,

    pub confirmation_threshold: f32,

    /// Distance threshold for considering a detection a match (pixels)
    #[serde(default = "default_distance_threshold")]
    pub distance_threshold: f32,

    /// Output path for evaluation results JSON
    #[serde(default = "default_output_path")]
    pub output_path: PathBuf,
}

fn default_distance_threshold() -> f32 {
    2.45
}

fn default_output_path() -> PathBuf {
    PathBuf::from("evaluation_results.json")
}

impl TryFrom<Value> for Config {
    type Error = AstrocapError;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        let config: Config = value.try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!(
                "Failed to deserialize PointDetectorEvaluator config: {}",
                e
            ))
        })?;

        // Validate that annotations file exists
        if !config.annotations_file.exists() {
            return Err(AstrocapError::PluginInvalidConfigError(format!(
                "Annotations file does not exist: {:?}",
                config.annotations_file
            )));
        }

        // Validate distance threshold
        if config.distance_threshold <= 0.0 {
            return Err(AstrocapError::PluginInvalidConfigError(
                "Distance threshold must be positive".to_string(),
            ));
        }

        Ok(config)
    }
}
