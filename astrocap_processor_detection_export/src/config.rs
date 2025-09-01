use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Output file path for exported detections
    pub output_path: PathBuf,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("detections_export.json"),
        }
    }
}
