use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Input file path for recorded detections JSON
    pub input_path: PathBuf,
    /// Frame width for dummy frames (since we don't have actual video)
    pub frame_width: u32,
    /// Frame height for dummy frames
    pub frame_height: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            input_path: PathBuf::from("detections_export.json"),
            frame_width: 1920,
            frame_height: 1080,
        }
    }
}
