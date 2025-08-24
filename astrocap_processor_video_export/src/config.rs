use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub(crate) struct Config {
    pub(crate) output_path: String,
    pub(crate) key: Option<String>,
    pub(crate) start_frame_index: Option<usize>,
    pub(crate) end_frame_index: Option<usize>,
    pub(crate) crop_left_px: usize,
    pub(crate) crop_right_px: usize,
    pub(crate) crop_top_px: usize,
    pub(crate) crop_bottom_px: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            output_path: "./cropped_test.mkv".to_string(),
            key: None,
            start_frame_index: None,
            end_frame_index: None,
            crop_left_px: 0,
            crop_right_px: 1920,
            crop_top_px: 0,
            crop_bottom_px: 1080,
        }
    }
}
