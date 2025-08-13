use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub(crate) struct MedianConfig {
    pub(crate) r#async: bool,
    pub(crate) window_size: u32,
}

impl Default for MedianConfig {
    fn default() -> Self {
        Self {
            r#async: false,
            window_size: 30,
        }
    }
}
