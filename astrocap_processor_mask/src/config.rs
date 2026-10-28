use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    pub file_path: String,
    #[serde(default)]
    pub gpu: bool,
}
