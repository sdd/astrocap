use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Config {
    pub subtractand_key: String,
    #[serde(default)]
    pub gpu: bool,
}
