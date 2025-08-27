use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub(crate) struct Config {}

impl Default for Config {
    fn default() -> Self {
        Self {}
    }
}
