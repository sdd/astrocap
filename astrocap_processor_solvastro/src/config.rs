use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub(crate) struct SolvastroProcessorConfig {}

impl Default for SolvastroProcessorConfig {
    fn default() -> Self {
        Self {}
    }
}
