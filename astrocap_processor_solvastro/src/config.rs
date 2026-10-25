use std::path::PathBuf;

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct SolvastroProcessorConfig {
    pub(crate) index_path: PathBuf,
}
