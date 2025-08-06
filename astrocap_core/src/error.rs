use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Plugin error: {0}")]
    PluginError(String),
}
