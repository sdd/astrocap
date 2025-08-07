use thiserror::Error;

#[derive(Error, Debug)]
pub enum AstrocapError {
    #[error("Plugin error: {0}")]
    PluginError(String),
}
