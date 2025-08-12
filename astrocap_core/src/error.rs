use thiserror::Error;

#[derive(Error, Debug)]
pub enum AstrocapError {
    #[error("Plugin error: {0}")]
    GeneralPluginError(String),

    #[error("Plugin config missing")]
    PluginMissingConfigError,

    #[error("Plugin config invalid: {0}")]
    PluginInvalidConfigError(String),

    #[error("Frame metadata not found")]
    FrameMetadataNotFoundError,

    #[error("Frame metadata downcast type error")]
    FrameMetadataTypeError,
}
