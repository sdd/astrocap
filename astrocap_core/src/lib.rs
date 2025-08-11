use crate::pipeline::PipelineContext;
use std::any::Any;
use std::collections::HashMap;

pub mod error;
pub mod frame;
pub mod pipeline;
pub mod stages;
pub mod structs;

pub use error::AstrocapError;
pub use frame::Frame;

// needed for the exported macros to work without the consuming crate having to import them
use crate::structs::DetectedPoint;
pub use inventory;
pub use paste;

pub struct FrameContext {
    pub metadata: HashMap<String, Box<dyn Any + Send + Sync>>,
    pub frame: Frame,
}

impl FrameContext {
    pub fn new(frame: Frame) -> Self {
        Self {
            metadata: HashMap::new(),
            frame,
        }
    }

    pub fn get_as<'a, T: 'static>(&'a self, key: &str) -> Result<&'a T, AstrocapError> {
        let Some(value) = self.metadata.get(key) else {
            tracing::error!("key \"{}\" not present in frame context metadata", key);
            return Err(AstrocapError::FrameMetadataNotFoundError);
        };

        let Some(downcasted) = value.downcast_ref::<T>() else {
            tracing::error!("Could not downcast metadata value to requested type");
            return Err(AstrocapError::FrameMetadataTypeError);
        };

        Ok(downcasted)
    }

    pub fn put<'a, T: Any + Send + Sync + 'static>(&'a mut self, key: &str, value: T) {
        self.metadata.insert(key.to_string(), Box::new(value));
    }
}

pub enum FrameProcessorResult {
    Continue,
    Skip,
}

pub trait FrameSource: Send + Sync {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext>;
    fn name(&self) -> &str;
}

pub trait FrameProcessor: Send + Sync {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult;
    fn name(&self) -> &str;
}

pub trait FrameSink: Send + Sync {
    fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext);
    fn name(&self) -> &str;
}

pub trait StageFactory: Send + Sync {
    fn create(&self, params: Option<&toml::Value>) -> Result<Box<dyn Any>, AstrocapError>;
    fn stage_type(&self) -> &'static str;
}
