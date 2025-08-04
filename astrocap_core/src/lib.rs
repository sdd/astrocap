use crate::pipeline::PipelineContext;
use image::{ImageBuffer, Luma};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

pub mod error;
pub mod pipeline;
pub mod stages;

pub use error::Error;

// needed for the exported macros to work without the consuming crate having to import them
pub use inventory;
pub use paste;

#[non_exhaustive]
pub enum Frame {
    ImgBufArc(ImageBuffer<Luma<u8>, Arc<[u8]>>),
    ImgBuf(ImageBuffer<Luma<u8>, Vec<u8>>),
    // May also have GPU buffer in here at some point
}

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
}

pub trait FrameSource: Send + Sync {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext>;
    fn name(&self) -> &str;
}

pub trait FrameProcessor: Send + Sync {
    fn process(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext) -> bool;
    fn name(&self) -> &str;
}

pub trait FrameSink: Send + Sync {
    fn consume(&mut self, frame_ctx: &FrameContext, ctx: &mut PipelineContext);
    fn name(&self) -> &str;
}

pub trait StageFactory: Send + Sync {
    fn create(&self, params: Option<&toml::Value>) -> Result<Box<dyn Any>, Error>;
    fn stage_type(&self) -> &'static str;
}
