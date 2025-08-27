use crate::frame::CpuFrame;
use crate::pipeline::PipelineContext;
use crate::statistics::ProcessingType;
use crate::structs::{DetectedPoint, FittedPoint};
use crate::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use std::any::Any;
use std::sync::Arc;

pub trait PointDetector: Send + Sync + 'static {
    fn detect(
        &self,
        frame: &Frame,
        median: Option<Arc<Frame>>,
        mask: Option<&Frame>,
    ) -> Vec<DetectedPoint>;
}

pub trait PointFitter: Send + Sync + 'static {
    fn fit(&self, frame: &CpuFrame, point: &DetectedPoint) -> FittedPoint;
}

pub trait FrameSource: Send + Sync {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext>;
    fn pipeline_ctx_init(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        Ok(())
    }
    fn name(&self) -> &str;
    fn processing_type(&self) -> ProcessingType;
}

pub trait FrameProcessor: Send + Sync {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult;
    fn pipeline_ctx_init(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        Ok(())
    }
    fn name(&self) -> &str;
    fn processing_type(&self) -> ProcessingType;
}

pub trait FrameSink: Send + Sync {
    fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext);
    fn pipeline_ctx_init(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        Ok(())
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        Ok(())
    }

    fn name(&self) -> &str;
    fn processing_type(&self) -> ProcessingType;
}

pub trait StageFactory: Send + Sync {
    fn create(&self, params: Option<&toml::Value>) -> Result<Box<dyn Any>, AstrocapError>;
    fn stage_type(&self) -> &'static str;
}
