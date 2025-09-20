use crate::frame::CpuFrame;
use crate::pipeline::PipelineContext;
use crate::statistics::ProcessingType;
use crate::structs::{Detection, FittedPoint};
use crate::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use std::any::Any;
use std::sync::Arc;

/// Represents a type that can be dumped to parquet for offline analysis
pub trait Dumpable: Send + Sync + 'static {
    /// Flat row type suitable for Parquet export
    type Row: for<'a> serde::Deserialize<'a> + serde::Serialize + Send + Sync;

    /// Identifier fragment used for output filenames
    const TABLE_NAME: &'static str;

    /// Schema version (bump if layout of Row changes)
    const VERSION: u32;

    /// Convert into a row, always including run_id and frame_index
    fn to_row(&self, run_id: u64, frame_index: usize) -> Self::Row;
}

#[derive(Debug, Clone)]
pub struct TrackSummary {
    pub id: u64,
    pub x: f32,
    pub y: f32,
    pub amplitude: f32,
    pub age: usize,
    pub log_odds: f32,

    pub first_seen: usize,
    pub start_x: f32,
    pub start_y: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarCandidate {
    pub x: f32,
    pub y: f32,
    pub amp: f32,
}

pub trait TrackSummarize: Send + Sync + 'static {
    fn summarize(&self) -> TrackSummary;
}

pub trait PointDetector: Send + Sync + 'static {
    fn detect(
        &mut self,
        frame: &Frame,
        median: Option<Arc<Frame>>,
        mask: Option<&Frame>,
    ) -> Vec<Detection>;
}

pub trait PointFitter: Send + Sync + 'static {
    fn fit(&self, frame: &CpuFrame, point: &Detection) -> FittedPoint;
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

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
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
