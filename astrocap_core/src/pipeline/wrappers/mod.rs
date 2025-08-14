use crate::pipeline::PipelineContext;
use crate::statistics::PipelineStatistics;
use crate::traits::{FrameProcessor, FrameSink, FrameSource};
use crate::{AstrocapError, FrameContext, FrameProcessorResult};
use std::sync::Arc;
use std::time::Instant;

pub struct FrameSourceWrapper {
    pub inner: Box<dyn FrameSource>,
    pub stage_type: &'static str,
}

impl FrameSourceWrapper {
    pub fn new(inner: Box<dyn FrameSource>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext> {
        // Sources handle their own timing to allow fine-grained control
        // over what work is measured vs coordination overhead
        self.inner.next_frame(ctx)
    }

    pub fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        self.inner.pipeline_ctx_init(ctx)
    }
}

pub struct FrameProcessorWrapper {
    pub inner: Box<dyn FrameProcessor>,
    pub stage_type: &'static str,
}

impl FrameProcessorWrapper {
    pub fn new(inner: Box<dyn FrameProcessor>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let start_time = Instant::now();
        let result = self.inner.process(frame_ctx, ctx);
        let duration_us = start_time.elapsed().as_micros() as u64;

        // Record timing if statistics are available
        if let Ok(stats) = ctx.get_as::<Arc<PipelineStatistics>>("pipeline_statistics") {
            stats.record_stage_timing(self.stage_type, duration_us);
        }

        result
    }

    pub fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        self.inner.pipeline_ctx_init(ctx)
    }
}

pub struct FrameSinkWrapper {
    pub inner: Box<dyn FrameSink>,
    pub stage_type: &'static str,
}

impl FrameSinkWrapper {
    pub fn new(inner: Box<dyn FrameSink>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext) {
        let start_time = Instant::now();
        self.inner.consume(frame_ctx, ctx);
        let duration_us = start_time.elapsed().as_micros() as u64;

        // Record timing if statistics are available
        if let Ok(stats) = ctx.get_as::<Arc<PipelineStatistics>>("pipeline_statistics") {
            stats.record_stage_timing(self.stage_type, duration_us);
        }
    }

    pub fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        self.inner.pipeline_ctx_init(ctx)
    }
}
