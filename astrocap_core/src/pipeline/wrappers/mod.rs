use crate::pipeline::timing::add_stage_timing;
use crate::pipeline::PipelineContext;
use crate::traits::{FrameProcessor, FrameSink, FrameSource};
use crate::{AstrocapError, FrameProcessorResult};
use std::time::Instant;

// Wrapper types that add stage_type functionality
pub(crate) struct FrameSourceWrapper {
    inner: Box<dyn FrameSource>,
    pub(crate) stage_type: &'static str,
}

pub(crate) struct FrameProcessorWrapper {
    inner: Box<dyn FrameProcessor>,
    pub(crate) stage_type: &'static str,
}

pub(crate) struct FrameSinkWrapper {
    inner: Box<dyn FrameSink>,
    pub(crate) stage_type: &'static str,
}

#[allow(unused)]
impl FrameSourceWrapper {
    pub fn new(inner: Box<dyn FrameSource>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn stage_type(&self) -> &'static str {
        self.stage_type
    }
}

#[allow(unused)]
impl FrameProcessorWrapper {
    pub fn new(inner: Box<dyn FrameProcessor>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn stage_type(&self) -> &'static str {
        self.stage_type
    }
}

#[allow(unused)]
impl FrameSinkWrapper {
    pub fn new(inner: Box<dyn FrameSink>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn stage_type(&self) -> &'static str {
        self.stage_type
    }
}

// Delegate trait methods to inner implementations with timing integration (in microseconds)
impl FrameSource for FrameSourceWrapper {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<crate::FrameContext> {
        let start_time = Instant::now();
        let mut result = self.inner.next_frame(ctx);
        let duration = start_time.elapsed();
        let duration_us = duration.as_micros() as u64;

        // Add timing to the frame context if we got a frame
        if let Some(ref mut frame_ctx) = result {
            add_stage_timing(frame_ctx, self.name(), self.stage_type, duration_us);

            // Log that we found existing GST timing data
            if let Some(timing_data) = frame_ctx
                .metadata
                .get("timing_data")
                .and_then(|data| data.downcast_ref::<Vec<(u64, String)>>())
            {
                let gst_event_count = timing_data
                    .iter()
                    .filter(|(_, name)| !name.starts_with("astrocap_"))
                    .count();
                if gst_event_count > 0 {
                    tracing::trace!(
                        gst_event_count,
                        "Frame source retrieved frame with GST timing data"
                    );
                }
            }
        }

        result
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        self.inner.pipeline_ctx_init(ctx)
    }
    fn pipeline_ctx_cleanup(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        self.inner.pipeline_ctx_cleanup(ctx)
    }
}

impl FrameProcessor for FrameProcessorWrapper {
    fn process(
        &mut self,
        frame_ctx: &mut crate::FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let start_time = Instant::now();
        let result = self.inner.process(frame_ctx, ctx);
        let duration = start_time.elapsed();
        let duration_us = duration.as_micros() as u64;

        add_stage_timing(frame_ctx, self.name(), self.stage_type, duration_us);

        result
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        self.inner.pipeline_ctx_init(ctx)
    }
    fn pipeline_ctx_cleanup(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        self.inner.pipeline_ctx_cleanup(ctx)
    }
}

impl FrameSink for FrameSinkWrapper {
    fn consume(&mut self, frame_ctx: &mut crate::FrameContext, ctx: &mut PipelineContext) {
        let start_time = Instant::now();
        self.inner.consume(frame_ctx, ctx);
        let duration = start_time.elapsed();
        let duration_us = duration.as_micros() as u64;

        add_stage_timing(frame_ctx, self.name(), self.stage_type, duration_us);
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        self.inner.pipeline_ctx_init(ctx)
    }
    fn pipeline_ctx_cleanup(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        self.inner.pipeline_ctx_cleanup(ctx)
    }
}
