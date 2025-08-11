use crate::pipeline::PipelineContext;
use crate::traits::FrameProcessor;
use crate::{register_astrocap_frame_processor, AstrocapError, FrameContext, FrameProcessorResult};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;
use toml::Value;

pub struct DummyProcessor {
    processing_delay_ms: Option<u64>,
}

impl DummyProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let processing_delay_ms = config
            .and_then(|c| c.get("processing_delay_ms"))
            .and_then(|d| d.as_integer())
            .map(|d| d as u64);

        Ok(Self {
            processing_delay_ms,
        })
    }
}

impl FrameProcessor for DummyProcessor {
    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        ctx.entry("frames_processed".to_string())
            .or_insert_with(|| Box::new(AtomicUsize::new(0)));

        Ok(())
    }

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        tracing::trace!(
            "Processor received frame with metadata keys: {:?}",
            frame_ctx.metadata.keys().collect::<Vec<_>>()
        );

        // Apply processing delay if configured
        if let Some(delay_ms) = self.processing_delay_ms {
            tracing::trace!("Applying processing delay of {} ms", delay_ms);
            std::thread::sleep(Duration::from_millis(delay_ms));
        }

        // Track frames processed in pipeline context
        let counter = ctx
            .entry("frames_processed".to_string())
            .or_insert_with(|| Box::new(AtomicUsize::new(0)));

        if let Some(atomic_counter) = counter.downcast_ref::<AtomicUsize>() {
            atomic_counter.fetch_add(1, Ordering::SeqCst);
        }

        FrameProcessorResult::Skip
    }

    fn name(&self) -> &str {
        "dummy_processor"
    }
}

register_astrocap_frame_processor!(DummyProcessor);
