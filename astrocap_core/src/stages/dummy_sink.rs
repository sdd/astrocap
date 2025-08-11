use crate::pipeline::PipelineContext;
use crate::traits::FrameSink;
use crate::{register_astrocap_frame_sink, AstrocapError, FrameContext};
use std::sync::atomic::{AtomicUsize, Ordering};
use toml::Value;

pub struct DummySink;

impl DummySink {
    pub fn new(_config: Option<&Value>) -> Result<Self, AstrocapError> {
        Ok(Self)
    }
}

impl FrameSink for DummySink {
    fn consume(&mut self, frame_ctx: &mut FrameContext, ctx: &mut PipelineContext) {
        tracing::debug!(
            "Sink received frame with metadata keys: {:?}",
            frame_ctx.metadata.keys().collect::<Vec<_>>()
        );

        // Track frames sunk in pipeline context
        let counter = ctx
            .entry("frames_sunk".to_string())
            .or_insert_with(|| Box::new(AtomicUsize::new(0)));

        if let Some(atomic_counter) = counter.downcast_ref::<AtomicUsize>() {
            atomic_counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn name(&self) -> &str {
        "dummy_sink"
    }
}

register_astrocap_frame_sink!(DummySink);
