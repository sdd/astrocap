use crate::frame::CpuFrame;
use crate::pipeline::PipelineContext;
use crate::{register_astrocap_frame_source, AstrocapError, Frame, FrameContext, FrameSource};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct DummySource {
    frame_count: usize,
}

impl DummySource {
    pub fn new(_params: Option<&toml::Value>) -> Result<Self, AstrocapError> {
        Ok(Self { frame_count: 5 })
    }
}

impl FrameSource for DummySource {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext> {
        if self.frame_count == 0 {
            None
        } else {
            self.frame_count -= 1;

            // Track frames generated in pipeline context
            let counter = ctx
                .entry("frames_sourced".to_string())
                .or_insert_with(|| Box::new(AtomicUsize::new(0)));

            if let Some(atomic_counter) = counter.downcast_ref::<AtomicUsize>() {
                atomic_counter.fetch_add(1, Ordering::SeqCst);
            }

            Some(FrameContext::new(Frame::Cpu(CpuFrame::new_owned(10, 10))))
        }
    }

    fn name(&self) -> &str {
        "dummy_source"
    }
}

register_astrocap_frame_source!(DummySource);
