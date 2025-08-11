use astrocap_core::frame::CpuFrame;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::FrameProcessorResult::Skip;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use image::Luma;
use imageproc::definitions::Image;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use toml::Value;

use crate::map_colors::map_colors;

pub struct FrameStackerProcessor {
    stack_depth: usize,
}

impl FrameStackerProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let stack_depth = config
            .and_then(|c| c.get("stack_depth"))
            .and_then(|d| d.as_integer())
            .map(|d| d as usize);

        let Some(stack_depth) = stack_depth else {
            return Err(AstrocapError::GeneralPluginError(
                "stack_depth not present in FrameStackerProcessor Config".to_string(),
            ));
        };

        Ok(Self { stack_depth })
    }
}

impl FrameProcessor for FrameStackerProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let Ok(frame) = frame_ctx.frame.get_image(None) else {
            tracing::warn!("No frame to process");
            return Skip;
        };

        let Ok(frame) = frame_ctx.take_frame().to_cpu(None) else {
            tracing::warn!("could not get CPU Frame");
            return Skip;
        };

        let frame = frame.to_shared();

        let frame_stack = ctx
            .entry("frame_stacker/frame_stack")
            .or_insert_with(|| Box::new(Arc::new(Mutex::new(VecDeque::<CpuFrame>::new()))))
            .downcast_ref::<Arc<Mutex<VecDeque<CpuFrame>>>>()
            .unwrap()
            .clone();

        let integration_frame = ctx
            .entry("video/integrated")
            .or_insert_with(|| {
                Box::new(CpuFrame::new_shared(
                    frame.width(),
                    frame.height(),
                    Arc::from(vec![0u8; (frame.width() * frame.height()) as usize]),
                ))
            })
            .downcast_ref::<CpuFrame>()
            .unwrap()
            .clone();

        // add frame to stack and remove oldest if stack is full
        let old_frame = {
            let mut frame_stack = frame_stack.lock().unwrap();
            frame_stack.push_back(frame.clone());

            if frame_stack.len() > self.stack_depth {
                Some(frame_stack.pop_front().unwrap())
            } else {
                None
            }
        };

        let integration_frame = map_colors(&integration_frame.img, &frame.img, |p, q| {
            Luma([(p[0]).saturating_add(q[0])])
        });

        let img = match old_frame {
            None => integration_frame,
            Some(old_frame) => map_colors(&integration_frame, &old_frame.img, |p, q| {
                Luma([(p[0]).saturating_sub(q[0])])
            }),
        };

        let cpu_storage: Arc<[u8]> = Arc::from(img.into_raw());
        let integration_frame = CpuFrame::new_shared(frame.width(), frame.height(), cpu_storage);

        ctx.put("video/integrated", integration_frame.clone());

        frame_ctx.frame = Frame::Cpu(integration_frame);

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "frame_stacker"
    }
}
