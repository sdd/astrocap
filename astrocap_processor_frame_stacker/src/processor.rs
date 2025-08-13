use crate::map_colors::map_colors;
use astrocap_core::frame::CpuFrame;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::FrameProcessorResult::Skip;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use image::Luma;
use serde::Deserialize;
use std::cmp::max;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use toml::Value;

const DEFAULT_STACK_DEPTH: usize = 8;
const DEFAULT_RENORMALIZE: Renormalize = Renormalize::Sqrt;

#[derive(Debug, Deserialize)]
enum Renormalize {
    None,
    Full,
    Sqrt,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct FrameStackerConfig {
    stack_depth: usize,
    renormalize: Renormalize,
}

impl Default for FrameStackerConfig {
    fn default() -> Self {
        Self {
            stack_depth: DEFAULT_STACK_DEPTH,
            renormalize: DEFAULT_RENORMALIZE,
        }
    }
}

pub struct FrameStackerProcessor {
    config: FrameStackerConfig,
}

impl FrameStackerProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: FrameStackerConfig = config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!(
                "FrameStackerProcessor:: {}",
                e.to_string()
            ))
        })?;

        if config.stack_depth != 4 && config.stack_depth != 8 {
            return Err(AstrocapError::PluginInvalidConfigError(
                "stack_depth must be either 4 or 8 FrameStackerProcessor Config".to_string(),
            ));
        }

        Ok(Self { config })
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

            if frame_stack.len() > self.config.stack_depth {
                Some(frame_stack.pop_front().unwrap())
            } else {
                None
            }
        };

        let add_fn = match (self.config.stack_depth, &self.config.renormalize) {
            (4, Renormalize::Full) => {
                |p: Luma<u8>, q: Luma<u8>| Luma([(p[0]).saturating_add(q[0] >> 2)])
            }
            (8, Renormalize::Full) => {
                |p: Luma<u8>, q: Luma<u8>| Luma([(p[0]).saturating_add(q[0] >> 3)])
            }
            (4, Renormalize::None) => {
                |p: Luma<u8>, q: Luma<u8>| Luma([(p[0]).saturating_add(q[0])])
            }
            (8, Renormalize::None) => {
                |p: Luma<u8>, q: Luma<u8>| Luma([(p[0]).saturating_add(q[0])])
            }
            (4, Renormalize::Sqrt) => {
                |p: Luma<u8>, q: Luma<u8>| Luma([(p[0]).saturating_add(q[0] >> 1)])
            }
            (8, Renormalize::Sqrt) => |p: Luma<u8>, q: Luma<u8>| {
                let tmp: u16 = (q[0] as u16) >> 1;
                let scaled: u16 = (tmp * 181) >> 8;
                Luma([(p[0]).saturating_add(scaled as u8)])
            },
            _ => {
                tracing::error!("Unexpected combination of stack depth and renormalize");
                return Skip;
            }
        };

        let integration_frame = map_colors(&integration_frame.img, &frame.img, add_fn);

        let sub_fn = match (self.config.stack_depth, &self.config.renormalize) {
            (4, Renormalize::Full) => {
                |p: Luma<u8>, q: Luma<u8>| Luma([(p[0]).saturating_sub(q[0] >> 2)])
            }
            (8, Renormalize::Full) => {
                |p: Luma<u8>, q: Luma<u8>| Luma([(p[0]).saturating_sub(q[0] >> 3)])
            }
            (4, Renormalize::None) => {
                |p: Luma<u8>, q: Luma<u8>| Luma([(p[0]).saturating_sub(q[0])])
            }
            (8, Renormalize::None) => {
                |p: Luma<u8>, q: Luma<u8>| Luma([(p[0]).saturating_sub(q[0])])
            }
            (4, Renormalize::Sqrt) => {
                |p: Luma<u8>, q: Luma<u8>| Luma([(p[0]).saturating_sub(q[0] >> 1)])
            }
            (8, Renormalize::Sqrt) => |p: Luma<u8>, q: Luma<u8>| {
                let tmp: u16 = (q[0] as u16) >> 1;
                let scaled: u16 = (tmp * 181) >> 8;
                Luma([(p[0]).saturating_sub(scaled as u8)])
            },
            _ => {
                tracing::error!("Unexpected combination of stack depth and renormalize");
                return Skip;
            }
        };

        let img = match old_frame {
            None => integration_frame,
            Some(old_frame) => map_colors(&integration_frame, &old_frame.img, sub_fn),
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
