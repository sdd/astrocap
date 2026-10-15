use astrocap_core::pipeline_services::IntegratedFrame;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::FrameProcessorResult::Skip;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use serde::Deserialize;
use std::sync::{Arc, Mutex};
use toml::Value;
use vyd::frame::pool::WaitPolicy;
use vyd::frame::CpuFrame;
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

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

#[derive(Clone)]
struct CircularFrameBuffer {
    frames: Vec<Option<CpuFrame>>,
    write_index: usize,
    count: usize,
    capacity: usize,
}

impl CircularFrameBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            frames: vec![None; capacity],
            write_index: 0,
            count: 0,
            capacity,
        }
    }

    fn add_frame(&mut self, frame: CpuFrame) -> Option<CpuFrame> {
        let old_frame = self.frames[self.write_index].take();
        self.frames[self.write_index] = Some(frame);

        self.write_index = (self.write_index + 1) % self.capacity;

        if self.count < self.capacity {
            self.count += 1;
            None // No frame to subtract during fill phase
        } else {
            old_frame // Return the frame that was just replaced
        }
    }
}

pub struct FrameStackerProcessor {
    config: FrameStackerConfig,
    frame_stack: Arc<Mutex<CircularFrameBuffer>>,
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

        if config.stack_depth == 0 {
            return Err(AstrocapError::PluginInvalidConfigError(
                "frame stack depth must be positive".into(),
            ));
        }
        if matches!(config.renormalize, Renormalize::Sqrt)
            && !matches!(config.stack_depth, 4 | 8 | 16 | 64)
        {
            return Err(AstrocapError::PluginInvalidConfigError(
                "sqrt renormalization supports stack depth 4, 8, 16, or 64".into(),
            ));
        }
        if matches!(config.renormalize, Renormalize::Full)
            && !matches!(config.stack_depth, 4 | 8 | 16 | 32 | 64 | 128)
        {
            return Err(AstrocapError::PluginInvalidConfigError(
                "full renormalization supports stack depth 4, 8, 16, 32, 64, or 128".into(),
            ));
        }

        Ok(Self {
            frame_stack: Arc::new(Mutex::new(CircularFrameBuffer::new(config.stack_depth))),
            config,
        })
    }
}

impl FrameProcessor for FrameStackerProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        match self.try_process(frame_ctx, ctx) {
            Ok(result) => result,
            Err(error) => {
                tracing::error!(%error, "frame stacking failed");
                Skip
            }
        }
    }

    fn try_process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, AstrocapError> {
        let start = std::time::Instant::now();

        let frame = match &frame_ctx.frame {
            Frame::Cpu(cpu) => cpu.clone(),
            _ => {
                return Err(AstrocapError::GeneralPluginError(
                    "frame stacker requires packed CPU Luma8".into(),
                ));
            }
        };

        let frame = frame.to_shared();

        let frame_stack = self.frame_stack.clone();

        let integration_frame = match ctx
            .services()
            .latest::<IntegratedFrame>()
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
        {
            Some(previous) => previous.0.clone(),
            None => ctx
                .frame_factory()
                .create_luma8(frame.width(), frame.height(), WaitPolicy::Immediate)
                .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?
                .to_cpu_frame()
                .ok_or_else(|| {
                    AstrocapError::GeneralPluginError("initial integration frame is not CPU".into())
                })?,
        };
        if integration_frame.dimensions() != frame.dimensions() {
            return Err(AstrocapError::GeneralPluginError(format!(
                "frame stacker dimensions changed from {:?} to {:?}",
                integration_frame.dimensions(),
                frame.dimensions()
            )));
        }

        // Add frame to circular buffer and get the frame to subtract (if any)
        let old_frame = {
            let frame_stack = frame_stack
                .lock()
                .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?;
            if frame_stack.count == frame_stack.capacity {
                frame_stack.frames[frame_stack.write_index].clone()
            } else {
                None
            }
        };

        let elapsed = start.elapsed();
        tracing::debug!("Frame Stack setup took {:?}", elapsed);
        let start = std::time::Instant::now();

        // Get raw slices for iterator-based processing
        let integration_pixels = integration_frame
            .try_read_luma8(WaitPolicy::Immediate)
            .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?;
        let new_pixels = frame
            .try_read_luma8(WaitPolicy::Immediate)
            .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?;
        let integration_frame = ctx
            .frame_factory()
            .create_luma8_with(
                frame.width(),
                frame.height(),
                WaitPolicy::Immediate,
                |result_data| {
                    // Simplified processing without complex scaling for debugging
                    match (&self.config.renormalize, old_frame.as_ref()) {
                        (Renormalize::None, Some(old_frame)) => {
                            let old_pixels = old_frame.try_read_luma8(WaitPolicy::Immediate)?;
                            result_data
                                .iter_mut()
                                .zip(integration_pixels.iter())
                                .zip(new_pixels.iter())
                                .zip(old_pixels.iter())
                                .for_each(|(((result, &integration), &new_pixel), &old_pixel)| {
                                    let new_val_i16: i16 =
                                        integration as i16 + new_pixel as i16 - old_pixel as i16;
                                    let clamped_val = new_val_i16.clamp(0, 255) as u8;
                                    *result = clamped_val;

                                    // *result = integration
                                    //     .saturating_add(new_pixel)
                                    //     .saturating_sub(old_pixel);
                                });
                        }
                        (Renormalize::None, None) => {
                            result_data
                                .iter_mut()
                                .zip(integration_pixels.iter())
                                .zip(new_pixels.iter())
                                .for_each(|((result, &integration), &new_pixel)| {
                                    *result = integration.saturating_add(new_pixel);
                                });
                        }
                        (Renormalize::Full, Some(old_frame)) => {
                            let old_pixels = old_frame.try_read_luma8(WaitPolicy::Immediate)?;
                            let shift = match self.config.stack_depth {
                                4 => 2u32,
                                8 => 3u32,
                                16 => 4u32,
                                32 => 5u32,
                                64 => 6u32,
                                128 => 7u32,
                                _ => 0u32,
                            };
                            result_data
                                .iter_mut()
                                .zip(integration_pixels.iter())
                                .zip(new_pixels.iter())
                                .zip(old_pixels.iter())
                                .for_each(|(((result, &integration), &new_pixel), &old_pixel)| {
                                    let after_add =
                                        integration.saturating_add(new_pixel.wrapping_shr(shift));
                                    *result =
                                        after_add.saturating_sub(old_pixel.wrapping_shr(shift));
                                });
                        }
                        (Renormalize::Full, None) => {
                            let shift = match self.config.stack_depth {
                                4 => 2u32,
                                8 => 3u32,
                                16 => 4u32,
                                32 => 5u32,
                                64 => 6u32,
                                128 => 7u32,
                                _ => 0u32,
                            };
                            result_data
                                .iter_mut()
                                .zip(integration_pixels.iter())
                                .zip(new_pixels.iter())
                                .for_each(|((result, &integration), &new_pixel)| {
                                    *result =
                                        integration.saturating_add(new_pixel.wrapping_shr(shift));
                                });
                        }
                        (Renormalize::Sqrt, Some(old_frame)) => {
                            let old_pixels = old_frame.try_read_luma8(WaitPolicy::Immediate)?;
                            if self.config.stack_depth == 8 {
                                // Special case for sqrt(8): multiply by 181, divide by 512
                                result_data
                                    .iter_mut()
                                    .zip(integration_pixels.iter())
                                    .zip(new_pixels.iter())
                                    .zip(old_pixels.iter())
                                    .for_each(
                                        |(((result, &integration), &new_pixel), &old_pixel)| {
                                            let new_scaled = ((new_pixel as u16 * 181) >> 9) as u8; // 512 = 2^9
                                            let old_scaled = ((old_pixel as u16 * 181) >> 9) as u8;
                                            *result = integration
                                                .saturating_add(new_scaled)
                                                .saturating_sub(old_scaled);
                                        },
                                    );
                            } else if self.config.stack_depth == 4 {
                                // stack_depth == 4, just halve
                                result_data
                                    .iter_mut()
                                    .zip(integration_pixels.iter())
                                    .zip(new_pixels.iter())
                                    .zip(old_pixels.iter())
                                    .for_each(
                                        |(((result, &integration), &new_pixel), &old_pixel)| {
                                            *result = integration
                                                .saturating_add(new_pixel >> 1)
                                                .saturating_sub(old_pixel >> 1);
                                        },
                                    );
                            } else if self.config.stack_depth == 16 {
                                // stack_depth == 16, >>2
                                result_data
                                    .iter_mut()
                                    .zip(integration_pixels.iter())
                                    .zip(new_pixels.iter())
                                    .zip(old_pixels.iter())
                                    .for_each(
                                        |(((result, &integration), &new_pixel), &old_pixel)| {
                                            *result = integration
                                                .saturating_add(new_pixel >> 2)
                                                .saturating_sub(old_pixel >> 2);
                                        },
                                    );
                            } else if self.config.stack_depth == 64 {
                                // stack_depth == 64, >> 3
                                result_data
                                    .iter_mut()
                                    .zip(integration_pixels.iter())
                                    .zip(new_pixels.iter())
                                    .zip(old_pixels.iter())
                                    .for_each(
                                        |(((result, &integration), &new_pixel), &old_pixel)| {
                                            *result = integration
                                                .saturating_add(new_pixel >> 3)
                                                .saturating_sub(old_pixel >> 3);
                                        },
                                    );
                            } else {
                                // 32 or 128? todo
                                unreachable!("sqrt stack depth checked in constructor");
                            }
                        }
                        (Renormalize::Sqrt, None) => {
                            if self.config.stack_depth == 8 {
                                result_data
                                    .iter_mut()
                                    .zip(integration_pixels.iter())
                                    .zip(new_pixels.iter())
                                    .for_each(|((result, &integration), &new_pixel)| {
                                        let new_scaled = ((new_pixel as u16 * 181) >> 9) as u8;
                                        *result = integration.saturating_add(new_scaled);
                                    });
                            } else if self.config.stack_depth == 4 {
                                result_data
                                    .iter_mut()
                                    .zip(integration_pixels.iter())
                                    .zip(new_pixels.iter())
                                    .for_each(|((result, &integration), &new_pixel)| {
                                        *result = integration.saturating_add(new_pixel >> 1);
                                    });
                            } else if self.config.stack_depth == 16 {
                                result_data
                                    .iter_mut()
                                    .zip(integration_pixels.iter())
                                    .zip(new_pixels.iter())
                                    .for_each(|((result, &integration), &new_pixel)| {
                                        *result = integration.saturating_add(new_pixel >> 2);
                                    });
                            } else if self.config.stack_depth == 64 {
                                result_data
                                    .iter_mut()
                                    .zip(integration_pixels.iter())
                                    .zip(new_pixels.iter())
                                    .for_each(|((result, &integration), &new_pixel)| {
                                        *result = integration.saturating_add(new_pixel >> 3);
                                    });
                            } else {
                                // 32 or 218? TODO
                                unreachable!("sqrt stack depth checked in constructor");
                            }
                        }
                    }

                    let elapsed = start.elapsed();
                    tracing::debug!("Frame Stacking took {:?}", elapsed);

                    Ok(())
                },
            )
            .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?;

        frame_stack
            .lock()
            .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?
            .add_frame(frame);

        let context_frame = integration_frame
            .to_cpu(None)
            .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?;
        ctx.services()
            .publish(Arc::new(IntegratedFrame(context_frame)))
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        frame_ctx.frame = integration_frame;

        let elapsed = start.elapsed();
        tracing::debug!("Frame Stacking post-processing took {:?}", elapsed);

        Ok(FrameProcessorResult::Continue)
    }

    fn name(&self) -> &str {
        "frame_stacker"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyd::frame::factory::FrameFactory;
    use vyd::frame::pool::{BufferPoolManager, PoolLimits};

    fn stacker(depth: usize, renormalize: &str) -> FrameStackerProcessor {
        let config: Value = toml::from_str(&format!(
            "stack_depth = {depth}\nrenormalize = '{renormalize}'"
        ))
        .unwrap();
        FrameStackerProcessor::new(Some(&config)).unwrap()
    }

    #[test]
    fn stacks_into_pooled_frames_and_retains_input_snapshots() {
        let mut processor = stacker(1, "None");
        let mut ctx = PipelineContext::new();
        let mut first = FrameContext::new(Frame::from_raw(2, 1, vec![10, 20]).unwrap(), 0);
        assert!(matches!(
            processor.try_process(&mut first, &mut ctx),
            Ok(FrameProcessorResult::Continue)
        ));
        assert_eq!(
            first
                .frame
                .try_read_cpu_luma8(WaitPolicy::Immediate)
                .unwrap(),
            &[10, 20]
        );
        assert_eq!(
            ctx.frame_factory()
                .cpu_pool()
                .snapshot()
                .unwrap()
                .active
                .len(),
            1
        );

        let mut second = FrameContext::new(Frame::from_raw(2, 1, vec![5, 7]).unwrap(), 1);
        assert!(matches!(
            processor.try_process(&mut second, &mut ctx),
            Ok(FrameProcessorResult::Continue)
        ));
        assert_eq!(
            second
                .frame
                .try_read_cpu_luma8(WaitPolicy::Immediate)
                .unwrap(),
            &[5, 7]
        );
        assert_eq!(
            first
                .frame
                .try_read_cpu_luma8(WaitPolicy::Immediate)
                .unwrap(),
            &[10, 20]
        );
    }

    #[test]
    fn pool_exhaustion_keeps_input_and_ring_unchanged() {
        let mut processor = stacker(1, "None");
        let mut ctx = PipelineContext::new();
        ctx.set_frame_factory(FrameFactory::new(BufferPoolManager::new(
            PoolLimits::new(2, 8).unwrap(),
        )));
        let mut first = FrameContext::new(Frame::from_raw(2, 1, vec![10, 20]).unwrap(), 0);
        processor.try_process(&mut first, &mut ctx).unwrap();
        let _spare = ctx
            .frame_factory()
            .cpu_pool()
            .try_acquire(2, WaitPolicy::Immediate)
            .unwrap();
        let mut second = FrameContext::new(Frame::from_raw(2, 1, vec![5, 7]).unwrap(), 1);
        assert!(processor.try_process(&mut second, &mut ctx).is_err());
        assert_eq!(
            second
                .frame
                .try_read_cpu_luma8(WaitPolicy::Immediate)
                .unwrap(),
            &[5, 7]
        );
        let ring = processor.frame_stack.lock().unwrap();
        assert_eq!(ring.write_index, 0);
        assert_eq!(
            ring.frames[0]
                .as_ref()
                .unwrap()
                .try_read_luma8(WaitPolicy::Immediate)
                .unwrap(),
            &[10, 20]
        );
    }

    #[test]
    fn rejects_unsupported_depth_before_processing() {
        let config: Value = toml::from_str("stack_depth = 32\nrenormalize = 'Sqrt'").unwrap();
        assert!(FrameStackerProcessor::new(Some(&config)).is_err());
    }
}
