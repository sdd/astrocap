use astrocap_core::frame::CpuFrame;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::FrameProcessorResult::Skip;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use serde::Deserialize;
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

        // if config.stack_depth != 4 && config.stack_depth != 8 {
        //     return Err(AstrocapError::PluginInvalidConfigError(
        //         "stack_depth must be either 4 or 8 FrameStackerProcessor Config".to_string(),
        //     ));
        // }

        Ok(Self { config })
    }
}

impl FrameProcessor for FrameStackerProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let start = std::time::Instant::now();

        let Ok(frame) = frame_ctx.take_frame().to_cpu(None) else {
            tracing::warn!("could not get CPU Frame");
            return Skip;
        };

        let frame = frame.to_shared();

        let frame_stack = ctx
            .get_as_or_insert::<Arc<Mutex<CircularFrameBuffer>>, _>(
                "frame_stacker/frame_stack",
                || {
                    Arc::new(Mutex::new(CircularFrameBuffer::new(
                        self.config.stack_depth,
                    )))
                },
            )
            .clone();

        let integration_frame = ctx
            .get_as_or_insert::<CpuFrame, _>("video/integrated", || {
                CpuFrame::from_shared(
                    frame.width(),
                    frame.height(),
                    Arc::from(vec![0u8; (frame.width() * frame.height()) as usize]),
                )
                .expect("Could not create frame to integrate into")
            })
            .clone();

        // Add frame to circular buffer and get the frame to subtract (if any)
        let old_frame = {
            let mut frame_stack = frame_stack.lock().unwrap();
            frame_stack.add_frame(frame.clone())
        };

        let elapsed = start.elapsed();
        tracing::debug!("Frame Stack setup took {:?}", elapsed);
        let start = std::time::Instant::now();

        // Get raw slices for iterator-based processing
        let integration_pixels = integration_frame.img.as_raw();
        let new_pixels = frame.img.as_raw();
        let pixel_count = integration_pixels.len();

        // Pre-allocate Vec with uninitialized memory - we'll write to every location
        let mut result_data = Vec::with_capacity(pixel_count);
        unsafe {
            #[allow(clippy::uninit_vec)]
            result_data.set_len(pixel_count);
        }

        // Simplified processing without complex scaling for debugging
        match (&self.config.renormalize, old_frame.as_ref()) {
            (Renormalize::None, Some(old_frame)) => {
                let old_pixels = old_frame.img.as_raw();
                result_data
                    .iter_mut()
                    .zip(integration_pixels.iter())
                    .zip(new_pixels.iter())
                    .zip(old_pixels.iter())
                    .for_each(|(((result, &integration), &new_pixel), &old_pixel)| {
                        *result = integration
                            .saturating_add(new_pixel)
                            .saturating_sub(old_pixel);
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
                let old_pixels = old_frame.img.as_raw();
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
                        let after_add = integration.saturating_add(new_pixel.wrapping_shr(shift));
                        *result = after_add.saturating_sub(old_pixel.wrapping_shr(shift));
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
                        *result = integration.saturating_add(new_pixel.wrapping_shr(shift));
                    });
            }
            (Renormalize::Sqrt, Some(old_frame)) => {
                let old_pixels = old_frame.img.as_raw();
                if self.config.stack_depth == 8 {
                    // Special case for sqrt(8): multiply by 181, divide by 512
                    result_data
                        .iter_mut()
                        .zip(integration_pixels.iter())
                        .zip(new_pixels.iter())
                        .zip(old_pixels.iter())
                        .for_each(|(((result, &integration), &new_pixel), &old_pixel)| {
                            let new_scaled = ((new_pixel as u16 * 181) >> 9) as u8; // 512 = 2^9
                            let old_scaled = ((old_pixel as u16 * 181) >> 9) as u8;
                            *result = integration
                                .saturating_add(new_scaled)
                                .saturating_sub(old_scaled);
                        });
                } else if self.config.stack_depth == 4 {
                    // stack_depth == 4, just halve
                    result_data
                        .iter_mut()
                        .zip(integration_pixels.iter())
                        .zip(new_pixels.iter())
                        .zip(old_pixels.iter())
                        .for_each(|(((result, &integration), &new_pixel), &old_pixel)| {
                            *result = integration
                                .saturating_add(new_pixel >> 1)
                                .saturating_sub(old_pixel >> 1);
                        });
                } else if self.config.stack_depth == 16 {
                    // stack_depth == 16, >>2
                    result_data
                        .iter_mut()
                        .zip(integration_pixels.iter())
                        .zip(new_pixels.iter())
                        .zip(old_pixels.iter())
                        .for_each(|(((result, &integration), &new_pixel), &old_pixel)| {
                            *result = integration
                                .saturating_add(new_pixel >> 2)
                                .saturating_sub(old_pixel >> 2);
                        });
                } else if self.config.stack_depth == 64 {
                    // stack_depth == 64, >> 3
                    result_data
                        .iter_mut()
                        .zip(integration_pixels.iter())
                        .zip(new_pixels.iter())
                        .zip(old_pixels.iter())
                        .for_each(|(((result, &integration), &new_pixel), &old_pixel)| {
                            *result = integration
                                .saturating_add(new_pixel >> 3)
                                .saturating_sub(old_pixel >> 3);
                        });
                } else {
                    // 32 or 128? todo
                    unimplemented!();
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
                    unimplemented!();
                }
            }
        }

        let elapsed = start.elapsed();
        tracing::debug!("Frame Stacking took {:?}", elapsed);
        let start = std::time::Instant::now();

        let cpu_storage: Arc<[u8]> = Arc::from(result_data);
        let integration_frame =
            CpuFrame::from_shared(frame.width(), frame.height(), cpu_storage).unwrap();

        ctx.put("video/integrated", integration_frame.clone());
        frame_ctx.frame = Frame::Cpu(integration_frame);

        let elapsed = start.elapsed();
        tracing::debug!("Frame Stacking post-processing took {:?}", elapsed);

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "frame_stacker"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
