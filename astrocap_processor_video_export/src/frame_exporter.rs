use std::io::Write;
use std::process::{Command, Stdio};

use crate::config::Config;
use astrocap_core::{AstrocapError, Frame};
use vyd::frame::CpuFrame;
use vyd::frame::pool::WaitPolicy;
use vyd::pipeline::PipelineContext;
use vyd::statistics::{MemoryOperation, StatsContext};

// Simple PNG sequence exporter for lossless storage
pub struct FrameExporter {
    output_path: String,
    crop_config: CropConfig,
    frame_count: usize,
    is_open: bool,
    is_closed_permanently: bool, // Add this flag
    ffmpeg_process: Option<std::process::Child>,
    width: usize,
    height: usize,
    // Cache the optimized crop bounds after first calculation
    optimized_crop_bounds: Option<(usize, usize, usize, usize)>,
    input_dimensions: Option<(u32, u32)>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyd::frame::CpuAllocationOrigin;

    #[test]
    fn crop_copies_rows_into_pooled_output() {
        let mut config = Config::default();
        config.crop_left_px = 1;
        config.crop_right_px = 3;
        config.crop_top_px = 0;
        config.crop_bottom_px = 2;
        let mut exporter = FrameExporter::new("unused.mkv", &config).unwrap();
        let ctx = PipelineContext::new();
        let input = CpuFrame::from_vec(4, 2, vec![0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
        let output = exporter.crop_frame(&input, &ctx).unwrap();
        // Alignment expands this small crop to the source bounds.
        assert_eq!(
            output.try_read_cpu_luma8(WaitPolicy::Immediate).unwrap(),
            &[0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            output
                .to_cpu(None)
                .unwrap()
                .img
                .as_raw()
                .allocation_origin(),
            CpuAllocationOrigin::Pool
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
    }

    #[test]
    fn invalid_crop_returns_error_before_starting_ffmpeg() {
        let config = Config::default();
        let mut exporter = FrameExporter::new("unused.mkv", &config).unwrap();
        let ctx = PipelineContext::new();
        let input = CpuFrame::from_vec(2, 2, vec![0; 4]).unwrap();
        assert!(exporter.crop_frame(&input, &ctx).is_err());
        assert_eq!(
            ctx.frame_factory()
                .cpu_pool()
                .snapshot()
                .unwrap()
                .active
                .len(),
            0
        );
    }
}

#[derive(Clone)]
struct CropConfig {
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
}

impl FrameExporter {
    pub(crate) fn new(output_path: &str, config: &Config) -> Result<Self, AstrocapError> {
        let crop_config = CropConfig {
            left: config.crop_left_px,
            right: config.crop_right_px,
            top: config.crop_top_px,
            bottom: config.crop_bottom_px,
        };

        Ok(Self {
            output_path: output_path.to_string(),
            crop_config,
            frame_count: 0,
            is_open: false,
            is_closed_permanently: false, // Initialize the flag
            ffmpeg_process: None,
            width: 0,
            height: 0,
            optimized_crop_bounds: None,
            input_dimensions: None,
        })
    }

    /// Get optimal crop bounds, calculating only once and caching the result
    fn get_optimized_crop_bounds(
        &mut self,
        frame_width: u32,
        frame_height: u32,
    ) -> Result<(usize, usize, usize, usize), AstrocapError> {
        if let Some(bounds) = self.optimized_crop_bounds {
            return Ok(bounds);
        }

        // Calculate the bounds for the first time
        let frame_w = frame_width as usize;
        let frame_h = frame_height as usize;
        if self.crop_config.left >= self.crop_config.right
            || self.crop_config.top >= self.crop_config.bottom
            || self.crop_config.right > frame_w
            || self.crop_config.bottom > frame_h
        {
            return Err(AstrocapError::GeneralPluginError(format!(
                "invalid video crop bounds for frame {frame_w}x{frame_h}"
            )));
        }

        // Start with the configured crop area
        let mut left = self.crop_config.left;
        let mut right = self.crop_config.right;
        let mut top = self.crop_config.top;
        let mut bottom = self.crop_config.bottom;

        // Calculate current crop dimensions
        let crop_width = right - left;
        let crop_height = bottom - top;

        // Calculate how much we need to expand to reach multiple of 16
        let target_width = crop_width.saturating_add(15) / 16 * 16;
        let target_height = crop_height.saturating_add(15) / 16 * 16;

        let width_expansion = target_width - crop_width;
        let height_expansion = target_height - crop_height;

        // Distribute expansion evenly on both sides, preferring to expand outward
        let left_expand = width_expansion / 2;
        let right_expand = width_expansion - left_expand;
        let top_expand = height_expansion / 2;
        let bottom_expand = height_expansion - top_expand;

        // Apply expansion while respecting frame boundaries
        left = left.saturating_sub(left_expand);
        right = right.saturating_add(right_expand).min(frame_w);
        top = top.saturating_sub(top_expand);
        bottom = bottom.saturating_add(bottom_expand).min(frame_h);

        // If we couldn't expand enough due to frame boundaries, expand the other direction
        let actual_width = right - left;
        let actual_height = bottom - top;

        if actual_width < target_width && left > 0 {
            let additional_left = (target_width - actual_width).min(left);
            left -= additional_left;
        }
        if actual_width < target_width && right < frame_w {
            let additional_right = (target_width - actual_width).min(frame_w - right);
            right += additional_right;
        }
        if actual_height < target_height && top > 0 {
            let additional_top = (target_height - actual_height).min(top);
            top -= additional_top;
        }
        if actual_height < target_height && bottom < frame_h {
            let additional_bottom = (target_height - actual_height).min(frame_h - bottom);
            bottom += additional_bottom;
        }

        let bounds = (left, right, top, bottom);

        // Log the optimization result only once
        tracing::info!(
            "Crop optimization: {}x{} -> {}x{}, bounds: ({},{}) to ({},{}) -> ({},{}) to ({},{})",
            crop_width,
            crop_height,
            right - left,
            bottom - top,
            self.crop_config.left,
            self.crop_config.top,
            self.crop_config.right,
            self.crop_config.bottom,
            left,
            top,
            right,
            bottom
        );

        // Cache the result
        self.optimized_crop_bounds = Some(bounds);
        Ok(bounds)
    }

    fn crop_frame(
        &mut self,
        frame: &CpuFrame,
        ctx: &PipelineContext,
    ) -> Result<Frame, AstrocapError> {
        let frame_width = frame.width();
        let frame_height = frame.height();

        // Get cached optimal crop bounds
        let (left, right, top, bottom) =
            self.get_optimized_crop_bounds(frame_width, frame_height)?;

        let width = frame_width as usize;
        let height = frame_height as usize;

        // Validate bounds
        if right > width || bottom > height {
            return Err(AstrocapError::GeneralPluginError(
                "Optimized crop bounds exceed frame dimensions".to_string(),
            ));
        }

        let crop_width = right - left;
        let crop_height = bottom - top;

        let input = frame
            .try_read_luma8(WaitPolicy::Immediate)
            .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?;
        let cropped = ctx
            .frame_factory()
            .create_luma8_with(
                crop_width as u32,
                crop_height as u32,
                WaitPolicy::Immediate,
                |output| {
                    for (row, y) in (top..bottom).enumerate() {
                        let row_start = y * width + left;
                        let row_end = y * width + right;
                        output[row * crop_width..(row + 1) * crop_width]
                            .copy_from_slice(&input[row_start..row_end]);
                    }
                    Ok(())
                },
            )
            .map_err(|e| {
                AstrocapError::GeneralPluginError(format!("Failed to create cropped frame: {e}"))
            })?;
        StatsContext::record_memory_operation(MemoryOperation::CpuCopy(crop_width * crop_height));
        Ok(cropped)
    }

    /// Initialize dimensions and open FFmpeg process if not already done
    fn ensure_open(&mut self, frame: &Frame) -> Result<(), AstrocapError> {
        if self.is_open || self.is_closed_permanently {
            return Ok(());
        }

        let cpu_frame = match frame {
            Frame::Cpu(cpu_frame) => cpu_frame,
            _ => {
                return Err(AstrocapError::GeneralPluginError(
                    "Cannot determine dimensions from non-CPU frame".to_string(),
                ));
            }
        };

        // Set dimensions from the first frame
        if self.width == 0 || self.height == 0 {
            let (left, right, top, bottom) =
                self.get_optimized_crop_bounds(cpu_frame.width(), cpu_frame.height())?;
            self.width = right - left;
            self.height = bottom - top;

            tracing::info!(
                "Video export dimensions set to {}x{} (optimized from crop config)",
                self.width,
                self.height
            );
        }

        // Start FFmpeg process with the determined dimensions
        let ffmpeg = Command::new("ffmpeg")
            .args([
                "-f",
                "rawvideo",
                "-pixel_format",
                "gray",
                "-video_size",
                &format!("{}x{}", self.width, self.height),
                "-framerate",
                "30",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                // "-crf",
                // "18", // Changed from -qp 0 for better compatibility
                "-qp",
                "0", // Back to lossless
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart", // Better playback compatibility
                "-y",
                &self.output_path,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| {
                AstrocapError::GeneralPluginError(format!(
                    "Failed to start FFmpeg process: {}. Make sure FFmpeg is installed.",
                    e
                ))
            })?;

        // Verify we can write to stdin
        if ffmpeg.stdin.is_none() {
            return Err(AstrocapError::GeneralPluginError(
                "Failed to get FFmpeg stdin pipe".to_string(),
            ));
        }

        self.ffmpeg_process = Some(ffmpeg);
        self.is_open = true;
        self.frame_count = 0; // Reset frame count when opening

        tracing::info!(
            "Video export opened: {} ({}x{})",
            self.output_path,
            self.width,
            self.height
        );

        Ok(())
    }

    // Remove the old calculate_optimal_crop_bounds method as it's now integrated into get_optimized_crop_bounds
}
impl FrameExporter {
    pub(crate) fn write_frame(
        &mut self,
        frame: &Frame,
        ctx: &PipelineContext,
    ) -> Result<(), AstrocapError> {
        // Check if the exporter has been permanently closed
        if self.is_closed_permanently {
            tracing::trace!("Ignoring write_frame() call - exporter is permanently closed");
            return Ok(());
        }

        let cpu_frame = match frame {
            Frame::Cpu(cpu_frame) => cpu_frame,
            _ => {
                return Err(AstrocapError::GeneralPluginError(
                    "Cannot export non-CPU frame".into(),
                ));
            }
        };
        let dimensions = cpu_frame.dimensions();
        if let Some(previous) = self.input_dimensions {
            if previous != dimensions {
                return Err(AstrocapError::GeneralPluginError(format!(
                    "video export dimensions changed from {previous:?} to {dimensions:?}"
                )));
            }
        } else {
            self.input_dimensions = Some(dimensions);
        }

        // Ensure the exporter is open before writing frames
        self.ensure_open(frame)?;

        let cropped_frame = self.crop_frame(cpu_frame, ctx)?;
        let cropped_cpu = cropped_frame.to_cpu_frame().ok_or_else(|| {
            AstrocapError::GeneralPluginError("crop result is not a CPU frame".into())
        })?;

        // Add debugging info
        tracing::debug!(
            "Writing frame {} ({}x{}, {} bytes)",
            self.frame_count,
            cropped_cpu.width(),
            cropped_cpu.height(),
            cropped_cpu
                .try_read_luma8(WaitPolicy::Immediate)
                .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?
                .len()
        );

        self.write_raw_frame_to_ffmpeg(&cropped_cpu)?;
        self.frame_count += 1;

        // Log every 100 frames to track progress
        if self.frame_count % 100 == 0 {
            tracing::info!("Exported {} frames so far", self.frame_count);
        }

        Ok(())
    }

    fn write_raw_frame_to_ffmpeg(&mut self, frame: &CpuFrame) -> Result<(), AstrocapError> {
        if let Some(ref mut process) = self.ffmpeg_process {
            // Check if the process is still alive first
            match process.try_wait() {
                Ok(Some(status)) => {
                    // Process has exited - read stderr for diagnostics
                    let mut error_output = String::new();
                    if let Some(stderr) = process.stderr.take() {
                        use std::io::Read;
                        let _ = std::io::BufReader::new(stderr).read_to_string(&mut error_output);
                    }
                    tracing::error!(
                        "FFmpeg process died after {} frames with status {}: {}",
                        self.frame_count,
                        status,
                        error_output
                    );
                    return Err(AstrocapError::GeneralPluginError(format!(
                        "FFmpeg process died after {} frames with status: {}",
                        self.frame_count, status
                    )));
                }
                Ok(None) => {
                    // Process is still running, continue
                }
                Err(e) => {
                    return Err(AstrocapError::GeneralPluginError(format!(
                        "Failed to check FFmpeg process status: {}",
                        e
                    )));
                }
            }

            if let Some(ref mut stdin) = process.stdin {
                let pixel_data = frame
                    .try_read_luma8(WaitPolicy::Immediate)
                    .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?;
                tracing::trace!("Writing {} bytes to FFmpeg stdin", pixel_data.len());

                match stdin.write_all(pixel_data) {
                    Ok(_) => {
                        // Flush to ensure data is sent immediately
                        if let Err(e) = stdin.flush() {
                            tracing::warn!("Failed to flush FFmpeg stdin: {}", e);
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to write frame data to FFmpeg: {}", e);
                        return Err(AstrocapError::GeneralPluginError(format!(
                            "Failed to write frame to FFmpeg: {}",
                            e
                        )));
                    }
                }
            } else {
                return Err(AstrocapError::GeneralPluginError(
                    "FFmpeg stdin not available".to_string(),
                ));
            }
        } else {
            return Err(AstrocapError::GeneralPluginError(
                "FFmpeg process not running".to_string(),
            ));
        }

        Ok(())
    }

    pub(crate) fn close(&mut self) -> Result<(), AstrocapError> {
        if !self.is_open {
            return Ok(());
        }

        tracing::info!("Closing video exporter after {} frames", self.frame_count);
        self.is_open = false;
        self.is_closed_permanently = true; // Set the permanent flag

        if let Some(mut process) = self.ffmpeg_process.take() {
            // Close stdin to signal end of input
            if let Some(stdin) = process.stdin.take() {
                drop(stdin);
            }

            // Wait for FFmpeg to finish processing and exit
            match process.wait() {
                Ok(status) => {
                    if status.success() {
                        tracing::info!(
                            "Video export completed successfully: {} ({} frames)",
                            self.output_path,
                            self.frame_count
                        );
                    } else {
                        // Read stderr for error details
                        let mut error_output = String::new();
                        if let Some(stderr) = process.stderr.take() {
                            use std::io::Read;
                            let _ =
                                std::io::BufReader::new(stderr).read_to_string(&mut error_output);
                        }
                        tracing::error!(
                            "FFmpeg process exited with status {} after {} frames: {}",
                            status,
                            self.frame_count,
                            error_output
                        );
                        return Err(AstrocapError::GeneralPluginError(format!(
                            "FFmpeg exited with status {status}: {error_output}"
                        )));
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to wait for FFmpeg process: {}", e);
                    return Err(AstrocapError::GeneralPluginError(format!(
                        "Failed to wait for FFmpeg process: {}",
                        e
                    )));
                }
            }
        }

        tracing::info!("Video exporter permanently closed");
        Ok(())
    }
}
