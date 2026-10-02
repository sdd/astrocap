use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

use crate::config::Config;
use astrocap_core::{AstrocapError, Frame};
use vyd::frame::CpuFrame;

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
        })
    }

    /// Get optimal crop bounds, calculating only once and caching the result
    fn get_optimized_crop_bounds(
        &mut self,
        frame_width: u32,
        frame_height: u32,
    ) -> (usize, usize, usize, usize) {
        if let Some(bounds) = self.optimized_crop_bounds {
            return bounds;
        }

        // Calculate the bounds for the first time
        let frame_w = frame_width as usize;
        let frame_h = frame_height as usize;

        // Start with the configured crop area
        let mut left = self.crop_config.left;
        let mut right = self.crop_config.right;
        let mut top = self.crop_config.top;
        let mut bottom = self.crop_config.bottom;

        // Calculate current crop dimensions
        let crop_width = right - left;
        let crop_height = bottom - top;

        // Calculate how much we need to expand to reach multiple of 16
        let target_width = ((crop_width + 15) / 16) * 16; // Round up to nearest 16
        let target_height = ((crop_height + 15) / 16) * 16; // Round up to nearest 16

        let width_expansion = target_width - crop_width;
        let height_expansion = target_height - crop_height;

        // Distribute expansion evenly on both sides, preferring to expand outward
        let left_expand = width_expansion / 2;
        let right_expand = width_expansion - left_expand;
        let top_expand = height_expansion / 2;
        let bottom_expand = height_expansion - top_expand;

        // Apply expansion while respecting frame boundaries
        left = left.saturating_sub(left_expand);
        right = (right + right_expand).min(frame_w);
        top = top.saturating_sub(top_expand);
        bottom = (bottom + bottom_expand).min(frame_h);

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
        bounds
    }

    fn crop_frame(&mut self, frame: &CpuFrame) -> Result<CpuFrame, AstrocapError> {
        let frame_width = frame.width();
        let frame_height = frame.height();

        // Get cached optimal crop bounds
        let (left, right, top, bottom) = self.get_optimized_crop_bounds(frame_width, frame_height);

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

        let mut cropped_data = Vec::with_capacity(crop_width * crop_height);

        // Copy cropped rows
        for y in top..bottom {
            let row_start = y * width + left;
            let row_end = y * width + right;
            let row_data = &frame.img.as_raw()[row_start..row_end];
            cropped_data.extend_from_slice(row_data);
        }

        CpuFrame::from_vec(crop_width as u32, crop_height as u32, cropped_data).map_err(|e| {
            AstrocapError::GeneralPluginError(format!("Failed to create cropped frame: {}", e))
        })
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
                self.get_optimized_crop_bounds(cpu_frame.width(), cpu_frame.height());
            self.width = right - left;
            self.height = bottom - top;

            tracing::info!(
                "Video export dimensions set to {}x{} (optimized from crop config)",
                self.width,
                self.height
            );
        }

        // Start FFmpeg process with the determined dimensions
        let mut ffmpeg = Command::new("ffmpeg")
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
    pub(crate) fn write_frame(&mut self, frame: &Frame) -> Result<(), AstrocapError> {
        // Check if the exporter has been permanently closed
        if self.is_closed_permanently {
            tracing::trace!("Ignoring write_frame() call - exporter is permanently closed");
            return Ok(());
        }

        let cpu_frame = match frame {
            Frame::Cpu(cpu_frame) => cpu_frame,
            _ => {
                tracing::warn!("Cannot export non-CPU frame");
                return Ok(());
            }
        };

        // Ensure the exporter is open before writing frames
        self.ensure_open(frame)?;

        let cropped_frame = self.crop_frame(cpu_frame)?;

        // Add debugging info
        tracing::debug!(
            "Writing frame {} ({}x{}, {} bytes)",
            self.frame_count,
            cropped_frame.width(),
            cropped_frame.height(),
            cropped_frame.img.as_raw().len()
        );

        self.write_raw_frame_to_ffmpeg(&cropped_frame)?;
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
                let pixel_data = frame.img.as_raw();
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
