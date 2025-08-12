use astrocap_core::frame::CpuImgBuf;
use astrocap_core::structs::DetectedPoint;
use astrocap_core::traits::PointDetector;
use astrocap_core::Frame;
use image::Pixel;
use serde::Deserialize;

const DEFAULT_BASE_THRESHOLD: u8 = 20; // Base threshold for dark regions
const DEFAULT_DARK_THRESHOLD: u8 = 90; // Median value that we consider "dark"
const DEFAULT_BRIGHT_THRESHOLD: u8 = 140; // Median value that we consider "bright"
const DEFAULT_DARK_AREA_THRESHOLD: u8 = 30; // Higher threshold in dark areas (more noise)
const DEFAULT_BRIGHT_AREA_THRESHOLD: u8 = 20; // Lower threshold in bright areas (better SNR)

const DEFAULT_MIN_SEPARATION: f32 = 20.0;
const DEFAULT_CENTROID_WINDOW: i32 = 5; // Window size for centroid calculation

#[derive(Debug, Deserialize)]
#[serde(default)]
pub(crate) struct AdaptiveCentroidConfig {
    base_threshold: u8,
    dark_threshold: u8,
    bright_threshold: u8,
    dark_area_threshold: u8,
    bright_area_threshold: u8,
    min_separation: f32,
    centroid_window: i32,
}

impl Default for AdaptiveCentroidConfig {
    fn default() -> Self {
        Self {
            base_threshold: DEFAULT_BASE_THRESHOLD,
            dark_threshold: DEFAULT_DARK_THRESHOLD,
            bright_threshold: DEFAULT_BRIGHT_THRESHOLD,
            dark_area_threshold: DEFAULT_DARK_AREA_THRESHOLD,
            bright_area_threshold: DEFAULT_BRIGHT_AREA_THRESHOLD,
            min_separation: DEFAULT_MIN_SEPARATION,
            centroid_window: DEFAULT_CENTROID_WINDOW,
        }
    }
}

pub struct PointDetectAdaptiveCentroid {
    pub(crate) config: AdaptiveCentroidConfig,
}

impl PointDetector for PointDetectAdaptiveCentroid {
    fn detect(
        &self,
        img: &Frame,
        median: Option<&Frame>,
        mask: Option<&Frame>,
    ) -> Vec<DetectedPoint> {
        let min_separation_2 = self.config.min_separation * self.config.min_separation;
        let mut points: Vec<DetectedPoint> = vec![];

        let Some(img) = img.as_cpu_image() else {
            tracing::error!("CPU image not retrieved for frame");
            return points;
        };

        let median = if let Some(median) = median {
            let Some(median) = median.as_cpu_image() else {
                tracing::error!("CPU image not retrieved for median");
                return points;
            };
            Some(median)
        } else {
            None
        };

        let mask = if let Some(mask) = mask {
            let Some(mask) = mask.as_cpu_image() else {
                tracing::error!("CPU image not retrieved for frame");
                return points;
            };
            Some(mask)
        } else {
            None
        };

        let img_w = img.width();
        let img_h = img.height();

        // First pass: find local maxima with adaptive thresholds
        for y in 1..(img_h - 1) {
            for x in 1..(img_w - 1) {
                // Check mask
                if let Some(mask) = &mask {
                    if mask.get_pixel(x, y).channels()[0] == 0 {
                        continue;
                    }
                }

                let center_val = img.get_pixel(x, y).channels()[0];

                // Calculate adaptive threshold based on local background
                let adaptive_threshold = if let Some(median) = &median {
                    let local_background: u8 = median.get_pixel(x, y).channels()[0];

                    // Linear interpolation between dark and bright thresholds
                    if local_background <= self.config.dark_threshold {
                        self.config.dark_area_threshold
                    } else if local_background >= self.config.bright_threshold {
                        self.config.bright_area_threshold
                    } else {
                        // Linear interpolation between the two
                        let ratio = (local_background - self.config.dark_threshold) as f32
                            / (self.config.bright_threshold - self.config.dark_threshold) as f32;
                        let interpolated = self.config.dark_threshold as f32 * (1.0 - ratio)
                            + self.config.bright_area_threshold as f32 * ratio;
                        interpolated as u8
                    }
                } else {
                    self.config.base_threshold
                };

                if center_val < adaptive_threshold {
                    continue;
                }

                // Check if this is a local maximum in 3x3 neighborhood
                let mut is_local_max = true;
                'outer: for dy in -1..=1i32 {
                    for dx in -1..=1i32 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }

                        let nx = (x as i32 + dx) as u32;
                        let ny = (y as i32 + dy) as u32;

                        if img.get_pixel(nx, ny).channels()[0] > center_val {
                            is_local_max = false;
                            break 'outer;
                        }
                    }
                }

                if is_local_max {
                    // Calculate centroid in larger window for sub-pixel accuracy
                    let (centroid_x, centroid_y, peak_intensity) =
                        calculate_centroid(&*img, x, y, self.config.centroid_window, &mask);

                    points.push(DetectedPoint {
                        x: centroid_x.round(),
                        y: centroid_y.round(),
                        amplitude: peak_intensity,
                        fitted: None,
                    });
                }
            }
        }

        // Second pass: remove points that are too close to each other
        let mut final_points = Vec::new();
        points.sort_by(|a, b| b.amplitude.partial_cmp(&a.amplitude).unwrap());

        for candidate in points {
            let too_close = final_points.iter().any(|existing: &DetectedPoint| {
                let dx = candidate.x - existing.x;
                let dy = candidate.y - existing.y;
                (dx * dx + dy * dy) < min_separation_2
            });

            if !too_close {
                final_points.push(candidate);
            }
        }

        final_points
    }
}

fn calculate_centroid(
    img: &CpuImgBuf,
    center_x: u32,
    center_y: u32,
    window: i32,
    mask: &Option<&CpuImgBuf>,
) -> (f32, f32, f32) {
    let mut sum_intensity = 0.0;
    let mut sum_x_weighted = 0.0;
    let mut sum_y_weighted = 0.0;
    let mut peak_intensity = 0.0;

    for dy in -window..=window {
        for dx in -window..=window {
            let x = center_x as i32 + dx;
            let y = center_y as i32 + dy;

            if x >= 0 && y >= 0 && x < img.width() as i32 && y < img.height() as i32 {
                let x_u32 = x as u32;
                let y_u32 = y as u32;

                // Check mask - skip masked pixels
                if let Some(mask_img) = mask {
                    if mask_img.get_pixel(x_u32, y_u32).channels()[0] == 0 {
                        continue;
                    }
                }

                let intensity = img.get_pixel(x_u32, y_u32).channels()[0] as f32;

                if intensity > peak_intensity {
                    peak_intensity = intensity;
                }

                sum_intensity += intensity;
                sum_x_weighted += intensity * x as f32;
                sum_y_weighted += intensity * y as f32;
            }
        }
    }

    (
        sum_x_weighted / sum_intensity,
        sum_y_weighted / sum_intensity,
        peak_intensity,
    )
}
