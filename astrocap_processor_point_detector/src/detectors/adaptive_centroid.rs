use crate::state::DetectedPoint;
use crate::traits::{AstroFloat, ImageLumaExtractor, PointDetector};
use az::{Az, Cast};
use std::sync::Arc;

const BASE_THRESHOLD: u8 = 20; // Base threshold for dark regions
const DARK_THRESHOLD: u8 = 90; // Median value that we consider "dark"
const BRIGHT_THRESHOLD: u8 = 140; // Median value that we consider "bright"
const DARK_AREA_THRESHOLD: u8 = 30; // Higher threshold in dark areas (more noise)
const BRIGHT_AREA_THRESHOLD: u8 = 20; // Lower threshold in bright areas (better SNR)

const MIN_SEPARATION: f64 = 20.0;
const CENTROID_WINDOW: i32 = 5; // Window size for centroid calculation

pub struct PointDetectAdaptiveCentroid {}

impl<F: AstroFloat> PointDetector<F> for PointDetectAdaptiveCentroid
where
    u32: Cast<F>,
    u8: Cast<F>,
    F: Cast<u32>,
    f64: Cast<F>,
{
    fn detect(
        img: Arc<dyn ImageLumaExtractor>,
        median: Option<Arc<dyn ImageLumaExtractor>>,
        mask: Option<Arc<dyn ImageLumaExtractor>>,
    ) -> Vec<DetectedPoint<F>> {
        let img_w = img.width();
        let img_h = img.height();
        let mut candidates = Vec::new();

        // First pass: find local maxima with adaptive thresholds
        for y in 1..(img_h - 1) {
            for x in 1..(img_w - 1) {
                // Check mask
                if let Some(ref mask_img) = mask {
                    if mask_img.get_luma8_for_pixel(x, y) == 0 {
                        continue;
                    }
                }

                let center_val = img.get_luma8_for_pixel(x, y);

                // Calculate adaptive threshold based on local background
                let adaptive_threshold = if let Some(ref median_img) = median {
                    let local_background = median_img.get_luma8_for_pixel(x, y);

                    // Linear interpolation between dark and bright thresholds
                    if local_background <= DARK_THRESHOLD {
                        DARK_AREA_THRESHOLD
                    } else if local_background >= BRIGHT_THRESHOLD {
                        BRIGHT_AREA_THRESHOLD
                    } else {
                        // Linear interpolation between the two
                        let ratio = (local_background - DARK_THRESHOLD) as f32
                            / (BRIGHT_THRESHOLD - DARK_THRESHOLD) as f32;
                        let interpolated = DARK_AREA_THRESHOLD as f32 * (1.0 - ratio)
                            + BRIGHT_AREA_THRESHOLD as f32 * ratio;
                        interpolated as u8
                    }
                } else {
                    BASE_THRESHOLD
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

                        if img.get_luma8_for_pixel(nx, ny) > center_val {
                            is_local_max = false;
                            break 'outer;
                        }
                    }
                }

                if is_local_max {
                    // Calculate centroid in larger window for sub-pixel accuracy
                    let (centroid_x, centroid_y, peak_intensity) =
                        calculate_centroid(&*img, x, y, CENTROID_WINDOW, mask.as_ref());

                    candidates.push(DetectedPoint {
                        x: centroid_x.round() as u32,
                        y: centroid_y.round() as u32,
                        amplitude: peak_intensity.az::<F>(),
                        fitted_point: None,
                    });
                }
            }
        }

        // Second pass: remove points that are too close to each other
        let mut final_points = Vec::new();
        candidates.sort_by(|a, b| b.amplitude.partial_cmp(&a.amplitude).unwrap());

        for candidate in candidates {
            let too_close = final_points.iter().any(|existing: &DetectedPoint<F>| {
                let dx = candidate.x as f64 - existing.x as f64;
                let dy = candidate.y as f64 - existing.y as f64;
                (dx * dx + dy * dy) < MIN_SEPARATION * MIN_SEPARATION
            });

            if !too_close {
                final_points.push(candidate);
            }
        }

        final_points
    }
}

fn calculate_centroid(
    img: &dyn ImageLumaExtractor,
    center_x: u32,
    center_y: u32,
    window: i32,
    mask: Option<&Arc<dyn ImageLumaExtractor>>,
) -> (f64, f64, f64) {
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
                    if mask_img.get_luma8_for_pixel(x_u32, y_u32) == 0 {
                        continue;
                    }
                }

                let intensity = img.get_luma8_for_pixel(x_u32, y_u32) as f64;

                if intensity > peak_intensity {
                    peak_intensity = intensity;
                }

                sum_intensity += intensity;
                sum_x_weighted += intensity * x as f64;
                sum_y_weighted += intensity * y as f64;
            }
        }
    }

    if sum_intensity > 0.0 {
        (
            sum_x_weighted / sum_intensity,
            sum_y_weighted / sum_intensity,
            peak_intensity,
        )
    } else {
        (center_x as f64, center_y as f64, peak_intensity)
    }
}
