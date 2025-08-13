use astrocap_core::frame::CpuImgBuf;
use astrocap_core::structs::DetectedPoint;
use astrocap_core::traits::PointDetector;
use astrocap_core::Frame;
use image::Pixel;
use serde::Deserialize;
use std::sync::Arc;

const DEFAULT_POINT_THRESHOLD: u8 = 40;
const DEFAULT_MIN_SEPARATION: f64 = 20.0; // Minimum separation between stars (pixels)
const DEFAULT_CENTROID_WINDOW: i32 = 5; // Window size for centroid calculation

#[derive(Debug, Deserialize)]
#[serde(default)]
pub(crate) struct LocalMaximaConfig {
    point_threshold: u8,
    min_separation: f64,
    centroid_window: i32,
}

impl Default for LocalMaximaConfig {
    fn default() -> Self {
        Self {
            point_threshold: DEFAULT_POINT_THRESHOLD,
            min_separation: DEFAULT_MIN_SEPARATION,
            centroid_window: DEFAULT_CENTROID_WINDOW,
        }
    }
}

pub struct PointDetectLocalMaxima {
    pub(crate) config: LocalMaximaConfig,
}

impl PointDetector for PointDetectLocalMaxima {
    fn detect(
        &self,
        img: &Frame,
        _median: Option<Arc<Frame>>,
        mask: Option<&Frame>,
    ) -> Vec<DetectedPoint> {
        let min_separation_2 = self.config.min_separation * self.config.min_separation;

        let mut points: Vec<DetectedPoint> = vec![];

        let Some(img) = img.as_cpu_image() else {
            tracing::error!("CPU image not retrieved for frame");
            return points;
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

        // First pass: find local maxima
        for y in 1..(img_h - 1) {
            for x in 1..(img_w - 1) {
                // Check mask
                if let Some(ref mask) = mask {
                    if mask.get_pixel(x, y).channels()[0] == 0 {
                        continue;
                    }
                }

                let center_val: u8 = img.get_pixel(x, y).channels()[0];

                if center_val < self.config.point_threshold {
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
                        calculate_centroid(&*img, x, y, self.config.centroid_window);

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
                let dx = candidate.x as f64 - existing.x as f64;
                let dy = candidate.y as f64 - existing.y as f64;
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
                let intensity = img.get_pixel(x as u32, y as u32).channels()[0] as f32;

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
