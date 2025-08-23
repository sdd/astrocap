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

        let img_w = img.width() as usize;
        let img_h = img.height() as usize;

        let img_data: &[u8] = img.as_raw().as_ref();
        let mask_data = mask.map(|m| m.as_raw());

        // First pass: find local maxima
        for y in
            (self.config.centroid_window as usize)..(img_h - self.config.centroid_window as usize)
        {
            for x in (self.config.centroid_window as usize)
                ..(img_w - self.config.centroid_window as usize)
            {
                // Check mask
                if let Some(mask_data) = mask_data {
                    if mask_data[y * img_w + x] == 0 {
                        continue;
                    }
                }

                let center_val: u8 = img_data[y * img_w + x];

                if center_val < self.config.point_threshold {
                    continue;
                }

                // Optimized 3x3 neighborhood check - explicit comparisons
                let is_local_max = center_val > img_data[(y - 1) * img_w + (x - 1)]
                    && center_val > img_data[(y - 1) * img_w + x]
                    && center_val > img_data[(y - 1) * img_w + (x + 1)]
                    && center_val > img_data[y * img_w + (x - 1)]
                    && center_val > img_data[y * img_w + (x + 1)]
                    && center_val > img_data[(y + 1) * img_w + (x - 1)]
                    && center_val > img_data[(y + 1) * img_w + x]
                    && center_val > img_data[(y + 1) * img_w + (x + 1)];

                if is_local_max {
                    // Calculate centroid in larger window for sub-pixel accuracy
                    let (centroid_x, centroid_y, peak_intensity) = calculate_centroid(
                        img,
                        x,
                        y,
                        self.config.centroid_window as isize,
                        img_w,
                        img_h,
                    );

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
    img: &[u8],
    center_x: usize,
    center_y: usize,
    window: isize,
    img_w: usize,
    img_h: usize,
) -> (f32, f32, f32) {
    let mut sum_intensity = 0.0;
    let mut sum_x_weighted = 0.0;
    let mut sum_y_weighted = 0.0;
    let mut peak_intensity = 0.0;

    for dy in -window..=window {
        for dx in -window..=window {
            let x = ((center_x as isize) + dx) as usize;
            let y = ((center_y as isize) + dy) as usize;

            let intensity = img[y * img_w + x] as f32;

            if intensity > peak_intensity {
                peak_intensity = intensity;
            }

            sum_intensity += intensity;
            sum_x_weighted += intensity * x as f32;
            sum_y_weighted += intensity * y as f32;
        }
    }

    (
        sum_x_weighted / sum_intensity,
        sum_y_weighted / sum_intensity,
        peak_intensity,
    )
}
