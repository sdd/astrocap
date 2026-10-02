use astrocap_core::structs::Detection;
use astrocap_core::traits::PointDetector;
use astrocap_core::Frame;
use image::Pixel;
use serde::Deserialize;
use std::sync::Arc;
use vyd::frame::CpuImgBuf;

const DEFAULT_POINT_THRESHOLD: u8 = 40;
const DEFAULT_MIN_SEPARATION: f64 = 20.0; // Minimum separation between stars (pixels)
const DEFAULT_CENTROID_WINDOW: i32 = 5; // Window size for centroid calculation

// when computing the background noise we need to mask out the objects
// so we just use background pixels. The upper threshold is the maximum
// value that we consider to be background.
// Since we're currently clamping the median-sub to 0, it can be advisable
// to set the lower threshold to 1 rather than 0 to avoid the spike of values
// at 0.
const ESTIMATE_SIGMA_LOWER_THRESHOLD: u8 = 1;
const ESTIMATE_SIGMA_UPPER_THRESHOLD: u8 = 100;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub(crate) struct LocalMaximaConfig {
    point_threshold: u8,
    min_separation: f64,
    centroid_window: i32,
    estimate_sigma_enabled: bool,
    estimate_sigma_lower_threshold: u8,
    estimate_sigma_upper_threshold: u8,
}

impl Default for LocalMaximaConfig {
    fn default() -> Self {
        Self {
            point_threshold: DEFAULT_POINT_THRESHOLD,
            min_separation: DEFAULT_MIN_SEPARATION,
            centroid_window: DEFAULT_CENTROID_WINDOW,
            estimate_sigma_enabled: false,
            estimate_sigma_lower_threshold: ESTIMATE_SIGMA_LOWER_THRESHOLD,
            estimate_sigma_upper_threshold: ESTIMATE_SIGMA_UPPER_THRESHOLD,
        }
    }
}

pub struct PointDetectLocalMaxima {
    pub(crate) config: LocalMaximaConfig,
    pub sigma: Option<f32>,
}

impl PointDetectLocalMaxima {
    fn calculate_sigma(&mut self, mut vals: Vec<u8>) {
        if vals.is_empty() {
            return;
        } // fallback

        // compute median
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = vals.len();
        let median = if n % 2 == 1 {
            vals[n / 2] as f32
        } else {
            0.5 * (vals[n / 2 - 1] as f32 + vals[n / 2] as f32)
        };

        // compute MAD
        let mut devs = Vec::with_capacity(n);
        for &x in &vals {
            devs.push((x as f32 - median).abs());
        }
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if n % 2 == 1 {
            devs[n / 2]
        } else {
            0.5 * (devs[n / 2 - 1] + devs[n / 2])
        };

        // Convert to sigma (Gaussian assumption for core)
        self.sigma = Some(1.4826 * mad);

        tracing::info!(val_count=vals.len(), median, mad, sigma=?self.sigma, "calculated sigma")
    }
}

impl PointDetector for PointDetectLocalMaxima {
    fn detect(
        &mut self,
        img: &Frame,
        _median: Option<Arc<Frame>>,
        mask: Option<&Frame>,
    ) -> Vec<Detection> {
        let min_separation_2 = self.config.min_separation * self.config.min_separation;

        let mut points: Vec<Detection> = vec![];

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

        let mut background_pixels: Vec<u8> = Vec::new();

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

                if self.config.estimate_sigma_enabled
                    && center_val >= self.config.estimate_sigma_lower_threshold
                    && center_val < self.config.estimate_sigma_upper_threshold
                {
                    background_pixels.push(center_val);
                }

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

                    points.push(Detection::new(centroid_x, centroid_y, peak_intensity));
                }
            }
        }

        // Second pass: remove points that are too close to each other
        let mut final_points = Vec::new();
        points.sort_by(|a, b| b.amplitude.partial_cmp(&a.amplitude).unwrap());

        for candidate in points {
            let too_close = final_points.iter().any(|existing: &Detection| {
                let dx = candidate.position.x as f64 - existing.position.x as f64;
                let dy = candidate.position.y as f64 - existing.position.y as f64;
                (dx * dx + dy * dy) < min_separation_2
            });

            if !too_close {
                final_points.push(candidate);
            }
        }

        if self.config.estimate_sigma_enabled {
            self.calculate_sigma(background_pixels);
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
