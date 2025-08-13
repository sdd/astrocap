use astrocap_core::structs::DetectedPoint;
use astrocap_core::traits::PointDetector;
use astrocap_core::Frame;
use image::Pixel;
use ordered_float::OrderedFloat;
use serde::Deserialize;
use std::sync::Arc;

// const POINT_SKIP_STEP: u32 = 3;
const DEFAULT_POINT_EXCLUSION_RADIUS_2: i32 = 400;
const DEFAULT_POINT_THRESHOLD: u8 = 50;
const DEFAULT_STEP_X: u32 = 1;
const DEFAULT_STEP_Y: u32 = 1;
// pub const PATCH_SIZE: i32 = 4;

#[derive(Debug, Deserialize)]
#[serde(default)]
pub(crate) struct PeakConfig {
    point_exclusion_radius_2: i32,
    point_threshold: u8,
    step_x: u32,
    step_y: u32,
}

impl Default for PeakConfig {
    fn default() -> Self {
        Self {
            point_exclusion_radius_2: DEFAULT_POINT_EXCLUSION_RADIUS_2,
            point_threshold: DEFAULT_POINT_THRESHOLD,
            step_x: DEFAULT_STEP_X,
            step_y: DEFAULT_STEP_Y,
        }
    }
}

pub struct PointDetectPeak {
    pub(crate) config: PeakConfig,
}

impl PointDetector for PointDetectPeak {
    fn detect(
        &self,
        img: &Frame,
        _median: Option<Arc<Frame>>,
        mask: Option<&Frame>,
    ) -> Vec<DetectedPoint> {
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

        let mut x: u32 = 0;
        let mut y: u32 = 0;

        while y < img_h {
            while x < img_w {
                if let Some(mask) = mask {
                    if mask.get_pixel(x, y).channels()[0] == 0 {
                        x += self.config.step_x;
                        continue;
                    }
                }

                let val: u8 = img.get_pixel(x, y).channels()[0];

                if val > self.config.point_threshold {
                    let mut curr_val: u8 = val;
                    let mut point_x = x;
                    while (point_x + 1) < img_w && img.get_pixel(x + 1, y).channels()[0] > curr_val
                    {
                        point_x += 1;
                        curr_val = img.get_pixel(x, y).channels()[0];
                    }

                    let mut point_y = y;
                    while point_y + 1 < img_h && img.get_pixel(x, y + 1).channels()[0] > curr_val {
                        point_y += 1;
                        curr_val = img.get_pixel(x, y).channels()[0]
                    }

                    // find existing matches within POINT_EXCLUSION_RADIUS of current match
                    let mut matching: Vec<_> = points
                        .iter()
                        .enumerate()
                        .filter(|&(_idx, existing)| {
                            let xd = (existing.x as i32 - point_x as i32).abs();
                            let yd = (existing.y as i32 - point_y as i32).abs();
                            ((xd * xd) + (yd * yd)) < self.config.point_exclusion_radius_2
                        })
                        .map(|(n, i)| (n, i.clone()))
                        .collect();

                    matching.sort_by_key(|(_, p)| OrderedFloat(p.amplitude));

                    // if one of these nearby existing matches has higher amplitude, skip
                    // the current point
                    let curr_val_f32 = curr_val as f32;
                    if let Some((_, last)) = matching.last() {
                        if last.amplitude > curr_val_f32 {
                            x += self.config.step_x;
                            continue;
                        } else {
                            // otherwise remove the matches in favour of our new one
                            matching.sort_by_key(|(idx, _)| 0 - (*idx as isize));
                            let points_len = points.len();
                            for (idx, _) in matching.iter() {
                                if *idx < points_len {
                                    points.remove(*idx);
                                }
                            }
                        }
                    }

                    let new_point = DetectedPoint {
                        x: point_x as f32,
                        y: point_y as f32,
                        amplitude: curr_val_f32,
                        fitted: None,
                    };

                    points.push(new_point.clone());
                }

                x += self.config.step_x;
            }
            x = 0;
            y += self.config.step_y;
        }

        points
    }
}

#[allow(dead_code)]
pub fn inside_existing_point(
    x: i32,
    y: i32,
    points: &[DetectedPoint],
    point_exclusion_radius_2: i32,
) -> bool {
    points.iter().any(|point| {
        let xd = (point.x as i32 - x).abs();
        let yd = (point.y as i32 - y).abs();
        ((xd * xd) + (yd * yd)) < point_exclusion_radius_2
    })
}

#[cfg(test)]
mod tests {
    use super::{
        PeakConfig, PointDetectPeak, DEFAULT_POINT_EXCLUSION_RADIUS_2, DEFAULT_POINT_THRESHOLD,
        DEFAULT_STEP_X, DEFAULT_STEP_Y,
    };
    use astrocap_core::structs::DetectedPoint;
    use astrocap_core::traits::PointDetector;
    use image::io::Reader as ImageReader;
    use image::{GrayImage, ImageBuffer, Luma};
    use imageproc::filter::median_filter;
    use imageproc::map::map_colors2;
    use kiddo::float::kdtree::KdTree;
    use kiddo::SquaredEuclidean;
    use std::collections::HashSet;
    use std::fs::File;
    use std::sync::Arc;

    type Tree = KdTree<f64, usize, 2, 32, u32>;

    struct Point {
        x: usize,
        y: usize,
        amp: u8,
    }

    const MAX_PERMITTED_MATCH_DIST2: f64 = 4.0;
    const MATCH_EXCLUSION_DIST2: f64 = 10.0;
    #[test]
    fn can_detect_known_stars() {
        // Load a set of images with known good star positions
        let raw_img = ImageReader::open("../test-images/astrocap_model/test-image-1.png")
            .unwrap()
            .decode()
            .unwrap();
        let img_height = raw_img.height();
        let img_width = raw_img.width();
        const MAX_FALSE_POSITIVES: usize = 500;

        let img =
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(img_width, img_height, raw_img.into_bytes())
                .expect("Could not create ImageBuffer from VideoFrame");

        // pre-process the images as per the pipeline
        let img_median: GrayImage = median_filter(&img, 30, 30);
        let subtracted = map_colors2(&img, &img_median, |p, q| {
            Luma([(p[0] as u8).saturating_sub(q[0] as u8)])
        });

        subtracted
            .save("../test-images/astrocap_model/test-image-1-subtracted.png")
            .unwrap();

        // perform the extract
        let config = PeakConfig {
            step_y: DEFAULT_STEP_Y,
            step_x: DEFAULT_STEP_X,
            point_threshold: DEFAULT_POINT_THRESHOLD,
            point_exclusion_radius_2: DEFAULT_POINT_EXCLUSION_RADIUS_2,
        };
        let point_detect_peak = PointDetectPeak { config };
        let results: Vec<DetectedPoint> =
            point_detect_peak.detect(&subtracted.into(), Some(&img_median.into()), None);

        serde_json::to_writer(
            File::create("../test-images/astrocap_model/test-image-1-detected.json").unwrap(),
            &results,
        )
        .unwrap();

        let mut tree: Tree = Tree::new();
        for (idx, point) in results.iter().enumerate() {
            tree.add(&[point.x as f64, point.y as f64], idx);
        }

        let known_good = [
            Point {
                x: 1054,
                y: 506,
                amp: 80,
            },
            Point {
                x: 1291,
                y: 427,
                amp: 52,
            },
            Point {
                x: 1578,
                y: 522,
                amp: 62,
            },
            Point {
                x: 316,
                y: 894,
                amp: 78,
            },
        ];

        // assert that the known good stars are detected
        let mut matched_indexes: HashSet<usize> = HashSet::new();
        for point in known_good.iter() {
            let nearest = tree.nearest_one::<SquaredEuclidean>(&[point.x as f64, point.y as f64]);
            assert!(nearest.distance < MAX_PERMITTED_MATCH_DIST2);
            matched_indexes.insert(nearest.item);
        }
        assert_eq!(matched_indexes.len(), known_good.len());

        // assert that there are no stars within an exclusion zone around each known good star
        for point in known_good.iter() {
            let nearest = tree.within::<SquaredEuclidean>(
                &[point.x as f64, point.y as f64],
                MATCH_EXCLUSION_DIST2,
            );
            assert_eq!(nearest.len(), 1);
        }

        // assert that the number of false positives are within a reasonable limit
        assert!(results.len() - 4 < MAX_FALSE_POSITIVES);
    }
}
