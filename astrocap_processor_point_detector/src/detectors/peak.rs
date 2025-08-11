use crate::point_detector::DetectedPoint;
use crate::PointDetector;
use astrocap_core::Frame;

// const POINT_SKIP_STEP: u32 = 3;
const POINT_EXCLUSION_RADIUS_2: i32 = 400;
const POINT_THRESHOLD: u8 = 50;
const STEP_X: i32 = 1;
const STEP_Y: i32 = 1;
// pub const PATCH_SIZE: i32 = 4;

pub struct PointDetectPeak {}

impl PointDetector for PointDetectPeak {
    fn detect(
        &self,
        img: &Frame,
        _median: Option<&Frame>,
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

        let img_w = img.width() as i32;
        let img_h = img.height() as i32;

        let img_raw: &[u8] = img.as_raw();
        let mask_raw = mask.map(|mask| mask.as_raw());

        let mut x: i32 = 0;
        let mut y: i32 = 0;
        let mut idx: usize = 0;

        while y < img_h {
            while x < img_w {
                if let Some(mask_raw) = mask_raw {
                    if mask_raw[idx] == 0 {
                        x += STEP_X;
                        continue;
                    }
                }

                let val: u8 = img_raw[idx];

                if val > POINT_THRESHOLD {
                    let mut curr_val: u8 = val;
                    let mut point_idx = idx;
                    let mut point_x = x;
                    while (point_idx as i32 + 1) < img_w && img_raw[point_idx + 1] > curr_val {
                        point_idx += 1;
                        point_x += 1;
                        curr_val = img_raw[point_idx];
                    }

                    let mut point_y = y;
                    let mut point_idx = idx;
                    while point_y + 1 < img_h && img_raw[idx + img_w as usize] > curr_val {
                        point_y += 1;
                        point_idx += img_w as usize;
                        curr_val = img_raw[point_idx];
                    }

                    // find existing matches within POINT_EXCLUSION_RADIUS of current match
                    let mut matching: Vec<_> = points
                        .iter()
                        .enumerate()
                        .filter(|&(_idx, existing)| {
                            let xd = (existing.x as i32 - point_x).abs();
                            let yd = (existing.y as i32 - point_y).abs();
                            ((xd * xd) + (yd * yd)) < POINT_EXCLUSION_RADIUS_2
                        })
                        .map(|(n, i)| (n, i.clone()))
                        .collect();

                    matching.sort_by_key(|(_, p)| p.amplitude);

                    // if one of these nearby existing matches has higher amplitude, skip
                    // the current point
                    if let Some((_, last)) = matching.last() {
                        if last.amplitude > curr_val {
                            x += STEP_X;
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
                        x: point_x as u32,
                        y: point_y as u32,
                        amplitude: curr_val,
                        fitted: None,
                    };

                    points.push(new_point.clone());
                }

                x += STEP_X;
                idx += STEP_X as usize;
            }
            x = 0;
            y += STEP_Y;
        }

        points
    }
}

#[allow(dead_code)]
pub fn inside_existing_point(x: i32, y: i32, points: &[DetectedPoint]) -> bool {
    points.iter().any(|point| {
        let xd = (point.x as i32 - x).abs();
        let yd = (point.y as i32 - y).abs();
        ((xd * xd) + (yd * yd)) < POINT_EXCLUSION_RADIUS_2
    })
}

#[cfg(test)]
mod tests {
    use image::io::Reader as ImageReader;
    use image::{GrayImage, ImageBuffer, Luma};
    use imageproc::filter::median_filter;
    use imageproc::map::map_colors2;
    use kiddo::float::kdtree::KdTree;
    use kiddo::SquaredEuclidean;
    use std::collections::HashSet;
    use std::fs::File;
    use std::sync::Arc;

    use super::PointDetectPeak;
    use crate::PointDetector;

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

        let arc_sub = Arc::new(subtracted);

        // perform the extract
        let results: Vec<DetectedPoint> = PointDetectPeak::detect(arc_sub, Some(img_median), None);

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
