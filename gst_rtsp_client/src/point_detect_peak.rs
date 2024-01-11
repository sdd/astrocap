use crate::point_extractor_consumer::ImagePointCandidate;
use image::{GrayImage, ImageBuffer, Luma};
use ordered_float::OrderedFloat;
use std::sync::Arc;

// const POINT_SKIP_STEP: u32 = 3;
const POINT_EXCLUSION_RADIUS: f64 = 20.0;
const EXISTING_POINT_EXCLUSION_RADIUS_2: f64 = 100.0;
const POINT_EXCLUSION_RADIUS_2: f64 = 400.0;
const POINT_THRESHOLD: u8 = 43;
const STEP_X: u32 = 1;
const STEP_Y: u32 = 1;
pub const PATCH_SIZE: u32 = 20;

const INITIAL_RADIUS: f64 = 2.2;

pub trait ImageLumaExtractor {
    fn get_luma8_for_pixel(&self, x: u32, y: u32) -> u8;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}

pub trait PointDetector {
    fn extract_from_img(
        &self,
        img: &GrayImage,
        existing_points: &[ImagePointCandidate],
        mask: Option<Arc<ImageBuffer<Luma<u8>, Vec<u8>>>>,
    ) -> Vec<ImagePointCandidate>;
}

pub struct PointDetectPeak {}

impl PointDetector for PointDetectPeak {
    fn extract_from_img(
        &self,
        img: &GrayImage,
        existing_points: &[ImagePointCandidate],
        mask: Option<Arc<ImageBuffer<Luma<u8>, Vec<u8>>>>,
    ) -> Vec<ImagePointCandidate> {
        let mut points: Vec<ImagePointCandidate> = vec![];

        let (img_w, img_h) = img.dimensions();
        let mut x: u32 = 0;
        let mut y: u32 = 0;

        while y < img_h {
            while x < img_w {
                if let Some(ref mask) = mask {
                    if mask.get_pixel(x, y)[0] == 0 {
                        x += STEP_X;
                        continue;
                    }
                }

                let val: u8 = img.get_pixel(x, y)[0];

                if val > POINT_THRESHOLD {
                    let mut curr_val: u8 = val;
                    let mut point_x = x;
                    while point_x + 1 < img_w && img.get_pixel(point_x + 1, y)[0] > curr_val {
                        point_x += 1;
                        curr_val = img.get_pixel(point_x, y)[0];
                    }

                    let mut point_y = y;
                    while point_y + 1 < img_h && img.get_pixel(point_x, point_y + 1)[0] > curr_val {
                        point_y += 1;
                        curr_val = img.get_pixel(point_x, point_y)[0];
                    }

                    // find matches in pre-existing state within POINT_EXCLUSION_RADIUS of current match
                    let matching_existing = existing_points.iter().any(|&existing| {
                        let xd = existing.x - (point_x as f64).abs();
                        let yd = (existing.y - (point_y as f64)).abs();
                        ((xd * xd) + (yd * yd)) < EXISTING_POINT_EXCLUSION_RADIUS_2
                    });
                    if matching_existing {
                        x += STEP_X;
                        continue;
                    }

                    // find existing matches within POINT_EXCLUSION_RADIUS of current match
                    let mut matching: Vec<(usize, ImagePointCandidate)> = points
                        .iter()
                        .enumerate()
                        .filter(|&(_idx, existing)| {
                            let xd = existing.x - (point_x as f64).abs();
                            let yd = (existing.y - (point_y as f64)).abs();
                            ((xd * xd) + (yd * yd)) < POINT_EXCLUSION_RADIUS_2
                        })
                        .map(|(n, i)| (n, i.clone()))
                        .collect();

                    matching.sort_by_key(|(_, p)| OrderedFloat(p.amplitude));

                    // if one of these nearby existing matches has higher amplitude, skip
                    // the current point
                    if let Some((_, last)) = matching.last() {
                        if last.amplitude > curr_val as f64 {
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

                    let new_point = ImagePointCandidate {
                        x: point_x as f64,
                        y: point_y as f64,
                        amplitude: curr_val as f64,
                        radius: INITIAL_RADIUS,
                        log_likelihood: 0.0,
                        age: 0,
                        latest_score: -10e10,
                        matched_last_frame: false,
                    };

                    points.push(new_point.clone());
                }

                x += STEP_X;
            }
            x = 0;
            y += STEP_Y;
        }

        points
    }
}

pub fn inside_existing_point(x: u32, y: u32, points: &[ImagePointCandidate]) -> bool {
    points.iter().any(|point| {
        let xd = point.x - (x as f64).abs();
        let yd = (point.y - (y as f64)).abs();
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

    use crate::point_detect_peak::{PointDetectPeak, PointDetector};

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
        let raw_img = ImageReader::open("../test-images/gst_rtsp_client/test-image-1.png")
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
            .save("../test-images/gst_rtsp_client/test-image-1-subtracted.png")
            .unwrap();

        // perform the extract
        let detector = PointDetectPeak {};
        let results = detector.extract_from_img(&subtracted, &vec![], None);

        serde_json::to_writer(
            File::create("../test-images/gst_rtsp_client/test-image-1-detected.json").unwrap(),
            &results,
        )
        .unwrap();

        let mut tree: Tree = Tree::new();
        for (idx, point) in results.iter().enumerate() {
            tree.add(&[point.x, point.y], idx);
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
