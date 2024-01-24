use std::sync::Arc;

use argmin::core::ArgminFloat;
use az::{Az, Cast};
use kiddo::float::kdtree::Axis;
use ordered_float::OrderedFloat;

use crate::state::DetectedPoint;
use crate::traits::{ImageLumaExtractor, PointDetector};

// const POINT_SKIP_STEP: u32 = 3;
const POINT_EXCLUSION_RADIUS_2: f64 = 400.0;
const POINT_THRESHOLD: u8 = 43;
const STEP_X: u32 = 1;
const STEP_Y: u32 = 1;
pub const PATCH_SIZE: u32 = 20;

pub struct PointDetectPeak {}

impl<F: Axis + ArgminFloat> PointDetector<F> for PointDetectPeak
where
    u32: Cast<F>,
    u8: Cast<F>,
    F: Cast<u32>,
{
    #[inline]
    fn detect(
        img: Arc<dyn ImageLumaExtractor>,
        mask: Option<Arc<dyn ImageLumaExtractor>>,
    ) -> Vec<DetectedPoint<F>> {
        let mut points: Vec<DetectedPoint<F>> = vec![];

        let img_w = img.width();
        let img_h = img.height();

        let mut x: u32 = 0;
        let mut y: u32 = 0;

        while y < img_h {
            while x < img_w {
                if let Some(ref mask) = mask {
                    if mask.get_luma8_for_pixel(x, y) == 0 {
                        x += STEP_X;
                        continue;
                    }
                }

                let val: u8 = img.get_luma8_for_pixel(x, y);

                if val > POINT_THRESHOLD {
                    let mut curr_val: u8 = val;
                    let mut point_x = x;
                    while point_x + 1 < img_w && img.get_luma8_for_pixel(point_x + 1, y) > curr_val
                    {
                        point_x += 1;
                        curr_val = img.get_luma8_for_pixel(point_x, y);
                    }

                    let mut point_y = y;
                    while point_y + 1 < img_h
                        && img.get_luma8_for_pixel(point_x, point_y + 1) > curr_val
                    {
                        point_y += 1;
                        curr_val = img.get_luma8_for_pixel(point_x, point_y);
                    }

                    // find existing matches within POINT_EXCLUSION_RADIUS of current match
                    let mut matching: Vec<_> = points
                        .iter()
                        .enumerate()
                        .filter(|&(_idx, existing)| {
                            let xd = existing.x.az::<f64>() - (point_x as f64).abs();
                            let yd = (existing.y.az::<f64>() - (point_y as f64)).abs();
                            ((xd * xd) + (yd * yd)) < POINT_EXCLUSION_RADIUS_2
                        })
                        .map(|(n, i)| (n, i.clone()))
                        .collect();

                    matching.sort_by_key(|(_, p)| OrderedFloat(p.amplitude));

                    // if one of these nearby existing matches has higher amplitude, skip
                    // the current point
                    if let Some((_, last)) = matching.last() {
                        if last.amplitude.az::<u32>() > curr_val as u32 {
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
                        x: point_x,
                        y: point_y,
                        amplitude: curr_val.az::<F>(),
                        fitted_point: None,
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

pub fn inside_existing_point<F: Axis + ArgminFloat>(
    x: u32,
    y: u32,
    points: &[DetectedPoint<F>],
) -> bool {
    points.iter().any(|point| {
        let xd = point.x.az::<f64>() - (x as f64).abs();
        let yd = (point.y.az::<f64>() - (y as f64)).abs();
        ((xd * xd) + (yd * yd)) < POINT_EXCLUSION_RADIUS_2
    })
}

#[cfg(test)]
mod tests {
    use crate::state::DetectedPoint;
    use image::io::Reader as ImageReader;
    use image::{GrayImage, ImageBuffer, Luma};
    use imageproc::filter::median_filter;
    use imageproc::map::map_colors2;
    use kiddo::float::kdtree::KdTree;
    use kiddo::SquaredEuclidean;
    use std::collections::HashSet;
    use std::fs::File;
    use std::sync::Arc;

    use crate::point_detect_peak::PointDetectPeak;
    use crate::traits::PointDetector;

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
        // let detector = PointDetectPeak {};
        let results: Vec<DetectedPoint<f64>> = PointDetectPeak::detect(arc_sub, None);

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
