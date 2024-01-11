use argmin::core::{CostFunction, State, TerminationReason};
use argmin::solver::neldermead::NelderMead;
use image::GrayImage;
use lazy_static::lazy_static;
use ndarray::Array1;
use tracing::{debug, warn};

use crate::point_extractor_consumer::ImagePointCandidate;

pub trait PointFitter {
    fn fit_point(&self, point: &ImagePointCandidate, img: &GrayImage) -> ImagePointCandidate;
}

const INITIAL_GAUSSIAN_ALPHA: f64 = 2.5;
const MAX_ITERATIONS: u64 = 100;
const SD_TOLERANCE: f64 = 1.0;
const PATCH_SIZE: u32 = 20;

lazy_static! {
    static ref PERTURBATIONS: Vec<f64> = vec![-4.0, -4.0, 1.0, 1.0, 1.0];
}

const COST_FIT_COST_MULTIPLIER: f64 = 0.00002f64;
const COST_RADIUS_TARGET: f64 = 2.3;
const COST_RADIUS_MULTIPLIER: f64 = 10.0;

const COST_AMPLITUDE_TARGET: f64 = 68.0;
const COST_AMPLITUDE_MULTIPLIER: f64 = 0.001;

const COST_OFFSET: f64 = 9.50;

pub fn transform_cost(cost: f64, radius_x: f64, radius_y: f64, amplitude: f64) -> f64 {
    (cost * COST_FIT_COST_MULTIPLIER)
        - ((COST_RADIUS_TARGET - radius_x).powi(2) * COST_RADIUS_MULTIPLIER)
        - ((COST_RADIUS_TARGET - radius_y).powi(2) * COST_RADIUS_MULTIPLIER)
        - ((COST_AMPLITUDE_TARGET - amplitude).powi(2) * COST_AMPLITUDE_MULTIPLIER)
        + COST_OFFSET
}

pub struct PointFitterGaussianNelderMead {}

impl PointFitter for PointFitterGaussianNelderMead {
    fn fit_point(&self, point: &ImagePointCandidate, img: &GrayImage) -> ImagePointCandidate {
        let problem = Gaussian2DFitProblem {
            img,
            centre_x: point.x.max(0.0).min((img.width() - 1) as f64),
            centre_y: point.y.max(0.0).min((img.height() - 1) as f64),
        };

        let initial_params: Vec<f64> = vec![
            0f64,
            0f64,
            INITIAL_GAUSSIAN_ALPHA,
            INITIAL_GAUSSIAN_ALPHA,
            img.get_pixel(problem.centre_x as u32, problem.centre_y as u32)[0] as f64,
        ];

        let solver = NelderMead::new(create_simplex(&initial_params, &PERTURBATIONS))
            //.with_initial_params(create_simplex(&initial_params, &PERTURBATIONS))
            .with_sd_tolerance(SD_TOLERANCE)
            .unwrap();

        let result = argmin::core::Executor::new(problem, solver)
            //.add_observer(ArgminSlogLogger::term(), ObserverMode::NewBest)
            .configure(|state| state.max_iters(100))
            .timer(false)
            .run()
            .unwrap();

        let best = result.state().get_best_param().unwrap();
        let cost = result.state().get_best_cost();

        let avg_radius = (best[2] + best[3]) / 2.0;

        let latest_score = transform_cost(cost, best[2], best[3], best[4]);

        let response = ImagePointCandidate {
            x: point.x + best[0],
            y: point.y + best[1],
            radius: avg_radius,
            log_likelihood: point.log_likelihood + latest_score,
            amplitude: best[4],
            age: point.age,
            latest_score,
            matched_last_frame: point.matched_last_frame,
        };
        if cost == 0.0 {
            warn!(?result.state, "Cost of zero")
        }

        if result.state().get_termination_reason().unwrap() != &TerminationReason::MaxItersReached {
            debug!(
                "point fit unexpected result: {:#} (response: {:?})",
                &result, &response
            );
        }

        response
    }
}

pub fn create_simplex(point: &[f64], perturbations: &[f64]) -> Vec<Array1<f64>> {
    let simplex = perturbations
        .iter()
        .enumerate()
        .map(|(perturbation_idx, &perturbation)| {
            Array1::from_vec(
                point
                    .iter()
                    .enumerate()
                    .map(|(coord_idx, &coord)| {
                        if coord_idx == perturbation_idx {
                            coord + perturbation
                        } else {
                            coord
                        }
                    })
                    .collect(),
            )
        })
        .collect();

    simplex
}

#[derive(Debug)]
struct Gaussian2DFitProblem<'a> {
    img: &'a GrayImage,
    centre_x: f64,
    centre_y: f64,
}

impl CostFunction for Gaussian2DFitProblem<'_> {
    type Param = Array1<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut residual: f64 = 0.0;
        let mut within_image = false;

        let x_range = (self.centre_x as i32 - PATCH_SIZE as i32).max(0) as u32
            ..(self.centre_x as i32 + PATCH_SIZE as i32).min((self.img.width() - 1) as i32) as u32;

        let y_range = (self.centre_y as i32 - PATCH_SIZE as i32).max(0) as u32
            ..(self.centre_y as i32 + PATCH_SIZE as i32).min((self.img.height() - 1) as i32) as u32;

        for x in x_range.clone() {
            for y in y_range.clone() {
                within_image = true;
                let img_val = self.img.get_pixel(x, y)[0] as f64;
                let model_val = gaussian_2d(
                    x as f64,
                    y as f64,
                    self.centre_x + params[0],
                    self.centre_y + params[1],
                    params[2],
                    params[3],
                    params[4],
                );

                let diff_sq = (model_val - img_val) * (model_val - img_val);
                residual += diff_sq;
            }
        }

        if !within_image {
            warn!(?self.centre_x, ?self.centre_y, ?x_range, ?y_range, ?params, "Not within image");
            Ok(f64::INFINITY)
        } else {
            Ok(residual)
        }
    }
}

pub fn gaussian_2d(
    x: f64,
    y: f64,
    x0: f64,
    y0: f64,
    x_alpha: f64,
    y_alpha: f64,
    amplitude: f64,
) -> f64 {
    let x_part: f64 = (x - x0) / x_alpha;
    let y_part: f64 = (y - y0) / y_alpha;
    amplitude * (-(x_part * x_part) - (y_part * y_part)).exp()
}

#[cfg(test)]
mod tests {
    use image::io::Reader as ImageReader;
    use image::{ImageBuffer, Luma};
    use kiddo::float::kdtree::KdTree;
    use kiddo::SquaredEuclidean;
    use std::collections::HashSet;
    use std::fs::File;

    use crate::fit_nelder_mead::{PointFitter, PointFitterGaussianNelderMead};
    use crate::point_extractor_consumer::ImagePointCandidate;

    struct Point {
        x: usize,
        y: usize,
        amp: u8,
    }

    type Tree = KdTree<f64, usize, 2, 32, u32>;

    const MAX_GOOD_STAR_MATCH_RADIUS: f64 = 10.0;

    #[test]
    fn can_fit_known_stars() {
        let fitter = PointFitterGaussianNelderMead {};

        // Load a set of images with known good star positions
        let raw_img =
            ImageReader::open("../test-images/gst_rtsp_client/test-image-1-subtracted.png")
                .unwrap()
                .decode()
                .unwrap();
        let img_height = raw_img.height();
        let img_width = raw_img.width();

        let img =
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(img_width, img_height, raw_img.into_bytes())
                .expect("Could not create ImageBuffer from VideoFrame");

        let points: Vec<ImagePointCandidate> = serde_json::from_reader(
            File::open("../test-images/gst_rtsp_client/test-image-1-detected.json").unwrap(),
        )
        .unwrap();

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

        let results: Vec<_> = points.iter().map(|p| fitter.fit_point(p, &img)).collect();
        serde_json::to_writer(
            File::create("../test-images/gst_rtsp_client/test-image-1-fitted.json").unwrap(),
            &results,
        )
        .unwrap();

        let mut tree: Tree = Tree::new();
        for (idx, point) in results.iter().enumerate() {
            tree.add(&[point.x, point.y], idx);
        }

        // assert that the known good stars are detected
        let mut matched_indexes: HashSet<usize> = HashSet::new();
        let mut matched_candidates: Vec<&ImagePointCandidate> = Vec::new();
        for point in known_good.iter() {
            let nearest = tree.nearest_one::<SquaredEuclidean>(&[point.x as f64, point.y as f64]);
            assert!(nearest.distance < MAX_GOOD_STAR_MATCH_RADIUS);
            matched_indexes.insert(nearest.item);
            matched_candidates.push(&results[nearest.item]);
        }

        assert_eq!(matched_indexes.len(), known_good.len());
    }
}
