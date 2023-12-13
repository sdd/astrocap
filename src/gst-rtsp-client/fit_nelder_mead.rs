use argmin::core::{ArgminOp, Error, Executor, TerminationReason};
use argmin::solver::neldermead::NelderMead;

use image::GrayImage;
use lazy_static::lazy_static;

use solvastro::query::QueryPoint;

use tracing::debug;

use crate::point_extractor_consumer::ImagePointCandidate;

pub trait PointFitter {
    fn fit_point(&self, point: &ImagePointCandidate, img: &GrayImage) -> QueryPoint;
}

const INITIAL_GAUSSIAN_ALPHA: f64 = 3.5f64;
const MAX_ITERATIONS: u64 = 100;
const SD_TOLERANCE: f64 = 1.0;

const PATCH_SIZE: u32 = 20;

lazy_static! {
    static ref PERTURBATIONS: Vec<f64> = vec![-4.0, -4.0, 1.0, 1.0, 1.0];
}

pub struct PointFitterGaussianNelderMead {}

impl PointFitter for PointFitterGaussianNelderMead {
    fn fit_point(&self, point: &ImagePointCandidate, img: &GrayImage) -> ImagePointCandidate {
        let problem = Gaussian2DFitProblem {
            img,
            centre_x: point.x,
            centre_y: point.y,
        };

        let initial_params: Vec<f64> = vec![
            0f64,
            0f64,
            INITIAL_GAUSSIAN_ALPHA,
            INITIAL_GAUSSIAN_ALPHA,
            img.get_pixel(problem.centre_x as u32, problem.centre_y as u32)[0] as f64,
        ];

        let solver = NelderMead::new()
            .with_initial_params(create_simplex(&initial_params, &PERTURBATIONS))
            .sd_tolerance(SD_TOLERANCE);

        let result = Executor::new(problem, solver, initial_params)
            //.add_observer(ArgminSlogLogger::term(), ObserverMode::NewBest)
            .max_iters(MAX_ITERATIONS)
            .timer(false)
            .run()
            .unwrap();

        let response = QueryPoint {
            x: point.x + result.state.best_param[0],
            y: point.y + result.state.best_param[1],
            radius: (result.state.best_param[2] + result.state.best_param[3]) / 2.0,
            amplitude: result.state.best_param[4]
                * ((result.state.best_param[2] + result.state.best_param[3]) / 2.0),
        };

        if result.state.termination_reason != TerminationReason::TargetToleranceReached {
            debug!(
                "point fit unexpected result: {:#} (response: {:?})",
                &result, &response
            );
        }

        response
    }
}

pub fn create_simplex(point: &[f64], perturbations: &[f64]) -> Vec<Vec<f64>> {
    let simplex = perturbations
        .iter()
        .enumerate()
        .map(|(perturbation_idx, &perturbation)| {
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
                .collect()
        })
        .collect();

    //println!("Simplex: {:?}", &simplex);
    simplex
}

struct Gaussian2DFitProblem<'a> {
    img: &'a GrayImage,
    centre_x: f64,
    centre_y: f64,
}

impl ArgminOp for Gaussian2DFitProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, params: &Self::Param) -> Result<Self::Output, Error> {
        let mut residual: f64 = 0.0;

        for x in (self.centre_x as u32 - PATCH_SIZE).max(0)
            ..(self.centre_x as u32 + PATCH_SIZE).min(self.img.width())
        {
            for y in (self.centre_y as u32 - PATCH_SIZE).max(0)
                ..(self.centre_y as u32 + PATCH_SIZE).min(self.img.height())
            {
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

        Ok(residual)
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
