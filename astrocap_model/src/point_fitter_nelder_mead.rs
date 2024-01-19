use crate::state::{DetectedPoint, FittedPoint};
use crate::traits::{ImageLumaExtractor, PointFitter};
use argmin::core::{ArgminFloat, CostFunction, State, TerminationReason};
use argmin::solver::neldermead::NelderMead;
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::{Az, Cast};
use kiddo::float::kdtree::Axis;
use ndarray::{Array1, ArrayBase, Dim, OwnedRepr};
use num_traits::float::Float;
use std::iter::Sum;
use std::sync::Arc;
use tracing::{debug, warn};

const INITIAL_GAUSSIAN_ALPHA: f64 = 2.5;
const MAX_ITERATIONS: u64 = 100;
const SD_TOLERANCE: f64 = 1.0;
const PATCH_SIZE: u32 = 4;

const COST_FIT_COST_MULTIPLIER: f64 = 0.00002;
const COST_RADIUS_TARGET: f64 = 2.3;
const COST_RADIUS_MULTIPLIER: f64 = 10.0;

const COST_AMPLITUDE_TARGET: f64 = 68.0;
const COST_AMPLITUDE_MULTIPLIER: f64 = 0.001;

const COST_OFFSET: f64 = 9.50;

pub fn transform_cost<F: Axis + Float>(cost: F, radius_x: F, radius_y: F, amplitude: F) -> F
where
    f64: Cast<F>,
{
    (cost * COST_FIT_COST_MULTIPLIER.az::<F>())
        - Float::powi(COST_RADIUS_TARGET.az::<F>() - radius_x, 2) * COST_RADIUS_MULTIPLIER.az::<F>()
        - Float::powi(COST_RADIUS_TARGET.az::<F>() - radius_y, 2) * COST_RADIUS_MULTIPLIER.az::<F>()
        - (Float::powi(COST_AMPLITUDE_TARGET.az::<F>() - amplitude, 2)
            * COST_AMPLITUDE_MULTIPLIER.az::<F>())
        + COST_OFFSET.az::<F>()
}

pub struct PointFitterGaussianNelderMead {
    pub frame: Arc<dyn ImageLumaExtractor>,
}

impl<F: ArgminFloat + Axis + Sum> PointFitter<F> for PointFitterGaussianNelderMead
where
    u32: Cast<F>,
    u8: Cast<F>,
    f64: Cast<F>,
    F: Cast<u32>,
    F: Cast<i32>,
    ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>: ArgminAdd<
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
    >,
    ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>: ArgminSub<
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
        ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>,
    >,
    ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>:
        ArgminMul<F, ArrayBase<OwnedRepr<F>, Dim<[usize; 1]>>>,
{
    fn new(frame: Arc<dyn ImageLumaExtractor>) -> Self {
        PointFitterGaussianNelderMead { frame }
    }

    fn fit(&self, point: &DetectedPoint<F>) -> FittedPoint<F> {
        let img = self.frame.as_ref();

        let problem = Gaussian2DFitProblem {
            img,
            centre_x: point.x.max(0).min(img.width() - 1).az::<F>(),
            centre_y: point.y.max(0).min(img.height() - 1).az::<F>(),
        };

        let initial_params: Vec<F> = vec![
            F::zero(),
            F::zero(),
            INITIAL_GAUSSIAN_ALPHA.az::<F>(),
            INITIAL_GAUSSIAN_ALPHA.az::<F>(),
            img.get_luma8_for_pixel(problem.centre_x.az::<u32>(), problem.centre_y.az::<u32>())
                .az::<F>(),
        ];

        let perturbations: Vec<F> = vec![
            -4.0.az::<F>(),
            -4.0.az::<F>(),
            1.0.az::<F>(),
            1.0.az::<F>(),
            1.0.az::<F>(),
        ];

        let solver = NelderMead::new(create_simplex(&initial_params, &perturbations))
            //.with_initial_params(create_simplex(&initial_params, &PERTURBATIONS))
            .with_sd_tolerance(SD_TOLERANCE.az::<F>())
            .unwrap();

        let result = argmin::core::Executor::new(problem, solver)
            //.add_observer(ArgminSlogLogger::term(), ObserverMode::NewBest)
            .configure(|state| state.max_iters(100))
            .timer(false)
            .run()
            .unwrap();

        let best = result.state().get_best_param().unwrap();
        let cost = result.state().get_best_cost();

        let avg_radius = (best[2] + best[3]) / 2.0.az::<F>();

        let latest_score = transform_cost(cost, best[2], best[3], best[4]);

        let response = FittedPoint {
            x: point.x.az::<F>() + best[0],
            y: point.y.az::<F>() + best[1],
            radius: avg_radius,
            score: latest_score,
            amplitude: best[4],
        };
        if cost == F::zero() {
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

pub fn create_simplex<F: Axis + Float>(point: &[F], perturbations: &[F]) -> Vec<Array1<F>> {
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

struct Gaussian2DFitProblem<'a, F: Axis + Float> {
    img: &'a dyn ImageLumaExtractor,
    centre_x: F,
    centre_y: F,
}

impl<F: Axis + Float> CostFunction for Gaussian2DFitProblem<'_, F>
where
    u8: Cast<F>,
    u32: Cast<F>,
    F: Cast<i32>,
{
    type Param = Array1<F>;
    type Output = F;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut residual = F::zero();
        let mut within_image = false;

        let x_range = (self.centre_x.az::<i32>() - PATCH_SIZE.az::<i32>()).max(0) as u32
            ..(self.centre_x.az::<i32>() + PATCH_SIZE.az::<i32>())
                .min((self.img.width() - 1).az::<i32>()) as u32;

        let y_range = (self.centre_y.az::<i32>() - PATCH_SIZE.az::<i32>()).max(0) as u32
            ..(self.centre_y.az::<i32>() + PATCH_SIZE.az::<i32>())
                .min((self.img.height() - 1).az::<i32>()) as u32;

        for x in x_range.clone() {
            for y in y_range.clone() {
                within_image = true;
                let img_val = self.img.get_luma8_for_pixel(x, y).az::<F>();
                let model_val = gaussian_2d(
                    x.az::<F>(),
                    y.az::<F>(),
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
            Ok(<F as Float>::infinity())
        } else {
            Ok(residual)
        }
    }
}

pub fn gaussian_2d<F: Axis + Float>(
    x: F,
    y: F,
    x0: F,
    y0: F,
    x_alpha: F,
    y_alpha: F,
    amplitude: F,
) -> F {
    let x_part = (x - x0) / x_alpha;
    let y_part = (y - y0) / y_alpha;
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
    use std::sync::Arc;

    use crate::point_fitter_nelder_mead::PointFitterGaussianNelderMead;
    use crate::state::DetectedPoint;
    use crate::traits::{ImageLumaExtractor, PointFitter};

    struct Point {
        x: usize,
        y: usize,
        amp: u8,
    }

    type Tree<F> = KdTree<F, usize, 2, 32, u32>;

    const MAX_GOOD_STAR_MATCH_RADIUS: f64 = 10.0;

    #[test]
    fn can_fit_known_stars() {
        // Load a set of images with known good star positions
        let raw_img =
            ImageReader::open("../test-images/astrocap_model/test-image-1-subtracted.png")
                .unwrap()
                .decode()
                .unwrap();
        let img_height = raw_img.height();
        let img_width = raw_img.width();

        let img =
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(img_width, img_height, raw_img.into_bytes())
                .expect("Could not create ImageBuffer from VideoFrame");

        let points: Vec<DetectedPoint<f64>> = serde_json::from_reader(
            File::open("../test-images/astrocap_model/test-image-1-detected.json").unwrap(),
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

        let img_arc: Arc<dyn ImageLumaExtractor> = Arc::new(img);
        // let img_arc_clone = img_arc.clone();

        let fitter = PointFitterGaussianNelderMead { frame: img_arc };

        let mut results: Vec<_> = vec![];
        for point in points {
            results.push(fitter.fit(&point));
        }
        serde_json::to_writer(
            File::create("../test-images/astrocap_model/test-image-1-fitted.json").unwrap(),
            &results,
        )
        .unwrap();

        let mut tree: Tree<f64> = Tree::new();
        for (idx, point) in results.iter().enumerate() {
            tree.add(&[point.x, point.y], idx);
        }

        // assert that the known good stars are detected
        let mut matched_indexes: HashSet<usize> = HashSet::new();
        let mut matched_candidates: Vec<_> = Vec::new();
        for point in known_good.iter() {
            let nearest = tree.nearest_one::<SquaredEuclidean>(&[point.x as f64, point.y as f64]);
            assert!(nearest.distance < MAX_GOOD_STAR_MATCH_RADIUS);
            matched_indexes.insert(nearest.item);
            matched_candidates.push(&results[nearest.item]);
        }

        assert_eq!(matched_indexes.len(), known_good.len());
    }
}
