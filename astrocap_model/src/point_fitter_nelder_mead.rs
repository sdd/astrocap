use crate::state::{DetectedPoint, FittedPoint};
use crate::traits::{AstroFloat, ImageLumaExtractor, PointFitter};
use argmin::core::{CostFunction, State, TerminationReason};
use argmin::solver::neldermead::NelderMead;
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::{Az, Cast};

use ndarray::{Array1, ArrayBase, Dim, OwnedRepr};
use num_traits::float::{Float, FloatCore};

use std::sync::Arc;
use tracing::{debug, warn};

use crate::state::point_handling::FittedPointQuality;
use serde::Serialize;

const INITIAL_GAUSSIAN_ALPHA: f64 = 1.0; // was 2.5;
const MAX_ITERATIONS: u64 = 100;
const SD_TOLERANCE: f64 = 1.0;
const PATCH_SIZE: u32 = 4;
const MIN_SIGMA: f64 = 0.8; // Prevent unrealistically narrow fits
const MAX_SIGMA: f64 = 8.0; // Prevent unrealistically wide fits
const MIN_AMPLITUDE: f64 = 5.0; // Minimum reasonable star amplitude

const COST_FIT_COST_MULTIPLIER: f64 = 0.00002;
const COST_RADIUS_TARGET: f64 = 2.3;
const COST_RADIUS_MULTIPLIER: f64 = 15.0;

const COST_AMPLITUDE_TARGET: f64 = 68.0;
const COST_AMPLITUDE_MULTIPLIER: f64 = 0.001;

const COST_OFFSET: f64 = 10.0;

const SCORE_FLOOR: f64 = -10.0;

pub fn calculate_fit_quality<F: AstroFloat>(
    img: &dyn ImageLumaExtractor,
    centre_x: F,
    centre_y: F,
    residual_sum_squares: F,
    degrees_of_freedom: u32,
    amplitude: F,
) -> FittedPointQuality<F>
where
    f64: Cast<F>,
    u8: Cast<F>,
    u32: Cast<F>,
    F: Cast<i32>,
{
    let dof = degrees_of_freedom.az::<F>();

    // Reduced chi-squared (should be ~1.0 for good fits)
    let reduced_chi_squared = residual_sum_squares / dof;

    // Calculate total variance of the patch
    let total_variance = calculate_patch_variance(img, centre_x, centre_y);

    // R-squared equivalent (fraction of variance explained)
    let r_squared = if total_variance > F::zero() {
        (total_variance - residual_sum_squares) / total_variance
    } else {
        F::zero()
    };

    // Estimate noise from residuals (RMS)
    let rms_residual = (residual_sum_squares / dof).sqrt();

    // Signal-to-noise ratio (amplitude vs noise)
    let snr = if rms_residual > F::zero() {
        amplitude / rms_residual
    } else {
        amplitude // If no noise, SNR is just the amplitude
    };

    FittedPointQuality {
        reduced_chi_squared,
        snr,
        r_squared,
        rms_residual,
    }
}

// Simplified star scoring focused on the most reliable metrics
pub fn calculate_star_score<F: AstroFloat>(fitted_point: &FittedPoint<F>) -> F
where
    f64: Cast<F>,
{
    let mut score = F::zero();

    // Primary score: amplitude-based with logarithmic scaling
    // Amplitude of 100+ should give ~6-8 points, amplitude of 50 should give ~4-5 points
    let amplitude_score = if fitted_point.amplitude > 10.0f64.az::<F>() {
        fitted_point.amplitude.ln() * 1.8f64.az::<F>()
    } else {
        // Very low amplitude gets negative score
        -3.0f64.az::<F>()
    };
    score += amplitude_score;

    // Strong radius penalties for unrealistic star shapes
    let avg_radius = (fitted_point.radius_x + fitted_point.radius_y) / 2.0f64.az::<F>();

    let radius_penalty = if avg_radius < 0.7f64.az::<F>() {
        // Too small - probably noise or hot pixels - heavy penalty
        -8.0f64.az::<F>()
    } else if avg_radius > 3.0f64.az::<F>() {
        // Too large - probably not a star - heavy penalty
        -6.0f64.az::<F>()
    } else if avg_radius < 1.0f64.az::<F>() || avg_radius > 2.5f64.az::<F>() {
        // Somewhat suspicious radius - moderate penalty
        -2.0f64.az::<F>()
    } else {
        // Good radius range (1.0 - 2.5) - small bonus
        1.0f64.az::<F>()
    };
    score += radius_penalty;

    // Strong penalty for extremely asymmetric shapes (elongated artifacts)
    let radius_ratio = if fitted_point.radius_x > fitted_point.radius_y {
        fitted_point.radius_x / fitted_point.radius_y
    } else {
        fitted_point.radius_y / fitted_point.radius_x
    };

    let asymmetry_penalty = if radius_ratio > 2.5f64.az::<F>() {
        // Very elongated - definitely not a star
        -5.0f64.az::<F>()
    } else if radius_ratio > 1.8f64.az::<F>() {
        // Quite elongated - probably not a star
        -2.0f64.az::<F>()
    } else if radius_ratio > 1.4f64.az::<F>() {
        // Somewhat elongated - small penalty
        -0.5f64.az::<F>()
    } else {
        // Round-ish - small bonus
        0.5f64.az::<F>()
    };
    score += asymmetry_penalty;

    // Light SNR contribution (much less emphasis than before)
    let snr_bonus = if fitted_point.fit_quality.snr > 3.0f64.az::<F>() {
        fitted_point.fit_quality.snr.ln() * 0.5f64.az::<F>()
    } else {
        F::zero()
    };
    score += snr_bonus;

    // Cap the score at our desired bounds
    score = FloatCore::min(FloatCore::max(score, -10.0f64.az::<F>()), 10.0f64.az::<F>());

    score
}

pub struct PointFitterGaussianNelderMead {
    pub frame: Arc<dyn ImageLumaExtractor>,
}

// TODO: refactor to Levenberg-Marquardt instead of Nelder-Mead
// https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.LevenbergMarquardt.html

impl<F: AstroFloat> PointFitter<F> for PointFitterGaussianNelderMead
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

    #[inline]
    fn fit(&self, point: &DetectedPoint<F>) -> FittedPoint<F> {
        let img = self.frame.as_ref();

        let centre_x = point.x.max(0).min(img.width() - 1).az::<F>();
        let centre_y = point.y.max(0).min(img.height() - 1).az::<F>();

        let problem = Gaussian2DFitProblem {
            img,
            centre_x,
            centre_y,
        };

        let (initial_params, perturbations) = better_initial_params(img, centre_x, centre_y);

        let solver = NelderMead::new(create_simplex(&initial_params, &perturbations))
            .with_sd_tolerance(SD_TOLERANCE.az::<F>())
            .unwrap();

        let result = argmin::core::Executor::new(problem, solver)
            .configure(|state| state.max_iters(MAX_ITERATIONS))
            .timer(false)
            .run()
            .unwrap();

        let best = result.state().get_best_param().unwrap();
        let cost = result.state().get_best_cost();

        // Calculate degrees of freedom: pixels in patch minus number of fitted parameters
        let patch_size = (2 * PATCH_SIZE + 1) * (2 * PATCH_SIZE + 1);
        let num_parameters = 5; // x_offset, y_offset, sigma_x, sigma_y, amplitude
        let degrees_of_freedom = patch_size.saturating_sub(num_parameters);

        // Calculate fit quality
        let fit_quality =
            calculate_fit_quality(img, centre_x, centre_y, cost, degrees_of_freedom, best[4]);

        let mut response = FittedPoint {
            x: point.x.az::<F>() + best[0],
            y: point.y.az::<F>() + best[1],
            radius_x: best[2],
            radius_y: best[3],
            amplitude: best[4],
            score: F::zero(),
            fit_quality,
        };

        // Calculate score using new quality-based approach
        let score = calculate_star_score(&response);
        response.score = score;

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

fn create_simplex<F: AstroFloat>(point: &[F], perturbations: &[F]) -> Vec<Array1<F>> {
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

struct Gaussian2DFitProblem<'a, F: AstroFloat> {
    img: &'a dyn ImageLumaExtractor,
    centre_x: F,
    centre_y: F,
}

impl<F: AstroFloat> CostFunction for Gaussian2DFitProblem<'_, F>
where
    u8: Cast<F>,
    u32: Cast<F>,
    F: Cast<i32>,
    f64: Cast<F>,
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

        // Constrain parameters to physical ranges
        let sigma_x = FloatCore::min(
            FloatCore::max(params[2], MIN_SIGMA.az::<F>()),
            MAX_SIGMA.az::<F>(),
        );
        let sigma_y = FloatCore::min(
            FloatCore::max(params[3], MIN_SIGMA.az::<F>()),
            MAX_SIGMA.az::<F>(),
        );
        let amplitude = FloatCore::max(params[4], MIN_AMPLITUDE.az::<F>());

        for x in x_range.clone() {
            for y in y_range.clone() {
                within_image = true;
                let img_val = self.img.get_luma8_for_pixel(x, y).az::<F>();
                let model_val = gaussian_2d(
                    x.az::<F>(),
                    y.az::<F>(),
                    self.centre_x + params[0],
                    self.centre_y + params[1],
                    sigma_x,
                    sigma_y,
                    amplitude,
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

pub fn gaussian_2d<F: AstroFloat>(
    x: F,
    y: F,
    x0: F,
    y0: F,
    x_alpha: F,
    y_alpha: F,
    amplitude: F,
) -> F
where
    f64: Cast<F>,
{
    let x_part = (x - x0) / x_alpha;
    let y_part = (y - y0) / y_alpha;
    amplitude * (-(x_part * x_part + y_part * y_part) / 2.0f64.az::<F>()).exp()
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

pub fn is_good_stellar_fit<F: AstroFloat>(
    reduced_chi_squared: F,
    sigma_x: F,
    sigma_y: F,
    amplitude: F,
    converged: bool,
) -> bool
where
    f64: Cast<F>,
{
    converged
        && reduced_chi_squared < 3.0.az::<F>()
        && sigma_x > 0.5.az::<F>()
        && sigma_x < 3.0.az::<F>()
        && sigma_y > 0.5.az::<F>()
        && sigma_y < 3.0.az::<F>()
        && amplitude > 5.0.az::<F>()
}

// Move calculate_patch_variance to be a standalone function
fn calculate_patch_variance<F: AstroFloat>(
    img: &dyn ImageLumaExtractor,
    centre_x: F,
    centre_y: F,
) -> F
where
    u8: Cast<F>,
    u32: Cast<F>,
    F: Cast<i32>,
{
    let mut sum = F::zero();
    let mut sum_squared = F::zero();
    let mut count = F::zero();

    let x_range = (centre_x.az::<i32>() - PATCH_SIZE.az::<i32>()).max(0) as u32
        ..(centre_x.az::<i32>() + PATCH_SIZE.az::<i32>()).min((img.width() - 1).az::<i32>()) as u32;

    let y_range = (centre_y.az::<i32>() - PATCH_SIZE.az::<i32>()).max(0) as u32
        ..(centre_y.az::<i32>() + PATCH_SIZE.az::<i32>()).min((img.height() - 1).az::<i32>())
            as u32;

    // Calculate mean and variance of patch pixels
    for x in x_range.clone() {
        for y in y_range.clone() {
            let pixel_val = img.get_luma8_for_pixel(x, y).az::<F>();
            sum += pixel_val;
            sum_squared += pixel_val * pixel_val;
            count += F::one();
        }
    }

    if count > F::zero() {
        let mean = sum / count;
        let variance = (sum_squared / count) - (mean * mean);
        variance
    } else {
        F::zero()
    }
}

fn better_initial_params<F: AstroFloat>(
    img: &dyn ImageLumaExtractor,
    centre_x: F,
    centre_y: F,
) -> (Vec<F>, Vec<F>)
where
    u32: Cast<F>,
    F: Cast<u32>,
    u8: Cast<F>,
    f64: Cast<F>,
{
    // Estimate seeing from image characteristics or use typical value
    let typical_seeing_fwhm = 2.0f64; // pixels
    let initial_sigma = typical_seeing_fwhm / 2.355f64;

    // Better amplitude estimate: max in 3x3 region
    let mut max_amplitude = 0u8;
    let cx = centre_x.az::<u32>();
    let cy = centre_y.az::<u32>();

    for dy in -1..=1 {
        for dx in -1..=1 {
            let x = (cx as i32 + dx) as u32;
            let y = (cy as i32 + dy) as u32;
            if x < img.width() && y < img.height() {
                max_amplitude = max_amplitude.max(img.get_luma8_for_pixel(x, y));
            }
        }
    }

    let amplitude_estimate = max_amplitude.az::<F>();

    let initial_params = vec![
        F::zero(),               // x_offset
        F::zero(),               // y_offset
        initial_sigma.az::<F>(), // sigma_x
        initial_sigma.az::<F>(), // sigma_y
        amplitude_estimate,      // amplitude
    ];

    let perturbations = vec![
        0.5f64.az::<F>(),                      // x_offset (sub-pixel)
        0.5f64.az::<F>(),                      // y_offset
        (initial_sigma * 0.3f64).az::<F>(),    // sigma_x (30%)
        (initial_sigma * 0.3f64).az::<F>(),    // sigma_y
        amplitude_estimate * 0.2f64.az::<F>(), // amplitude (20%)
    ];

    (initial_params, perturbations)
}

fn clamp_params<F: AstroFloat>(params: &[F]) -> Vec<F>
where
    f64: Cast<F>,
    F: Cast<i32>,
{
    vec![
        // Clamp x and y offsets to reasonable sub-pixel values
        FloatCore::max(
            FloatCore::min(params[0], 2.0f64.az::<F>()),
            (-2.0f64).az::<F>(),
        ),
        FloatCore::max(
            FloatCore::min(params[1], 2.0f64.az::<F>()),
            (-2.0f64).az::<F>(),
        ),
        // Clamp sigma values to reasonable ranges
        FloatCore::max(
            FloatCore::min(params[2], MAX_SIGMA.az::<F>()),
            MIN_SIGMA.az::<F>(),
        ),
        FloatCore::max(
            FloatCore::min(params[3], MAX_SIGMA.az::<F>()),
            MIN_SIGMA.az::<F>(),
        ),
        // Amplitude should be positive
        FloatCore::max(params[4], F::zero()),
    ]
}
