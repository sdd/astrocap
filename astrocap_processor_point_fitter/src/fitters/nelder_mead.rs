use argmin::core::{CostFunction, State, TerminationReason};
use argmin::solver::neldermead::NelderMead;
use astrocap_core::frame::CpuFrame;
use astrocap_core::structs::{DetectedPoint, FittedPoint, FittedPointQuality};
use astrocap_core::traits::PointFitter;
use image::Pixel;
use ndarray::Array1;
use tracing::{debug, warn};

const INITIAL_GAUSSIAN_ALPHA: f32 = 1.0; // was 2.5;
const MAX_ITERATIONS: u64 = 20;
const SD_TOLERANCE: f32 = 1.0;
const PATCH_SIZE: u32 = 4;
const MIN_SIGMA: f32 = 0.8; // Prevent unrealistically narrow fits
const MAX_SIGMA: f32 = 8.0; // Prevent unrealistically wide fits
const MIN_AMPLITUDE: f32 = 5.0; // Minimum reasonable star amplitude

const COST_FIT_COST_MULTIPLIER: f32 = 0.00002;
const COST_RADIUS_TARGET: f32 = 2.3;
const COST_RADIUS_MULTIPLIER: f32 = 15.0;

const COST_AMPLITUDE_TARGET: f32 = 68.0;
const COST_AMPLITUDE_MULTIPLIER: f32 = 0.001;
const COST_OFFSET: f32 = 10.0;

const SCORE_FLOOR: f32 = -10.0;

pub fn calculate_fit_quality(
    frame: &CpuFrame,
    centre_x: f32,
    centre_y: f32,
    residual_sum_squares: f32,
    degrees_of_freedom: u32,
    amplitude: f32,
) -> FittedPointQuality {
    let dof = degrees_of_freedom;

    // Reduced chi-squared (should be ~1.0 for good fits)
    let reduced_chi_squared = residual_sum_squares / (dof as f32);

    // Calculate total variance of the patch
    let total_variance = calculate_patch_variance(frame, centre_x, centre_y);

    // R-squared equivalent (fraction of variance explained)
    let r_squared = if total_variance > 0f32 {
        (total_variance - residual_sum_squares) / total_variance
    } else {
        0f32
    };

    // Estimate noise from residuals (RMS)
    let rms_residual = (residual_sum_squares / (dof as f32)).sqrt();

    // Signal-to-noise ratio (amplitude vs noise)
    let snr = if rms_residual > 0f32 {
        amplitude / rms_residual
    } else {
        amplitude // If no noise, SNR is just the amplitude
    };

    FittedPointQuality {
        reduced_chi_squared,
        snr,
        r_squared,
        rms_residual,
        score: 0f32,
    }
}

// Smooth, continuous star scoring using linear/polynomial relationships
pub fn calculate_star_score(fitted_point: &FittedPoint) -> f32 {
    // 1. AMPLITUDE COMPONENT - Most discriminative (your key insight)
    // Real stars ≥34, spurious <34, with strong preference for higher amplitudes
    let amplitude_score = if fitted_point.amplitude >= 100.0f32 {
        // Very bright stars - asymptotic approach to max score
        30.0f32 + 10.0f32 * (1.0f32 - (-0.02f32 * (fitted_point.amplitude - 100.0f32)).exp())
    } else if fitted_point.amplitude >= 34.0f32 {
        // Above threshold - linear increase from 15 to 30
        linear_score(fitted_point.amplitude, 34.0f32, 100.0f32, 15.0f32, 30.0f32)
    } else {
        // Below threshold - exponential decay penalty
        let penalty_factor = (fitted_point.amplitude - 34.0f32) / 10.0f32;
        15.0f32 * penalty_factor.exp() - 25.0f32
    };

    // 2. R-SQUARED COMPONENT - Smooth transition based on fit quality
    // Your data: real stars -16 to -49, spurious -50 to -148
    let r_squared_score = sigmoid_score(
        fitted_point.fit_quality.r_squared,
        -50.0f32, // Midpoint between good and bad
        0.1f32,   // Steepness
        -15.0f32, // Score for very poor fits
        12.0f32,  // Score for good fits
    );

    // 3. SNR COMPONENT - Logarithmic scaling for SNR
    // Your data shows real stars tend to have higher SNR
    let snr_score = if fitted_point.fit_quality.snr > 0.1f32 {
        // Logarithmic scaling: log(SNR) gives smooth increase
        let log_snr = fitted_point.fit_quality.snr.ln();
        linear_score(
            log_snr, 0.69f32, // ln(2) - low SNR threshold
            2.3f32,  // ln(10) - high SNR threshold
            -5.0f32, // Penalty for low SNR
            10.0f32, // Bonus for high SNR
        )
    } else {
        -10.0f32 // Very low SNR penalty
    };

    // 4. RADIUS COMPONENT - Quadratic penalty for deviations from ideal
    let radius_x_abs = fitted_point.radius_x.abs();
    let radius_y_abs = fitted_point.radius_y.abs();
    let avg_radius = (radius_x_abs + radius_y_abs) / 2.0f32;

    // Ideal radius around 1.2 pixels (from your data), quadratic penalty for deviations
    let ideal_radius = 1.2f32;
    let radius_deviation = (avg_radius - ideal_radius).abs();
    let radius_score = if !(0.3f32..=3.0f32).contains(&avg_radius) {
        // Hard limits for unreasonable radii
        -20.0f32
    } else {
        // Quadratic penalty: score = max_score - k * deviation²
        let max_radius_score = 8.0f32;
        let penalty_factor = 3.0f32; // Tunable steepness
        max_radius_score - penalty_factor * radius_deviation * radius_deviation
    };

    /*// 5. SYMMETRY COMPONENT - Quadratic penalty for asymmetry
    let radius_ratio = if radius_x_abs > radius_y_abs {
        radius_x_abs / radius_y_abs.max(0.1f32)
    } else {
        radius_y_abs / radius_x_abs.max(0.1f32)
    };

    // Ideal ratio is 1.0 (perfect circle), quadratic penalty for deviations
    let symmetry_deviation = radius_ratio - 1.0f32;

    let symmetry_score = if radius_ratio > 4.0f32 {
        // Hard limit for extreme asymmetry
        -25.0f32
    } else {
        // Quadratic penalty
        let max_symmetry_score = 6.0f32;
        let penalty_factor = 4.0f32;
        max_symmetry_score - penalty_factor * symmetry_deviation * symmetry_deviation
    };*/

    /*// 6. NEGATIVE RADIUS PENALTY - Smooth penalty for fitting artifacts
    let negative_radius_penalty = if fitted_point.radius_x < 0f32 || fitted_point.radius_y < 0f32 {
        // Mild penalty - your data shows this can happen with real stars
        let neg_x_penalty = if fitted_point.radius_x < 0f32 { fitted_point.radius_x.abs() } else { 0f32 };
        let neg_y_penalty = if fitted_point.radius_y < 0f32 { fitted_point.radius_y.abs() } else { 0f32 };
        -(neg_x_penalty + neg_y_penalty) * 2.0f32
    } else {
        0f32
    };*/

    /*// 7. REDUCED CHI-SQUARED - Gentle sigmoid around ideal value of 1.0
    let chi2_score = if fitted_point.fit_quality.reduced_chi_squared > 0.0f32 {
        sigmoid_score(
            fitted_point.fit_quality.reduced_chi_squared.ln(), // Log scale for wide range
            0.0f32,   // ln(1) = 0, ideal chi-squared
            2.0f32,   // Moderate steepness
            -8.0f32,  // Penalty for very poor fits
            3.0f32    // Small bonus for good fits
        )
    } else {
        -5.0f32 // Invalid chi-squared
    };*/

    // Combine all components

    //+ symmetry_score + negative_radius_penalty + chi2_score;

    // Expected score ranges with this system:
    // Excellent real stars (amp>50, good metrics): ~45-65
    // Good real stars (amp 34-50, decent metrics): ~25-45
    // Marginal candidates (amp ~30-34): ~0-20
    // Spurious detections: negative to ~10

    amplitude_score + r_squared_score + snr_score + radius_score
}

pub struct PointFitterGaussianNelderMead {}

// TODO: refactor to Levenberg-Marquardt instead of Nelder-Mead
// https://docs.rs/levenberg-marquardt/latest/levenberg_marquardt/struct.LevenbergMarquardt.html

impl PointFitter for PointFitterGaussianNelderMead {
    fn fit(&self, frame: &CpuFrame, point: &DetectedPoint) -> FittedPoint {
        let centre_x = point.x.min((frame.img.width() - 1) as f32);
        let centre_y = point.y.min((frame.img.height() - 1) as f32);

        let problem = Gaussian2DFitProblem {
            frame,
            centre_x,
            centre_y,
        };

        let (initial_params, perturbations) = better_initial_params(frame, centre_x, centre_y);

        let solver = NelderMead::new(create_simplex(&initial_params, &perturbations))
            .with_sd_tolerance(SD_TOLERANCE)
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
            calculate_fit_quality(frame, centre_x, centre_y, cost, degrees_of_freedom, best[4]);

        let mut response = FittedPoint {
            x: point.x as f32 + best[0],
            y: point.y as f32 + best[1],
            radius_x: best[2],
            radius_y: best[3],
            amplitude: best[4],
            score: 0f32,
            fit_quality,
        };

        // Calculate score using new quality-based approach
        let score = calculate_star_score(&response);
        response.score = score;

        if cost == 0f32 {
            warn!(?result.state, "Cost of zero")
        }

        if !matches!(
            result.state().get_termination_reason().unwrap(),
            &TerminationReason::MaxItersReached
        ) {
            debug!(
                "point fit unexpected result: {:#} (response: {:?})",
                &result, &response
            );
        }

        response
    }
}

fn create_simplex(point: &[f32], perturbations: &[f32]) -> Vec<Array1<f32>> {
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

struct Gaussian2DFitProblem<'a> {
    frame: &'a CpuFrame,
    centre_x: f32,
    centre_y: f32,
}

impl CostFunction for Gaussian2DFitProblem<'_> {
    type Param = Array1<f32>;
    type Output = f32;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let mut residual = 0f32;
        let mut within_image = false;

        let x_range = (self.centre_x as i32 - PATCH_SIZE as i32).max(0) as u32
            ..(self.centre_x as i32 + PATCH_SIZE as i32).min((self.frame.img.width() - 1) as i32)
                as u32;

        let y_range = (self.centre_y as i32 - PATCH_SIZE as i32).max(0) as u32
            ..(self.centre_y as i32 + PATCH_SIZE as i32).min((self.frame.img.height() - 1) as i32)
                as u32;

        // Constrain parameters to physical ranges
        let sigma_x = params[2].max(MIN_SIGMA).min(MAX_SIGMA);
        let sigma_y = params[3].max(MIN_SIGMA).min(MAX_SIGMA);
        let amplitude = params[4].max(MIN_AMPLITUDE);

        for x in x_range.clone() {
            for y in y_range.clone() {
                within_image = true;
                let img_val = self.frame.img.get_pixel(x, y).channels()[0] as f32;
                let model_val = gaussian_2d(
                    x as f32,
                    y as f32,
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
            Ok(f32::INFINITY)
        } else {
            Ok(residual)
        }
    }
}

pub fn gaussian_2d(
    x: f32,
    y: f32,
    x0: f32,
    y0: f32,
    x_alpha: f32,
    y_alpha: f32,
    amplitude: f32,
) -> f32 {
    let x_part = (x - x0) / x_alpha;
    let y_part = (y - y0) / y_alpha;
    amplitude * (-(x_part * x_part + y_part * y_part) / 2.0f32).exp()
}

#[cfg(test)]
mod tests {
    use super::PointFitterGaussianNelderMead;
    use astrocap_core::frame::{CpuFrame, CpuStorage};
    use astrocap_core::structs::DetectedPoint;
    use astrocap_core::traits::PointFitter;
    use image::io::Reader as ImageReader;
    use image::{ImageBuffer, Luma};
    use kiddo::float::kdtree::KdTree;
    use kiddo::SquaredEuclidean;
    use std::collections::HashSet;
    use std::fs::File;
    use std::sync::Arc;

    struct Point {
        x: usize,
        y: usize,
        amp: u8,
    }

    type Tree = KdTree<f32, usize, 2, 32, u32>;

    const MAX_GOOD_STAR_MATCH_RADIUS: f32 = 10.0;

    #[ignore]
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

        let img = CpuFrame::from_vec(img_width, img_height, raw_img.into_bytes()).unwrap();

        let points: Vec<DetectedPoint> = serde_json::from_reader(
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

        let fitter = PointFitterGaussianNelderMead {};

        let mut results: Vec<_> = vec![];
        for point in points {
            results.push(fitter.fit(&img, &point));
        }
        serde_json::to_writer(
            File::create("../test-images/astrocap_model/test-image-1-fitted.json").unwrap(),
            &results,
        )
        .unwrap();

        let mut tree: Tree = Tree::new();
        for (idx, point) in results.iter().enumerate() {
            tree.add(&[point.x, point.y], idx);
        }

        // assert that the known good stars are detected
        let mut matched_indexes: HashSet<usize> = HashSet::new();
        let mut matched_candidates: Vec<_> = Vec::new();
        for point in known_good.iter() {
            let nearest = tree.nearest_one::<SquaredEuclidean>(&[point.x as f32, point.y as f32]);
            assert!(nearest.distance < MAX_GOOD_STAR_MATCH_RADIUS);
            matched_indexes.insert(nearest.item);
            matched_candidates.push(&results[nearest.item]);
        }

        assert_eq!(matched_indexes.len(), known_good.len());
    }
}

pub fn is_good_stellar_fit(
    reduced_chi_squared: f32,
    sigma_x: f32,
    sigma_y: f32,
    amplitude: f32,
    converged: bool,
) -> bool {
    converged
        && reduced_chi_squared < 3.0
        && sigma_x > 0.5
        && sigma_x < 3.0
        && sigma_y > 0.5
        && sigma_y < 3.0
        && amplitude > 5.0
}

// Move calculate_patch_variance to be a standalone function
fn calculate_patch_variance(frame: &CpuFrame, centre_x: f32, centre_y: f32) -> f32 {
    let mut sum = 0f32;
    let mut sum_squared = 0f32;
    let mut count = 0f32;

    let x_range = (centre_x as i32 - PATCH_SIZE as i32).max(0) as u32
        ..(centre_x as i32 + PATCH_SIZE as i32).min((frame.img.width() - 1) as i32) as u32;

    let y_range = (centre_y as i32 - PATCH_SIZE as i32).max(0) as u32
        ..(centre_y as i32 + PATCH_SIZE as i32).min((frame.img.height() - 1) as i32) as u32;

    // Calculate mean and variance of patch pixels
    for x in x_range.clone() {
        for y in y_range.clone() {
            let pixel_val = frame.img.get_pixel(x, y).channels()[0] as f32;
            sum += pixel_val;
            sum_squared += pixel_val * pixel_val;
            count += 1f32;
        }
    }

    if count > 0f32 {
        let mean = sum / count;

        (sum_squared / count) - (mean * mean)
    } else {
        0f32
    }
}

fn better_initial_params(frame: &CpuFrame, centre_x: f32, centre_y: f32) -> (Vec<f32>, Vec<f32>) {
    // Estimate seeing from image characteristics or use typical value
    let typical_seeing_fwhm = 2.0f32; // pixels
    let initial_sigma = typical_seeing_fwhm / 2.355f32;

    // Better amplitude estimate: max in 3x3 region
    let mut max_amplitude = 0u8;
    let cx = centre_x as u32;
    let cy = centre_y as u32;

    for dy in -1..=1 {
        for dx in -1..=1 {
            let x = (cx as i32 + dx) as u32;
            let y = (cy as i32 + dy) as u32;
            if x < frame.img.width() && y < frame.img.height() {
                max_amplitude = max_amplitude.max(frame.img.get_pixel(x, y).channels()[0]);
            }
        }
    }

    let amplitude_estimate = max_amplitude as f32;

    let initial_params = vec![
        0f32,               // x_offset
        0f32,               // y_offset
        initial_sigma,      // sigma_x
        initial_sigma,      // sigma_y
        amplitude_estimate, // amplitude
    ];

    let perturbations = vec![
        0.5f32,                      // x_offset (sub-pixel)
        0.5f32,                      // y_offset
        (initial_sigma * 0.3f32),    // sigma_x (30%)
        (initial_sigma * 0.3f32),    // sigma_y
        amplitude_estimate * 0.2f32, // amplitude (20%)
    ];

    (initial_params, perturbations)
}

fn clamp_params(params: &[f32]) -> Vec<f32> {
    vec![
        // Clamp x and y offsets to reasonable sub-pixel values
        params[0].min(2.0f32).max(-2.0f32),
        params[1].min(2.0f32).max(-2.0f32),
        // Clamp sigma values to reasonable ranges
        params[2].min(MAX_SIGMA).max(MIN_SIGMA),
        params[3].min(MAX_SIGMA).max(MIN_SIGMA),
        // Amplitude should be positive
        params[4].max(0f32),
    ]
}

// Helper functions for smooth transitions
fn linear_score(x: f32, x_min: f32, x_max: f32, score_min: f32, score_max: f32) -> f32 {
    if x <= x_min {
        score_min
    } else if x >= x_max {
        score_max
    } else {
        let ratio = (x - x_min) / (x_max - x_min);
        score_min + ratio * (score_max - score_min)
    }
}

fn sigmoid_score(x: f32, midpoint: f32, steepness: f32, min_score: f32, max_score: f32) -> f32 {
    let exp_arg = -steepness * (x - midpoint);
    let sigmoid = 1f32 / (1f32 + exp_arg.exp());
    min_score + sigmoid * (max_score - min_score)
}
