pub mod moving_targets;
pub mod point_handling;
// pub mod solution_handling;
pub mod streaks;

pub use moving_targets::MovingTarget;
pub use point_handling::{DetectedPoint, FittedPoint, StarCandidate};
// pub use solution_handling::{StarMatch, Wcs};
use std::collections::HashSet;
pub use streaks::MovingStreak;

use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use std::error::Error;
use std::iter::Sum;
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

use crate::config::ModelConfig;
use crate::traits::{AstroFloat, ImageLumaExtractor, PointDetector, PointFitter};
use az::{Az, Cast};

// use crate::state::solution_handling::Solver;
use kiddo::{KdTree, SquaredEuclidean};
use ndarray::{ArrayBase, Dim, OwnedRepr};
use num_traits::float::FloatCore;
use ordered_float::OrderedFloat;
use rerun::RecordingStream;

const POINT_EXCLUSION_DIST: f64 = 3.4f64;

#[derive(Debug)]
pub struct FrameState<F: AstroFloat> {
    pub detected_points_list: Vec<DetectedPoint<F>>,
    pub detected_points_tree: KdTree<F, 2>,
}

#[derive(Debug)]
pub struct ModelState<F: AstroFloat> {
    model_config: ModelConfig<F>,
    rec: RecordingStream,

    pub recent_frame_states: Vec<FrameState<F>>,

    pub star_candidates: Vec<StarCandidate<F>>,
    #[allow(dead_code)]
    star_candidates_tree: KdTree<F, 2>,
    matched_point_indices: HashSet<usize>,

    // wcs: Option<Wcs>,
    #[allow(dead_code)]
    // star_matches: Option<Vec<StarMatch<F>>>,
    pub moving_targets: Vec<MovingTarget<F>>,
    pub moving_targets_tree: KdTree<F, 2>,

    #[allow(dead_code)]
    moving_streaks: Vec<MovingStreak>,
    // solver: Option<Solver>,
}

impl<F: AstroFloat> ModelState<F> {
    pub fn new(
        model_config: ModelConfig<F>,
        star_index_path: Option<PathBuf>,
        rec: RecordingStream,
    ) -> Self {
        ModelState {
            model_config,
            rec,
            recent_frame_states: vec![],
            star_candidates: vec![],
            star_candidates_tree: KdTree::new(),
            matched_point_indices: HashSet::new(),
            // wcs: None,
            // star_matches: None,
            moving_targets: vec![],
            moving_targets_tree: KdTree::new(),
            moving_streaks: vec![],
            // solver: star_index_path.map(|path| Solver::new(&path)),
        }
    }
}

impl<F: AstroFloat> FrameState<F>
where
    u32: Cast<F>,
    u8: Cast<F>,
    f64: Cast<F>,
    f32: Cast<F>,
    F: Cast<u32>,
    F: Cast<i32>,
    F: Cast<f32>,
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
    pub fn from_frame<PD: PointDetector<F>, PF: PointFitter<F>>(
        frame: Arc<dyn ImageLumaExtractor>,
        median: Option<Arc<dyn ImageLumaExtractor>>,
        mask: Option<Arc<dyn ImageLumaExtractor>>,
        rec: RecordingStream,
    ) -> Self {
        let point_fitter: Arc<dyn PointFitter<F>> = Arc::new(PF::new(frame.clone()));

        let mut detected_points_list = PD::detect(frame, median, mask);

        rec.log(
            format!("model/detected_peaks"),
            &rerun::Points2D::new(
                detected_points_list
                    .iter()
                    .map(|cand| (cand.x.az::<f32>(), cand.y.az::<f32>())),
            ),
        )
        .unwrap();

        let mut fitted_points_list: Vec<_> = detected_points_list
            .iter()
            .map(|cand| point_fitter.fit(cand))
            .collect();

        rec.log(
            format!("model/fitted_points"),
            &rerun::Points2D::new(
                fitted_points_list
                    .iter()
                    .map(|cand| (cand.x.az::<f32>(), cand.y.az::<f32>())),
            )
            .with_labels(fitted_points_list.iter().map(|cand| format!("{:?}", &cand))),
        )
        .unwrap();

        let mut detected_points_tree: KdTree<F, 2> =
            KdTree::with_capacity(detected_points_list.len());

        let mut detected_points_removed_index_list: Vec<_> =
            Vec::with_capacity(detected_points_list.len());

        for (idx, point) in detected_points_list.iter().enumerate() {
            let query = [point.x.az::<F>(), point.y.az::<F>()];
            let mut near_neighbours = detected_points_tree.nearest_n_within::<SquaredEuclidean>(
                &query,
                POINT_EXCLUSION_DIST.az::<F>(),
                NonZero::new(usize::MAX).unwrap(),
                false,
            );

            near_neighbours.sort_unstable_by_key(|nn| {
                OrderedFloat(detected_points_list[nn.item as usize].amplitude)
            });
            if let Some(brightest) = near_neighbours.pop() {
                let brightest_point = &detected_points_list[brightest.item as usize];
                if brightest_point.amplitude < point.amplitude {
                    detected_points_tree.remove(
                        &[brightest_point.x.az::<F>(), brightest_point.y.az::<F>()],
                        brightest.item,
                    );
                    detected_points_removed_index_list.push(brightest.item);
                    detected_points_tree.add(&[point.x.az::<F>(), point.y.az::<F>()], idx as u64);
                }
            } else {
                detected_points_tree.add(&[point.x.az::<F>(), point.y.az::<F>()], idx as u64);
            }
            for close_point_result in near_neighbours {
                let point = &detected_points_list[close_point_result.item as usize];
                detected_points_tree.remove(
                    &[point.x.az::<F>(), point.y.az::<F>()],
                    close_point_result.item,
                );
                detected_points_removed_index_list.push(close_point_result.item);
            }
        }

        detected_points_removed_index_list.sort_unstable();
        for &idx in detected_points_removed_index_list.iter().rev() {
            detected_points_list.remove(idx as usize);
        }

        info!(detected_point_qty = detected_points_list.len());

        Self {
            detected_points_list,
            detected_points_tree,
        }
    }
}

impl<F: AstroFloat> ModelState<F>
where
    u32: Cast<F>,
    u8: Cast<F>,
    f64: Cast<F>,
    f32: Cast<F>,
    F: Cast<u32>,
    F: Cast<i32>,
    F: Cast<f32>,
    F: Cast<f64>,
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
    pub fn process_frame<PD: PointDetector<F>, PF: PointFitter<F> + 'static>(
        &mut self,
        frame: Arc<dyn ImageLumaExtractor>,
        median: Option<Arc<dyn ImageLumaExtractor>>,
        mask: Option<Arc<dyn ImageLumaExtractor>>,
        rec: RecordingStream,
    ) -> Result<(), Box<dyn Error>> {
        // Create a new frame state and run the point
        // detector against the incoming frame
        self.recent_frame_states
            .push(FrameState::from_frame::<PD, PF>(
                frame.clone(),
                median.clone(),
                mask,
                rec.clone(),
            ));

        let point_fitter: Arc<dyn PointFitter<F>> = Arc::new(PF::new(frame.clone()));

        self.update_state_positions();

        self.matched_point_indices = if self.recent_frame_states.len() > 1 {
            // try to fit existing star candidates
            self.fit_existing_points(point_fitter.clone())
        } else {
            HashSet::<usize>::new()
        };

        self.update_state();

        // fit high_quality candidates from the current frame that didn't
        // already get fitted against existing candidates
        self.fit_strong_unmatched_new_points(point_fitter.clone(), frame.width(), frame.height());

        // if self.wcs.is_some() {
        //     // state is currently solved.
        //     // Verify against the current solution
        //     let solution_is_valid = self.verify_solution();
        //
        //     // if we're still solved after verification,
        //     // tune up the match
        //     if solution_is_valid {
        //         self.tune_solution();
        //     }
        // } else {
        //     // state is currently unsolved.
        //     // Determine if there is sufficient grounds to attempt a solution
        //     self.solve();
        // }

        self.update_moving_targets();

        self.detect_moving_targets();

        self.clean_up_state();

        self.log_to_rerun();

        Ok(())
    }

    fn update_state_positions(&mut self) {
        for candidate in self.star_candidates.iter_mut() {
            if candidate.kalman_initialized {
                candidate.kalman_predict();
            }
        }

        self.star_candidates_tree = KdTree::with_capacity(self.star_candidates.len());

        for (cand_idx, cand) in self.star_candidates.iter_mut().enumerate() {
            self.star_candidates_tree
                .add(&[cand.x, cand.y], cand_idx as u64);
        }
    }

    fn update_state(&mut self) {
        let Some(curr_frame_state) = self.recent_frame_states.last() else {
            return;
        };

        let mut max_score = <F as FloatCore>::min_value();
        let mut min_score = <F as FloatCore>::max_value();

        // reset the star candidate tree so that positions are updated
        // before the next frame
        self.star_candidates_tree = KdTree::with_capacity(self.star_candidates.len());

        for (cand_idx, cand) in self.star_candidates.iter_mut().enumerate() {
            cand.age += 1;

            if let Some(last) = cand.detected_point_match_history.last() {
                if let Some(fitted_point_idx) = last {
                    // matched this frame
                    let point = &curr_frame_state.detected_points_list[fitted_point_idx.get()];

                    let score = point.fitted_point.as_ref().unwrap().score;
                    max_score = FloatCore::max(max_score, score);
                    min_score = FloatCore::min(min_score, score);
                    cand.log_likelihood += score;

                    if point.amplitude < self.model_config.amplitude_penalty_threshold {
                        cand.log_likelihood -= self.model_config.amplitude_penalty;
                    }
                } else {
                    // no match this frame
                    cand.log_likelihood -= self.model_config.star_candidate_unmatched_penalty;
                }
            }

            self.star_candidates_tree
                .add(&[cand.x, cand.y], cand_idx as u64);
        }

        info!(
            min_score = min_score.az::<i32>(),
            max_score = max_score.az::<i32>()
        );
    }

    fn clean_up_state(&mut self) {
        let pre_discard_count = self.star_candidates.len();
        self.star_candidates = self
            .star_candidates
            .clone()
            .into_iter()
            .filter(|cand| cand.log_likelihood > self.model_config.star_candidate_discard_threshold)
            .collect();

        info!(
            state_len = self.star_candidates.len(),
            discarded = pre_discard_count - self.star_candidates.len(),
        );
    }

    fn log_to_rerun(&mut self) {
        self.rec
            .log(
                format!("model/star_candidates"),
                &rerun::Points2D::new(
                    self.star_candidates
                        .iter()
                        // .filter(|cand| {
                        //     cand.log_likelihood
                        //         > self.model_config.star_candidate_strong_match_threshold
                        // })
                        .map(|cand| (cand.x.az::<f32>(), cand.y.az::<f32>())),
                )
                .with_colors(
                    self.star_candidates
                        .iter()
                        // .filter(|cand| {
                        //     cand.log_likelihood
                        //         > self.model_config.star_candidate_strong_match_threshold
                        // })
                        .map(|cand| {
                            if cand.log_likelihood.az::<f32>() < 30.0f32 {
                                rerun::Color::from_rgb(255u8, 0u8, 0u8)
                            } else if cand.age < 30 {
                                rerun::Color::from_rgb(0u8, 255u8, 255u8)
                            } else {
                                rerun::Color::from_rgb(0u8, 255u8, 0u8)
                            }
                        }),
                ),
            )
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::point_detect_peak::PointDetectPeak;
    use crate::point_fitter_nelder_mead::PointFitterGaussianNelderMead;
    use crate::state::{ModelConfig, ModelState};
    use crate::traits::ImageLumaExtractor;
    use image::io::Reader as ImageReader;
    use image::{ImageBuffer, Luma, Pixel};
    use std::sync::Arc;

    impl ImageLumaExtractor for ImageBuffer<Luma<u8>, Vec<u8>> {
        fn get_luma8_for_pixel(&self, x: u32, y: u32) -> u8 {
            self.get_pixel(x, y).channels()[0]
        }
        fn width(&self) -> u32 {
            self.width()
        }
        fn height(&self) -> u32 {
            self.height()
        }
    }

    #[test]
    fn can_construct_correctly() {
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

        let arc_img = Arc::new(img);

        let mut model_state: ModelState<f64> = ModelState::new(ModelConfig::default(), None);

        model_state
            .process_frame::<PointDetectPeak, PointFitterGaussianNelderMead>(arc_img, None)
            .unwrap();
    }
}
