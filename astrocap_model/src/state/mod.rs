pub mod moving_targets;
pub mod point_handling;
pub mod solution_handling;
pub mod streaks;

pub use moving_targets::MovingTarget;
pub use point_handling::{DetectedPoint, FittedPoint, StarCandidate};
pub use solution_handling::{StarMatch, Wcs};
pub use streaks::MovingStreak;

use argmin::core::ArgminFloat;
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use std::error::Error;
use std::iter::Sum;
use std::sync::Arc;

use crate::config::ModelConfig;
use crate::traits::{ImageLumaExtractor, PointDetector, PointFitter};
use az::{Az, Cast};
use kiddo::float::kdtree::Axis;
use kiddo::KdTree;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct FrameState<F: Axis + ArgminFloat + Sum> {
    detected_points_list: Vec<DetectedPoint<F>>,
    detected_points_tree: KdTree<F, 2>,

    fitted_points_list: Vec<FittedPoint<F>>,
}

#[derive(Debug)]
pub struct ModelState<F: Axis + ArgminFloat + Sum> {
    model_config: ModelConfig,

    recent_frame_states: Vec<FrameState<F>>,

    star_candidates: Vec<StarCandidate<F>>,
    star_candidates_tree: KdTree<F, 2>,

    wcs: Option<Wcs>,
    star_matches: Option<Vec<StarMatch<F>>>,

    moving_targets: Vec<MovingTarget<F>>,
    moving_streaks: Vec<MovingStreak>,
}

impl<F: Axis + ArgminFloat + Sum> ModelState<F> {
    pub fn new(model_config: ModelConfig) -> Self {
        ModelState {
            model_config,
            recent_frame_states: vec![],
            star_candidates: vec![],
            star_candidates_tree: KdTree::new(),
            wcs: None,
            star_matches: None,
            moving_targets: vec![],
            moving_streaks: vec![],
        }
    }
}

impl<F: Axis + ArgminFloat + Sum> FrameState<F>
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
    pub fn from_frame<PD: PointDetector<F>>(frame: Arc<dyn ImageLumaExtractor>) -> Self {
        let detected_points_list = PD::detect(frame, None);

        let mut detected_points_tree: KdTree<F, 2> =
            KdTree::with_capacity(detected_points_list.len());
        for (idx, point) in detected_points_list.iter().enumerate() {
            detected_points_tree.add(&[point.x.az::<F>(), point.y.az::<F>()], idx as u64);
        }

        Self {
            detected_points_list,
            detected_points_tree,

            fitted_points_list: vec![],
        }
    }
}

impl<F: Axis + ArgminFloat + Sum> ModelState<F>
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
    pub fn process_frame<PD: PointDetector<F>, PF: PointFitter<F>>(
        self: &mut Self,
        frame: Arc<dyn ImageLumaExtractor>,
    ) -> Result<(), Box<dyn Error>> {
        // Create a new frame state and run the point
        // detector against the incoming frame
        self.recent_frame_states
            .push(FrameState::from_frame::<PD>(frame));

        if self.recent_frame_states.len() > 1 {
            // try to fit existing star candidates
            self.fit_existing_points::<PF>();
        }

        if self.wcs.is_some() {
            // state is currently solved.
            // Verify against the current solution
            let solution_is_valid = self.verify_solution();

            // if we're still solved after verification,
            // tune up the match
            if solution_is_valid {
                self.tune_solution();
            }
        } else {
            // state is currently unsolved.
            // Determine if there is sufficient grounds to attempt a solution
            if self.star_candidates.len() >= self.model_config.min_reqd_qty_to_attempt_solve {
                self.solve();
            }
        }

        // update existing moving targett
        self.update_moving_targets();

        // detect new moving targets
        self.detect_moving_targets();

        // clean up any stale state
        self.clean_up_state();

        Ok(())
    }

    fn clean_up_state(self: &mut Self) {
        // TODO
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

        let mut model_state: ModelState<f64> = ModelState::new(ModelConfig::default());

        model_state
            .process_frame::<PointDetectPeak, PointFitterGaussianNelderMead>(arc_img)
            .unwrap();
    }
}
