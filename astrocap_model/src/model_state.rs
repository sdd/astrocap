use argmin::core::ArgminFloat;
use std::error::Error;
use std::sync::Arc;

use crate::traits::{ImageLumaExtractor, PointDetector, PointFitter};
use az::{Az, Cast};
use kiddo::float::kdtree::Axis;
use kiddo::KdTree;
use nonmax::NonMaxUsize;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct ModelConfig {
    min_reqd_qty_to_attempt_solve: usize
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DetectedPoint<F: Axis> {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) amplitude: F,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FittedPoint<F: Axis> {
    pub(crate) x: F,
    pub(crate) y: F,
    pub(crate) amplitude: F,
    pub(crate) radius: F,
    pub(crate) score: F,
    pub(crate) detected_point_index: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StarCandidate<F: Axis> {
    age: usize,
    log_likelihood: F,

    fitted_point_match_history: Vec<Option<NonMaxUsize>>,
}

#[derive(Debug)]
pub struct FrameState<F: Axis> {
    detected_points_list: Vec<DetectedPoint<F>>,
    detected_points_tree: KdTree<F, 2>,

    fitted_points_list: Vec<FittedPoint<F>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StarMatch<F: Axis> {
    star_candidate_index: usize,
    catalogue_index: usize,
    log_odds: F,
}

#[derive(Debug)]
pub struct Wcs {}

#[derive(Debug, Deserialize, Serialize)]
pub struct MovingTarget<F: Axis> {
    log_odds: F,

    detected_point_match_history: Vec<Option<NonMaxUsize>>,

    last_frame_delta_x: F,
    last_frame_delta_y: F,
    // TODO:
    //  some kind of polynomial to represent the 2d path
    //  some kind of polynomial to fit the amplitude variation
    //  (eventually): Estimated Keplerian elements, assuming geocentric orbit
}

// TODO: something like the output of the ASTRIDE algo to
//       detect items moving so fast that they are streaks,
//       typically meteors
#[derive(Debug)]
pub struct MovingStreak {}

#[derive(Debug)]
pub struct ModelState<F: Axis> {
    model_config: ModelConfig,

    recent_frame_states: Vec<FrameState<F>>,

    star_candidates: Vec<StarCandidate<F>>,
    star_candidates_tree: KdTree<F, 2>,

    wcs: Option<Wcs>,
    star_matches: Option<Vec<StarMatch<F>>>,

    moving_targets: Vec<MovingTarget<F>>,
    moving_streaks: Vec<MovingStreak>,
}

impl<F: Axis> ModelState<F> {
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

impl<F: Axis + ArgminFloat> FrameState<F>
where
    usize: Cast<F>,
    u32: Cast<F>,
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

impl<F: Axis + ArgminFloat> ModelState<F>
where
    usize: Cast<F>,
    u32: Cast<F>,
{
    pub fn process_frame<PD: PointDetector<F>>(
        self: &mut Self,
        frame: Arc<dyn ImageLumaExtractor>,
    ) -> Result<(), Box<dyn Error>> {
        // Create a new frame state and run the point
        // detector against the incoming frame
        self.recent_frame_states.push(FrameState::from_frame::<PD>(frame));
        
        if self.recent_frame_states.len() > 1 {
            // try to fit existing star candidates
            self.fit_existing_points();
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

    fn fit_existing_points<PF: PointFitter<F>>(self: &mut Self) {
        for cand in self.star_candidates {
            // get position last frame

            // get matching points within radius from current frame

            // fit each match, if not already fitted

            // pick the best match from the fitted points
        }
    }
    
    fn verify_solution(self: &mut Self) {
        // TODO
    }

    fn tune_solution(self: &mut Self) {
        // TODO
    }
    
    fn solve(self: &mut Self) {
        // TODO
    }
    
    fn update_moving_targets(self: &mut Self) {
        for moving_target in self.moving_targets {
            // predict position of target in this frame

            // fit against predicted position
        }
    }
    
    fn detect_moving_targets(self: &mut Self) {
        // TODO
    }
    
    fn clean_up_state(self: &mut Self) {
        // TODO
    }
}

#[cfg(test)]
mod tests {
    use crate::model_state::{ModelConfig, ModelState};
    use crate::point_detect_peak::PointDetectPeak;
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

        let mut model_state: ModelState<f64> = ModelState::new(ModelConfig {});

        model_state
            .process_frame::<PointDetectPeak>(arc_img)
            .unwrap();
    }
}
