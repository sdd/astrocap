use crate::state::ModelState;
use crate::traits::{AstroFloat, PointFitter};

use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::{Az, Cast};

use kiddo::SquaredEuclidean;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use nonmax::NonMaxUsize;
use num_traits::float::FloatCore;
use serde::Serialize;
use std::collections::HashSet;

use std::sync::Arc;
use tracing::*;

#[derive(Clone, Debug, Serialize)]
pub struct DetectedPoint<F: AstroFloat> {
    pub x: u32,
    pub y: u32,
    pub amplitude: F,
    pub fitted_point: Option<FittedPoint<F>>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FittedPoint<F: AstroFloat> {
    pub x: F,
    pub y: F,
    pub amplitude: F,
    pub radius: F,
    pub score: F,
}

#[derive(Clone, Debug, Serialize)]
pub struct StarCandidate<F: AstroFloat> {
    pub x: F,
    pub y: F,
    pub amplitude: F,
    pub radius: F,
    pub age: usize,
    pub log_likelihood: F,

    pub match_name: Option<String>,

    pub detected_point_match_history: Vec<Option<NonMaxUsize>>,
}

impl<F: AstroFloat> ModelState<F>
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
    pub(crate) fn fit_existing_points(&mut self, frame: Arc<dyn PointFitter<F>>) -> HashSet<usize> {
        let mut match_indexes: HashSet<usize> = HashSet::new();

        let Some(curr_frame_state) = self.recent_frame_states.last_mut() else {
            warn!("Not enough previous frame states found when trying to fit existing points");
            return match_indexes;
        };

        let mut unmatched_cand_count = 0;
        let mut matched_detected_points: HashSet<usize> = HashSet::new();
        'outer: for (_cand_idx, cand) in self.star_candidates.iter_mut().enumerate() {
            let x = cand.x;
            let y = cand.y;

            // if cand.age > 5 {
            //     info!(
            //         "cand #{}: pos ({}, {}), dp index: {:?}",
            //         cand_idx,
            //         cand.x,
            //         cand.y,
            //         cand.detected_point_match_history.last()
            //     );
            // }

            // get close detected points within radius from current frame
            let nearby_detected_points = curr_frame_state
                .detected_points_tree
                .nearest_n_within::<SquaredEuclidean>(
                    &[x, y],
                    self.model_config.max_existing_candidate_match_dist,
                    usize::MAX,
                    true,
                );

            // fit those points
            let nearby_fitted_points = nearby_detected_points
                .iter()
                .map(|nn| {
                    let detected_point: &mut DetectedPoint<F> = curr_frame_state
                        .detected_points_list
                        .get_mut(nn.item as usize)
                        .unwrap();
                    let fitted_point = frame.fit(detected_point);
                    detected_point.fitted_point = Some(fitted_point);

                    nn.item as usize
                })
                .collect::<Vec<_>>();

            let rev_points = nearby_fitted_points.iter().rev();
            for &detected_point_idx in rev_points {
                let detected_point = &curr_frame_state.detected_points_list[detected_point_idx];

                let Some(fitted_point) = &detected_point.fitted_point else {
                    panic!("no fitted point, should not happen");
                };

                if fitted_point.radius > self.model_config.max_star_candidate_radius {
                    continue;
                }

                if matched_detected_points.contains(&detected_point_idx) {
                    cand.detected_point_match_history.push(None);
                    unmatched_cand_count += 1;
                    continue 'outer;
                }

                cand.detected_point_match_history
                    .push(Some(NonMaxUsize::try_from(detected_point_idx).unwrap()));
                matched_detected_points.insert(detected_point_idx);

                let delta_x = fitted_point.x - cand.x;
                let delta_y = fitted_point.y - cand.y;
                // info!(
                //     "cand #{} delta: {:.2}, {:.2}, DMH: {:?}",
                //     cand_idx, delta_x, delta_y, &cand.detected_point_match_history
                // );

                let update_scale =
                    FloatCore::max(2.0.az::<F>(), fitted_point.score) / 20.0f64.az::<F>();

                cand.x += delta_x * update_scale;
                cand.y += delta_y * update_scale;

                if fitted_point.score > self.model_config.star_candidate_strong_match_threshold {
                    cand.log_likelihood += self.model_config.star_candidate_strong_match_bonus;
                }

                match_indexes.insert(detected_point_idx);

                continue 'outer;
            }

            unmatched_cand_count += 1;
            cand.detected_point_match_history.push(None);
        }
        info!(
            existing_fitted_qty = match_indexes.len(),
            unmatched_cand_count
        );

        match_indexes
    }

    pub(crate) fn fit_strong_unmatched_new_points(
        &mut self,
        point_fitter: Arc<dyn PointFitter<F>>,
        img_w: u32,
        img_h: u32,
    ) {
        let Some(curr_frame_state) = self.recent_frame_states.last_mut() else {
            warn!("Not enough previous frame states found when trying to fit existing points");
            return;
        };

        let initial_len = self.star_candidates.len();

        for (detected_point_index, detected_point) in
            curr_frame_state.detected_points_list.iter_mut().enumerate()
        {
            // skip detected points that have already been matched up to existing
            // star candidates
            if self.matched_point_indices.contains(&detected_point_index) {
                continue;
            }

            // skip points that are too dim
            if detected_point.amplitude
                < self
                    .model_config
                    .detected_point_candidate_amplitude_threshold
            {
                continue;
            }

            // skip points that are too close to existing star candidates
            let close_candidates = self
                .star_candidates_tree
                .nearest_n_within::<SquaredEuclidean>(
                    &[detected_point.x.az::<F>(), detected_point.y.az::<F>()],
                    self.model_config.max_existing_candidate_match_dist,
                    1,
                    false,
                );
            if !close_candidates.is_empty() {
                continue;
            }

            if detected_point.fitted_point.is_none() {
                detected_point.fitted_point = Some(point_fitter.fit(detected_point));
            }

            let fitted_point = &detected_point.fitted_point.clone().unwrap();

            if fitted_point.x >= F::zero()
                && fitted_point.y >= F::zero()
                && fitted_point.x < img_w.az::<F>()
                && fitted_point.y < img_h.az::<F>()
                && fitted_point.score > self.model_config.min_new_star_candidate_score
                && fitted_point.radius < self.model_config.max_star_candidate_radius
            {
                let new_cand = StarCandidate {
                    x: fitted_point.x,
                    y: fitted_point.y,
                    amplitude: fitted_point.amplitude,
                    radius: fitted_point.radius,
                    age: 0,
                    log_likelihood: fitted_point.score,
                    match_name: None,
                    detected_point_match_history: vec![NonMaxUsize::new(detected_point_index)],
                };

                self.star_candidates.push(new_cand);
            }
        }

        info!(new_candidate_qty = self.star_candidates.len() - initial_len);
    }
}
