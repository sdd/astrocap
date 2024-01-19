use crate::state::ModelState;
use crate::traits::PointFitter;
use argmin::core::ArgminFloat;
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::{Az, Cast};
use kiddo::float::kdtree::Axis;
use kiddo::SquaredEuclidean;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use nonmax::NonMaxUsize;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::iter::Sum;
use std::sync::Arc;
use tracing::*;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DetectedPoint<F: Axis> {
    pub x: u32,
    pub y: u32,
    pub amplitude: F,
    pub fitted_point: Option<FittedPoint<F>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FittedPoint<F: Axis> {
    pub x: F,
    pub y: F,
    pub amplitude: F,
    pub radius: F,
    pub score: F,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StarCandidate<F: Axis> {
    pub x: F,
    pub y: F,
    pub age: usize,
    pub log_likelihood: F,

    pub detected_point_match_history: Vec<Option<NonMaxUsize>>,
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
    pub(crate) fn fit_existing_points<PF: PointFitter<F>>(
        self: &mut Self,
        frame: Arc<dyn PointFitter<F>>,
    ) -> HashSet<usize> {
        let mut match_indexes: HashSet<usize> = HashSet::new();

        let Some(mut curr_frame_state) = self.recent_frame_states.last_mut() else {
            warn!("Not enough previous frame states found when trying to fit existing points");
            return match_indexes;
        };

        for cand in self.star_candidates.iter_mut() {
            let x = cand.x;
            let y = cand.y;

            // get close detected points within radius from current frame
            let nearby_detected_points = curr_frame_state
                .detected_points_tree
                .nearest_n_within::<SquaredEuclidean>(
                    &[x, y],
                    self.model_config.max_existing_candidate_match_dist,
                    1, // TODO: increase this and pick the best rather than nearest?
                    true,
                );

            // fit those points
            let mut nearby_fitted_points = nearby_detected_points
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

            // pick the best match from the fitted points
            if let Some(&detected_point_idx) = nearby_fitted_points.first() {
                // TODO: some kind of matching based on the star_candidate's amplitude and radius
                //       rather than just closest
                cand.detected_point_match_history
                    .push(Some(NonMaxUsize::try_from(detected_point_idx).unwrap()));

                let detected_point: &DetectedPoint<F> =
                    &curr_frame_state.detected_points_list[detected_point_idx];

                // TODO: smoother update
                cand.x = detected_point
                    .fitted_point
                    .as_ref()
                    .and_then(|fp| Some(fp.x))
                    .unwrap();
                cand.y = detected_point
                    .fitted_point
                    .as_ref()
                    .and_then(|fp| Some(fp.y))
                    .unwrap();

                match_indexes.insert(detected_point_idx);
            } else {
                cand.detected_point_match_history.push(None);
            }
        }
        info!(existing_fitted_qty = match_indexes.len());

        match_indexes
    }

    pub(crate) fn fit_strong_unmatched_new_points<PF: PointFitter<F>>(
        self: &mut Self,
        point_fitter: Arc<dyn PointFitter<F>>,
        img_w: u32,
        img_h: u32,
        matched_point_indices: HashSet<usize>,
    ) {
        let Some(curr_frame_state) = self.recent_frame_states.last_mut() else {
            warn!("Not enough previous frame states found when trying to fit existing points");
            return;
        };

        let initial_len = self.star_candidates.len();

        for (detected_point_index, detected_point) in
            curr_frame_state.detected_points_list.iter_mut().enumerate()
        {
            if matched_point_indices.contains(&detected_point_index) {
                continue;
            }

            if detected_point.amplitude
                < self
                    .model_config
                    .detected_point_candidate_amplitude_threshold
            {
                continue;
            }

            if detected_point.fitted_point.is_none() {
                detected_point.fitted_point = Some(point_fitter.fit(&detected_point));
            }

            let fitted_point = &detected_point.fitted_point.clone().unwrap();

            if fitted_point.x >= F::zero()
                && fitted_point.y >= F::zero()
                && fitted_point.x < img_w.az::<F>()
                && fitted_point.y < img_h.az::<F>()
                && fitted_point.score > self.model_config.min_new_star_candidate_score
            {
                let new_cand = StarCandidate {
                    x: fitted_point.x,
                    y: fitted_point.y,
                    age: 0,
                    log_likelihood: fitted_point.score,
                    detected_point_match_history: vec![NonMaxUsize::new(detected_point_index)],
                };

                self.star_candidates.push(new_cand);
            }
        }

        info!(new_candidate_qty = self.star_candidates.len() - initial_len);
    }
}
