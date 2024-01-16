use crate::state::ModelState;
use crate::traits::{ImageLumaExtractor, PointFitter};
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
use tracing::{debug, error, info, warn};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DetectedPoint<F: Axis> {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) amplitude: F,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FittedPoint<F: Axis> {
    pub x: F,
    pub y: F,
    pub amplitude: F,
    pub radius: F,
    pub score: F,
    pub(crate) detected_point_index: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct StarCandidate<F: Axis> {
    pub age: usize,
    pub log_likelihood: F,

    pub fitted_point_match_history: Vec<Option<NonMaxUsize>>,
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
    ) {
        let penultimate_frame_idx = self.recent_frame_states.len() - 2;
        let [prev_frame_state, curr_frame_state] =
            &self.recent_frame_states[penultimate_frame_idx..]
        else {
            warn!("Not enough previous frame states found when trying to fit existing points");
            return;
        };

        let mut existing_fitted_qty = 0;
        for cand in self.star_candidates.iter_mut() {
            if let Some(Some(idx)) = cand.fitted_point_match_history.last() {
                // get position last frame
                let FittedPoint { x, y, .. } = prev_frame_state.fitted_points_list[idx.get()];

                // get matching points within radius from current frame
                let matches = curr_frame_state
                    .detected_points_tree
                    .nearest_n_within::<SquaredEuclidean>(
                        &[x, y],
                        self.model_config.max_existing_candidate_match_dist,
                        10, // arbitrary
                        true,
                    );

                // fit each match, if not already fitted
                let fitted_matches = matches
                    .iter()
                    .map(|nn| (nn, &prev_frame_state.detected_points_list[nn.item as usize]))
                    .collect::<Vec<_>>();

                // pick the best match from the fitted points
                if let Some(closest_match) = fitted_matches.first() {
                    // TODO: some kind of matching based on the star_candidate's amplitude and radius
                    //       rather than just closest
                    cand.fitted_point_match_history.push(Some(
                        NonMaxUsize::try_from(closest_match.0.item as usize).unwrap(),
                    ));
                    existing_fitted_qty += 1;
                } else {
                    cand.fitted_point_match_history.push(None);
                }
            }
        }
        info!(existing_fitted_qty);
    }

    pub(crate) fn fit_strong_unmatched_new_points<PF: PointFitter<F>>(
        self: &mut Self,
        point_fitter: Arc<dyn PointFitter<F>>,
    ) {
        let Some(curr_frame_state) = &self.recent_frame_states.last() else {
            warn!("Not enough previous frame states found when trying to fit existing points");
            return;
        };

        // TODO: could this be more elegant?
        let mut already_fitted: HashSet<usize> = HashSet::new();
        for sc in &self.star_candidates {
            if let Some(last) = sc.fitted_point_match_history.last() {
                if let Some(idx) = last {
                    already_fitted.insert(idx.get());
                }
            }
        }

        let mut new_fitted_points = vec![];
        for (idx, point) in curr_frame_state.detected_points_list.iter().enumerate() {
            if already_fitted.contains(&idx) {
                continue;
            }

            if point.amplitude < 40.0.az::<F>() {
                continue;
            }

            let fitted_point = point_fitter.fit(point);

            if fitted_point.amplitude < 40.0.az::<F>() {
                continue;
            }

            new_fitted_points.push(fitted_point);
        }

        info!(new_fitted_qty = new_fitted_points.len());
    }
}
