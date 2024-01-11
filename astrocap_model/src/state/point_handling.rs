use crate::state::ModelState;
use crate::traits::PointFitter;
use argmin::core::ArgminFloat;
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::Cast;
use kiddo::float::kdtree::Axis;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use nonmax::NonMaxUsize;
use serde::{Deserialize, Serialize};
use std::iter::Sum;

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
    pub(crate) fn fit_existing_points<PF: PointFitter<F>>(self: &mut Self) {
        for cand in &self.star_candidates {
            // get position last frame

            // get matching points within radius from current frame

            // fit each match, if not already fitted

            // pick the best match from the fitted points
        }
    }
}
