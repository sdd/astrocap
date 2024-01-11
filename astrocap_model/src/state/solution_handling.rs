use crate::state::ModelState;
use argmin::core::ArgminFloat;
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::Cast;
use kiddo::float::kdtree::Axis;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use serde::{Deserialize, Serialize};
use std::iter::Sum;

#[derive(Debug, Deserialize, Serialize)]
pub struct StarMatch<F: Axis + Sum> {
    star_candidate_index: usize,
    catalogue_index: usize,
    log_odds: F,
}

#[derive(Debug)]
pub struct Wcs {}

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
    pub(crate) fn verify_solution(self: &mut Self) -> bool {
        // TODO
        true
    }

    pub(crate) fn tune_solution(self: &mut Self) {
        // TODO
    }

    pub(crate) fn solve(self: &mut Self) {
        // TODO
    }
}
