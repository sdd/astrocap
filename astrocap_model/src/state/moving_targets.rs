use crate::state::ModelState;
use argmin::core::ArgminFloat;
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::Cast;
use kiddo::float::kdtree::Axis;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use nonmax::NonMaxUsize;
use serde::{Deserialize, Serialize};
use std::iter::Sum;

#[derive(Debug, Serialize, Deserialize)]
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
    pub(crate) fn update_moving_targets(&mut self) {
        for _moving_target in &self.moving_targets {
            // predict position of target in this frame

            // fit against predicted position
        }
    }

    pub(crate) fn detect_moving_targets(&mut self) {
        // TODO
    }
}
