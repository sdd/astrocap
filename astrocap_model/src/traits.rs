use crate::state::{DetectedPoint, FittedPoint};
use argmin::core::ArgminFloat;
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::Cast;
use kiddo::float::kdtree::Axis;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use num_traits::float::FloatCore;
use serde::Serialize;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};
use std::sync::Arc;

pub trait AstroFloat:
    ArgminFloat
    + Axis
    + Sum
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + FloatCore
    + Sync
    + Send
    + Copy
    + Default
    + Debug
    + Serialize
    + Cast<f64>
{
}

impl<
        T: ArgminFloat
            + Axis
            + Sum
            + AddAssign
            + SubAssign
            + MulAssign
            + DivAssign
            + FloatCore
            + Sync
            + Send
            + Copy
            + Default
            + Debug
            + Serialize
            + Cast<f64>
            + Cast<f32>
            + Cast<u32>
            + Cast<i32>,
    > AstroFloat for T
{
}

pub trait PointDetector<F: AstroFloat> {
    fn detect(
        img: Arc<dyn ImageLumaExtractor>,
        mask: Option<Arc<dyn ImageLumaExtractor>>,
    ) -> Vec<DetectedPoint<F>>;
}

pub trait PointFitter<F: AstroFloat>
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
    fn new(frame: Arc<dyn ImageLumaExtractor>) -> Self
    where
        Self: Sized;

    fn fit(&self, point: &DetectedPoint<F>) -> FittedPoint<F>;
}

pub trait ImageLumaExtractor {
    fn get_luma8_for_pixel(&self, x: u32, y: u32) -> u8;
    fn width(&self) -> u32;
    fn height(&self) -> u32;
}
