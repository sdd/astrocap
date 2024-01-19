use crate::state::{DetectedPoint, FittedPoint};
use argmin::core::ArgminFloat;
use argmin_math::{ArgminAdd, ArgminMul, ArgminSub};
use az::Cast;
use kiddo::float::kdtree::Axis;
use ndarray::{ArrayBase, Dim, OwnedRepr};
use std::iter::Sum;
use std::sync::Arc;

pub trait PointDetector<F: ArgminFloat + Axis> {
    fn detect(
        img: Arc<dyn ImageLumaExtractor>,
        mask: Arc<dyn ImageLumaExtractor>,
    ) -> Vec<DetectedPoint<F>>;
}

pub trait PointFitter<F: ArgminFloat + Axis + Sum>
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
