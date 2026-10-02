use crate::Frame;
use crate::frame::CpuFrame;
use crate::structs::{Detection, FittedPoint};
use std::sync::Arc;

pub use vyd::traits::{Dumpable, FrameProcessor, FrameSink, FrameSource, StageFactory};

#[derive(Debug, Clone)]
pub struct TrackSummary {
    pub id: u64,
    pub x: f32,
    pub y: f32,
    pub amplitude: f32,
    pub age: usize,
    pub log_odds: f32,

    pub first_seen: usize,
    pub start_x: f32,
    pub start_y: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StarCandidate {
    pub x: f32,
    pub y: f32,
    pub amp: f32,
}

pub trait TrackSummarize: Send + Sync + 'static {
    fn summarize(&self) -> TrackSummary;
}

pub trait PointDetector: Send + Sync + 'static {
    fn detect(
        &mut self,
        frame: &Frame,
        median: Option<Arc<Frame>>,
        mask: Option<&Frame>,
    ) -> Vec<Detection>;
}

pub trait PointFitter: Send + Sync + 'static {
    fn fit(&self, frame: &CpuFrame, point: &Detection) -> FittedPoint;
}
