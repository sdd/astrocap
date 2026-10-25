use crate::Frame;
use crate::structs::{Detection, FittedPoint};
use crate::traits::TrackSummary;
use std::sync::Arc;

/// Per-frame median image published by Astrocap's median processor.
#[derive(Clone)]
pub struct MedianFrame(pub Arc<Frame>);

/// Detector output shared by recorded and live sources and their consumers.
#[derive(Clone, Debug)]
pub struct DetectedPoints(pub Vec<Detection>);

/// Fitter output. Currently producer-only on the supported pipeline.
#[derive(Clone, Debug)]
pub struct FittedPoints(pub Vec<FittedPoint>);

/// One selected track per physical target, published on every tracker frame.
#[derive(Clone, Debug)]
pub struct TrackSummaries(pub Vec<TrackSummary>);
