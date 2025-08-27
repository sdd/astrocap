use std::sync::atomic::{AtomicU64, Ordering};

use nalgebra::Matrix4;

static NEXT_TRACK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,

    pub state: TrackState,
    pub age: usize,
    pub confidence: f32,
}

impl Track {
    pub fn new(state: TrackState, confidence: f32) -> Self {
        Self {
            id: NEXT_TRACK_ID.fetch_add(1, Ordering::AcqRel),
            state,
            age: 0,
            confidence,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrackState {
    /// State vector: [x, y, vx, vy]
    pub state: Matrix4<f32>,
    pub covariance: Matrix4<f32>,
}
