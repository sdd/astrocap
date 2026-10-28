use std::sync::atomic::{AtomicU64, Ordering};

use astrocap_core::{
    structs::Detection,
    traits::{TrackSummarize, TrackSummary},
};
use nalgebra::{Matrix4, Vector4};

static NEXT_TRACK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,

    pub state: TrackState,
    pub age: usize,
    pub confidence: f32,
    pub first_seen: u64,
    pub start_x: f32,
    pub start_y: f32,
    pub amplitude: f32,
}

impl Track {
    pub fn from_detection(d: &Detection, confidence: f32) -> Self {
        let state = TrackState::from_detection(d);

        Self {
            id: NEXT_TRACK_ID.fetch_add(1, Ordering::AcqRel),
            state,
            age: 0,
            confidence,
            first_seen: 0,
            start_x: d.position.x,
            start_y: d.position.y,
            amplitude: d.amplitude,
        }
    }
}

impl TrackSummarize for Track {
    fn summarize(&self) -> TrackSummary {
        TrackSummary {
            id: self.id,
            x: self.state.state.x,
            y: self.state.state.y,
            amplitude: self.amplitude,
            age: self.age,
            log_odds: self.confidence,

            start_x: self.start_x,
            start_y: self.start_y,
            first_seen: self.first_seen,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrackState {
    /// State vector: [x, y, vx, vy]
    pub state: Vector4<f32>,
    pub covariance: Matrix4<f32>,
}

impl TrackState {
    pub fn from_pos(x: f32, y: f32) -> Self {
        Self {
            state: Vector4::new(x, y, 0.0, 0.0),
            covariance: Matrix4::identity(),
        }
    }

    pub fn from_detection(d: &Detection) -> Self {
        Self::from_pos(d.position.x, d.position.y)
    }
}
