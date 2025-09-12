use std::sync::atomic::{AtomicU64, Ordering};

use astrocap_core::traits::{TrackSummarize, TrackSummary};
use nalgebra::{Matrix4, Vector2, Vector4};

static NEXT_DETECTION_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_TRACK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone)]
pub struct Detection {
    pub id: u64,
    pub position: Vector2<f32>,
    pub amplitude: f32,
}

impl Detection {
    pub fn new(x: f32, y: f32, amplitude: f32) -> Self {
        Self {
            id: NEXT_DETECTION_ID.fetch_add(1, Ordering::AcqRel),
            position: Vector2::new(x, y),
            amplitude,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,

    pub state: TrackState,
    pub age: usize,
    pub confidence: f32,
}

impl Track {
    pub fn from_detection(d: &Detection, confidence: f32) -> Self {
        let state = TrackState::from_detection(d);

        Self {
            id: NEXT_TRACK_ID.fetch_add(1, Ordering::AcqRel),
            state,
            age: 0,
            confidence,
        }
    }
}

impl TrackSummarize for Track {
    fn summarize(&self) -> TrackSummary {
        TrackSummary {
            id: self.id,
            x: self.state.state.x,
            y: self.state.state.y,
            amplitude: 0.0f32,
            age: self.age,
            log_odds: self.confidence,

            start_x: 0.0,
            start_y: 0.0,
            first_seen: 0,
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
