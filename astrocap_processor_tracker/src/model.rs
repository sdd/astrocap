use std::sync::atomic::{AtomicU64, Ordering};

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
    pub parent_id: Option<u64>,

    pub state: TrackState,
    pub age: usize,
    pub confidence: f32,
}

impl Track {
    pub fn from_detection(d: &Detection, confidence: f32) -> Self {
        let state = TrackState::from_detection(d);

        Self {
            id: NEXT_TRACK_ID.fetch_add(1, Ordering::AcqRel),
            parent_id: None,
            state,
            age: 0,
            confidence,
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
