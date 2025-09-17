use crate::Dumpable;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DETECTION_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
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

#[derive(serde::Deserialize, serde::Serialize)]
pub struct DetectionRow {
    pub id: u64,
    pub run_id: u64,
    pub frame_index: usize,

    pub x: f32,
    pub y: f32,
    pub amplitude: f32,
}

impl Dumpable for Detection {
    type Row = DetectionRow;

    const TABLE_NAME: &'static str = "detection";
    const VERSION: u32 = 1;
    fn to_row(&self, run_id: u64, frame_index: usize) -> Self::Row {
        DetectionRow {
            run_id,
            frame_index,
            id: self.id,
            x: self.position.x,
            y: self.position.y,
            amplitude: self.amplitude,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FittedPoint {
    pub x: f32,
    pub y: f32,
    pub amplitude: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub score: f32,
    pub fit_quality: FittedPointQuality,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FittedPointQuality {
    pub reduced_chi_squared: f32,
    pub snr: f32,
    pub r_squared: f32,
    pub rms_residual: f32,
    pub score: f32,
}
