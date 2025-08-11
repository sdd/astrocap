use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct DetectedPoint {
    pub x: u32,
    pub y: u32,
    pub amplitude: u8,
    pub fitted: Option<FittedPoint>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FittedPoint {
    pub x: f32,
    pub y: f32,
    pub amplitude: f32,
    pub radius_x: f32,
    pub radius_y: f32,
    pub score: f32,
    pub fit_quality: FittedPointQuality,
}

#[derive(Debug, Clone, Serialize)]
pub struct FittedPointQuality {
    pub reduced_chi_squared: f32,
    pub snr: f32,
    pub r_squared: f32,
    pub rms_residual: f32,
    pub score: f32,
}
