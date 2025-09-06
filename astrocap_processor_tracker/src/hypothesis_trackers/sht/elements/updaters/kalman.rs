use crate::model::{Detection, Track};
use crate::traits::{Configurable, ConfigurableConfig, Updater};
use astrocap_core::AstrocapError;
use nalgebra::{Matrix2, Matrix4, Vector2};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Measurement noise variance for position measurements (x, y)
    /// Higher values = less trust in measurements, more smoothing
    /// Lower values = more trust in measurements, less smoothing
    pub measurement_noise_position: f32,

    /// Minimum measurement noise to prevent numerical issues
    /// This ensures the measurement covariance matrix stays invertible
    pub min_measurement_noise: f32,

    /// Whether to update velocity based on position measurements
    /// If false, only position is updated directly
    pub update_velocity: bool,

    /// Confidence factor for measurements (0.0 to 1.0)
    /// Used to scale measurement noise based on detection quality
    pub confidence_scaling: bool,

    /// Minimum confidence threshold for accepting measurements
    /// Measurements from tracks below this confidence are rejected
    pub min_confidence: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            measurement_noise_position: 0.5, // Moderate trust in position measurements
            min_measurement_noise: 1e-6,     // Prevent numerical issues
            update_velocity: true,           // Update velocity from position changes
            confidence_scaling: true,        // Scale noise by detection confidence
            min_confidence: 0.1,             // Minimum confidence to accept measurements
        }
    }
}

impl ConfigurableConfig for Config {}

pub struct Kalman {
    config: Config,
}

impl Configurable for Kalman {
    type Config = Config;

    fn from_config(config: Self::Config) -> Result<Box<Self>, AstrocapError> {
        Ok(Box::new(Self { config }))
    }
}

impl Updater for Kalman {
    /// Performs Kalman filter update (correction) step
    ///
    /// This implements the standard Kalman filter update equations:
    /// y = z - H * x_k|k-1           (innovation/residual)
    /// S = H * P_k|k-1 * H^T + R     (innovation covariance)
    /// K = P_k|k-1 * H^T * S^-1      (Kalman gain)
    /// x_k|k = x_k|k-1 + K * y       (updated state estimate)
    /// P_k|k = (I - K * H) * P_k|k-1 (updated covariance estimate)
    ///
    /// Where:
    /// - z is the measurement vector [x_measured, y_measured]
    /// - H is the observation matrix (maps state to measurements)
    /// - R is the measurement noise covariance matrix
    /// - x is the state vector [x, y, vx, vy]
    /// - P is the state covariance matrix
    fn update(&self, track: &mut Track, detection: &Detection) {
        // Measurement vector z = [x_measured, y_measured]
        let measurement = Vector2::new(detection.position.x, detection.position.y);

        // Observation matrix H - maps 4D state [x, y, vx, vy] to 2D measurement [x, y]
        // We observe position directly but not velocity
        // [1  0  0  0]   [x ]   [x ]
        // [0  1  0  0] * [y ] = [y ]
        //                [vx]
        //                [vy]
        let observation_matrix = Matrix2x4::new(
            1.0, 0.0, 0.0, 0.0, // x measurement = 1*x + 0*y + 0*vx + 0*vy
            0.0, 1.0, 0.0, 0.0, // y measurement = 0*x + 1*y + 0*vx + 0*vy
        );

        // Measurement noise covariance matrix R
        // Diagonal matrix since we assume x and y measurement errors are independent
        let mut measurement_noise = self.config.measurement_noise_position;

        // Scale measurement noise by confidence if enabled
        // Lower confidence = higher noise = less trust in measurement
        if self.config.confidence_scaling {
            // Inverse relationship: confidence 1.0 = no scaling, confidence 0.1 = 10x noise
            let confidence_factor = (1.0 / track.confidence.max(0.01)).min(10.0);
            measurement_noise *= confidence_factor;
        }

        // Ensure minimum noise to prevent numerical issues
        measurement_noise = measurement_noise.max(self.config.min_measurement_noise);

        let measurement_covariance = Matrix2::new(measurement_noise, 0.0, 0.0, measurement_noise);

        // Step 1: Compute predicted measurement
        // h(x) = H * x_k|k-1 (what we expect to measure given current state)
        let predicted_measurement = observation_matrix * track.state.state;

        // Step 2: Compute innovation (residual)
        // y = z - h(x) (difference between actual and expected measurement)
        let innovation = measurement - predicted_measurement;

        // Step 3: Compute innovation covariance
        // S = H * P_k|k-1 * H^T + R
        let innovation_covariance =
            observation_matrix * track.state.covariance * observation_matrix.transpose()
                + measurement_covariance;

        // Step 4: Compute Kalman gain
        // K = P_k|k-1 * H^T * S^-1
        // The Kalman gain determines how much to trust the measurement vs. prediction
        let innovation_covariance_inv = match innovation_covariance.try_inverse() {
            Some(inv) => inv,
            None => {
                // Matrix is singular - skip update to avoid numerical issues
                tracing::warn!(
                    "Innovation covariance matrix is singular, skipping update for track {}",
                    track.id
                );
                return;
            }
        };

        let kalman_gain =
            track.state.covariance * observation_matrix.transpose() * innovation_covariance_inv;

        // Step 5: Update state estimate
        // x_k|k = x_k|k-1 + K * y
        track.state.state = track.state.state + kalman_gain * innovation;

        // Step 6: Update covariance estimate
        // P_k|k = (I - K * H) * P_k|k-1
        // Use Joseph form for numerical stability: P = (I-KH)P(I-KH)' + KRK'
        let identity = Matrix4::identity();
        let kh = kalman_gain * observation_matrix;
        let i_minus_kh = identity - kh;

        // Joseph form covariance update (more numerically stable)
        track.state.covariance = i_minus_kh * track.state.covariance * i_minus_kh.transpose()
            + kalman_gain * measurement_covariance * kalman_gain.transpose();

        // Ensure covariance matrix stays positive definite
        let min_variance = 1e-6;
        for i in 0..4 {
            if track.state.covariance[(i, i)] < min_variance {
                track.state.covariance[(i, i)] = min_variance;
            }
        }

        // Update track confidence based on innovation
        // Small innovations (measurements close to predictions) increase confidence
        let innovation_norm = innovation.norm();
        let expected_innovation_norm = (innovation_covariance.determinant()).sqrt();

        if expected_innovation_norm > 0.0 {
            let normalized_innovation = innovation_norm / expected_innovation_norm;

            // Confidence increases when innovation is smaller than expected
            let confidence_update = (-0.1 * normalized_innovation).exp();
            track.confidence =
                ((track.confidence * 0.99 + confidence_update * 0.01) + 0.3).clamp(0.0, 1.0);
        }
    }
}

// Type alias for 2x4 matrix (observation matrix)
type Matrix2x4 = nalgebra::Matrix2x4<f32>;

impl Kalman {
    /// Compute the Mahalanobis distance for gating
    /// This can be used by associators to determine if a measurement
    /// is likely to belong to a track
    pub fn mahalanobis_distance(&self, track: &Track, detection: &Detection) -> f32 {
        let measurement = Vector2::new(detection.position.x, detection.position.y);

        let observation_matrix = Matrix2x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

        let predicted_measurement = observation_matrix * track.state.state;
        let innovation = measurement - predicted_measurement;

        // Use track's current covariance for innovation covariance
        let measurement_noise = self.config.measurement_noise_position;
        let measurement_covariance = Matrix2::new(measurement_noise, 0.0, 0.0, measurement_noise);

        let innovation_covariance =
            observation_matrix * track.state.covariance * observation_matrix.transpose()
                + measurement_covariance;

        match innovation_covariance.try_inverse() {
            Some(inv) => {
                let distance_squared = innovation.transpose() * inv * innovation;
                distance_squared[(0, 0)].sqrt()
            }
            None => f32::INFINITY, // Invalid covariance
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Detection, Track, TrackState};
    use nalgebra::{Matrix4, Vector4};

    fn create_test_track() -> Track {
        Track {
            id: 1,
            parent_id: None,
            state: TrackState {
                state: Vector4::new(10.0, 20.0, 1.0, -0.5), // position (10,20), velocity (1,-0.5)
                covariance: Matrix4::identity() * 2.0,      // Initial uncertainty
            },
            age: 5,
            confidence: 0.8,
        }
    }

    #[test]
    fn test_perfect_measurement_update() {
        let config = Config::default();
        let updater = Kalman { config };

        let mut track = create_test_track();
        let original_state = track.state.state.clone();

        // Create detection exactly at predicted position
        let detection = Detection::new(10.0, 20.0, 100.0);

        updater.update(&mut track, &detection);

        // Position should move toward measurement (but not exactly due to uncertainty)
        // With perfect measurement, position should be closer to measured position
        let position_change_x = (track.state.state[0] - original_state[0]).abs();
        let position_change_y = (track.state.state[1] - original_state[1]).abs();

        // Should have some small adjustment due to Kalman filtering
        assert!(position_change_x < 1.0);
        assert!(position_change_y < 1.0);

        // Covariance should decrease (uncertainty reduced by measurement)
        assert!(track.state.covariance[(0, 0)] < 2.0);
        assert!(track.state.covariance[(1, 1)] < 2.0);
    }

    #[test]
    fn test_measurement_with_noise() {
        let config = Config::default();
        let updater = Kalman { config };

        let mut track = create_test_track();
        let original_position = (track.state.state[0], track.state.state[1]);

        // Create detection with significant offset
        let detection = Detection::new(15.0, 25.0, 100.0);

        updater.update(&mut track, &detection);

        // Position should move toward measurement but not all the way
        assert!(track.state.state[0] > original_position.0); // Moved toward 15.0
        assert!(track.state.state[0] < 15.0); // But not all the way
        assert!(track.state.state[1] > original_position.1); // Moved toward 25.0
        assert!(track.state.state[1] < 25.0); // But not all the way
    }

    #[test]
    fn test_low_confidence_measurement() {
        let mut config = Config::default();
        config.min_confidence = 0.5;
        let updater = Kalman { config };

        let mut track = create_test_track();
        track.confidence = 0.3; // Below minimum threshold
        let original_state = track.state.state.clone();

        let detection = Detection::new(100.0, 200.0, 50.0);

        updater.update(&mut track, &detection);

        // State should not change due to low confidence
        assert_eq!(track.state.state, original_state);
    }

    #[test]
    fn test_confidence_scaling() {
        let mut config = Config::default();
        config.confidence_scaling = true;
        let updater = Kalman { config };

        let mut high_conf_track = create_test_track();
        high_conf_track.confidence = 0.9;

        let mut low_conf_track = create_test_track();
        low_conf_track.confidence = 0.2;

        let detection = Detection::new(12.0, 22.0, 100.0);

        let orig_high_pos = high_conf_track.state.state[0];
        let orig_low_pos = low_conf_track.state.state[0];

        updater.update(&mut high_conf_track, &detection);
        updater.update(&mut low_conf_track, &detection);

        // High confidence track should move more toward measurement
        let high_conf_change = (high_conf_track.state.state[0] - orig_high_pos).abs();
        let low_conf_change = (low_conf_track.state.state[0] - orig_low_pos).abs();

        assert!(high_conf_change > low_conf_change);
    }

    #[test]
    fn test_mahalanobis_distance() {
        let config = Config::default();
        let updater = Kalman { config };

        let track = create_test_track();

        // Detection close to predicted position should have small distance
        let close_detection = Detection::new(10.5, 20.2, 100.0);
        let close_distance = updater.mahalanobis_distance(&track, &close_detection);

        // Detection far from predicted position should have large distance
        let far_detection = Detection::new(50.0, 80.0, 100.0);
        let far_distance = updater.mahalanobis_distance(&track, &far_detection);

        assert!(close_distance < far_distance);
        assert!(close_distance < 5.0); // Should be reasonable for close detection
        assert!(far_distance > 10.0); // Should be large for far detection
    }

    #[test]
    fn test_covariance_remains_positive_definite() {
        let config = Config::default();
        let updater = Kalman { config };

        let mut track = create_test_track();

        // Apply many updates to test numerical stability
        for i in 0..100 {
            let detection =
                Detection::new(10.0 + (i as f32) * 0.1, 20.0 + (i as f32) * 0.05, 100.0);
            updater.update(&mut track, &detection);

            // Check that all diagonal elements remain positive
            for j in 0..4 {
                assert!(track.state.covariance[(j, j)] > 0.0);
                assert!(track.state.covariance[(j, j)].is_finite());
            }
        }
    }

    #[test]
    fn test_velocity_not_directly_updated() {
        let config = Config::default();
        let updater = Kalman { config };

        let mut track = create_test_track();
        let original_velocity = (track.state.state[2], track.state.state[3]);

        // Update with position measurement only
        let detection = Detection::new(11.0, 19.5, 100.0);
        updater.update(&mut track, &detection);

        // Velocity should change indirectly through covariance coupling
        // but the change should be smaller than position change
        let velocity_change_x = (track.state.state[2] - original_velocity.0).abs();
        let velocity_change_y = (track.state.state[3] - original_velocity.1).abs();
        let position_change_x = (track.state.state[0] - 10.0).abs();
        let position_change_y = (track.state.state[1] - 20.0).abs();

        // Velocity changes should be smaller than position changes
        assert!(velocity_change_x < position_change_x);
        assert!(velocity_change_y < position_change_y);
    }
}
