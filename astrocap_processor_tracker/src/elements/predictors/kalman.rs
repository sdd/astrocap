use crate::model::Track;
use crate::traits::{Configurable, ConfigurableConfig, Predictor};
use astrocap_core::AstrocapError;
use nalgebra::{Matrix4, Vector4};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Time step between predictions (typically 1.0 for frame-to-frame)
    pub dt: f32,

    /// Process noise variance for position coordinates
    /// Higher values = more trust in motion model, less in measurements
    pub process_noise_position: f32,

    /// Process noise variance for velocity coordinates
    /// Higher values = expect more velocity changes
    pub process_noise_velocity: f32,

    /// Initial position variance for new tracks
    /// Higher values = less certain about initial position
    pub initial_position_variance: f32,

    /// Initial velocity variance for new tracks
    /// Higher values = less certain about initial velocity (usually high since we start with 0)
    pub initial_velocity_variance: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dt: 1.0,                         // 1 frame time step
            process_noise_position: 0.1,     // Small position uncertainty per frame
            process_noise_velocity: 1.0,     // Moderate velocity uncertainty
            initial_position_variance: 1.0,  // Moderate initial position uncertainty
            initial_velocity_variance: 10.0, // High initial velocity uncertainty
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

impl Predictor for Kalman {
    /// Performs Kalman filter prediction step
    ///
    /// This implements the standard Kalman filter prediction equations:
    /// x_k|k-1 = F * x_k-1|k-1
    /// P_k|k-1 = F * P_k-1|k-1 * F^T + Q
    ///
    /// Where:
    /// - x is the state vector [x, y, vx, vy]
    /// - P is the covariance matrix
    /// - F is the state transition matrix
    /// - Q is the process noise covariance matrix
    fn predict(&self, track: &mut Track) {
        let dt = self.config.dt;

        // State transition matrix F for constant velocity model
        // [1  0  dt  0]   [x ]   [x + vx*dt]
        // [0  1   0 dt] * [y ] = [y + vy*dt]
        // [0  0   1  0]   [vx]   [vx       ]
        // [0  0   0  1]   [vy]   [vy       ]
        #[rustfmt::skip]
        let state_transition = Matrix4::new(
            1.0, 0.0, dt,  0.0,
            0.0, 1.0, 0.0, dt,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        );

        // Process noise covariance matrix Q
        // Models uncertainty in the constant velocity assumption
        // Larger values mean we expect more deviation from constant velocity
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        let dt4 = dt2 * dt2;

        // Continuous-time noise model integrated over dt
        // This accounts for acceleration noise affecting both position and velocity
        #[rustfmt::skip]
        let process_noise = Matrix4::new(
            // Position noise increases quadratically with time
            dt4/4.0 * self.config.process_noise_position, 0.0, dt3/2.0 * self.config.process_noise_position, 0.0,
            0.0, dt4/4.0 * self.config.process_noise_position, 0.0, dt3/2.0 * self.config.process_noise_position,
            // Position-velocity cross terms
            dt3/2.0 * self.config.process_noise_position, 0.0, dt2 * self.config.process_noise_position, 0.0,
            0.0, dt3/2.0 * self.config.process_noise_position, 0.0, dt2 * self.config.process_noise_position,
        ) + Matrix4::new(
            // Additional velocity noise
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, self.config.process_noise_velocity, 0.0,
            0.0, 0.0, 0.0, self.config.process_noise_velocity,
        );

        // Prediction step
        // x_k|k-1 = F * x_k-1|k-1
        track.state.state = state_transition * track.state.state;

        // P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        track.state.covariance =
            state_transition * track.state.covariance * state_transition.transpose()
                + process_noise;

        // Ensure covariance matrix stays positive definite by adding small diagonal terms if needed
        // This prevents numerical instability in long-running filters
        let min_variance = 1e-6;
        for i in 0..4 {
            if track.state.covariance[(i, i)] < min_variance {
                track.state.covariance[(i, i)] = min_variance;
            }
        }
    }
}

impl Kalman {
    /// Initialize a track's covariance matrix with appropriate initial uncertainties
    /// This is typically called when creating new tracks from detections
    pub fn initialize_covariance(&self) -> Matrix4<f32> {
        Matrix4::new(
            // Position variances (x, y)
            self.config.initial_position_variance,
            0.0,
            0.0,
            0.0,
            0.0,
            self.config.initial_position_variance,
            0.0,
            0.0,
            // Velocity variances (vx, vy) - typically much higher since we start with zero velocity
            0.0,
            0.0,
            self.config.initial_velocity_variance,
            0.0,
            0.0,
            0.0,
            0.0,
            self.config.initial_velocity_variance,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elements::predictors::kalman::{Config, Kalman};
    use crate::model::{Detection, Track, TrackState};
    use nalgebra::Vector4;

    #[test]
    fn test_prediction_with_zero_velocity() {
        let config = Config::default();
        let kalman = Kalman {
            config: config.clone(),
        };

        // Create a track at position (10, 20) with zero velocity
        let mut track = Track {
            id: 1,
            state: TrackState {
                state: Vector4::new(10.0, 20.0, 0.0, 0.0),
                covariance: Matrix4::identity(),
            },
            age: 0,
            confidence: 0.8,
        };

        let original_position = (track.state.state[0], track.state.state[1]);

        // Predict one step
        kalman.predict(&mut track);

        // With zero velocity, position should remain the same
        assert_eq!(track.state.state[0], original_position.0);
        assert_eq!(track.state.state[1], original_position.1);

        // Velocity should remain zero
        assert_eq!(track.state.state[2], 0.0);
        assert_eq!(track.state.state[3], 0.0);

        // Covariance should have increased (uncertainty grows over time)
        assert!(track.state.covariance[(0, 0)] > 1.0); // Position uncertainty increased
        assert!(track.state.covariance[(1, 1)] > 1.0);
    }

    #[test]
    fn test_prediction_with_constant_velocity() {
        let config = Config::default();
        let kalman = Kalman {
            config: config.clone(),
        };

        // Create a track at position (10, 20) with velocity (2, -1)
        let mut track = Track {
            id: 1,
            state: TrackState {
                state: Vector4::new(10.0, 20.0, 2.0, -1.0),
                covariance: Matrix4::identity(),
            },
            age: 0,
            confidence: 0.8,
        };

        // Predict one step (dt = 1.0)
        kalman.predict(&mut track);

        // Position should be updated by velocity: new_pos = old_pos + velocity * dt
        assert!((track.state.state[0] - 12.0).abs() < 1e-6); // 10 + 2*1
        assert!((track.state.state[1] - 19.0).abs() < 1e-6); // 20 + (-1)*1

        // Velocity should remain constant in prediction step
        assert!((track.state.state[2] - 2.0).abs() < 1e-6);
        assert!((track.state.state[3] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_prediction_with_custom_dt() {
        let mut config = Config::default();
        config.dt = 0.5; // Half time step
        let kalman = Kalman { config };

        let mut track = Track {
            id: 1,
            state: TrackState {
                state: Vector4::new(0.0, 0.0, 4.0, -2.0),
                covariance: Matrix4::identity(),
            },
            age: 0,
            confidence: 0.8,
        };

        kalman.predict(&mut track);

        // With dt=0.5, position change should be velocity * 0.5
        assert!((track.state.state[0] - 2.0).abs() < 1e-6); // 0 + 4*0.5
        assert!((track.state.state[1] - (-1.0)).abs() < 1e-6); // 0 + (-2)*0.5
    }

    #[test]
    fn test_covariance_growth() {
        let config = Config::default();
        let kalman = Kalman {
            config: config.clone(),
        };

        let mut track = Track {
            id: 1,
            state: TrackState {
                state: Vector4::new(0.0, 0.0, 0.0, 0.0),
                covariance: Matrix4::identity(),
            },
            age: 0,
            confidence: 0.8,
        };

        let initial_covariance = track.state.covariance[(0, 0)];

        // Run multiple predictions
        for _ in 0..5 {
            kalman.predict(&mut track);
        }

        // Covariance should grow over time (uncertainty increases without measurements)
        assert!(track.state.covariance[(0, 0)] > initial_covariance);
        assert!(track.state.covariance[(1, 1)] > initial_covariance);

        // All diagonal elements should be positive (positive definite matrix)
        for i in 0..4 {
            assert!(track.state.covariance[(i, i)] > 0.0);
        }
    }

    #[test]
    fn test_initialize_covariance() {
        let config = Config {
            initial_position_variance: 2.0,
            initial_velocity_variance: 20.0,
            ..Default::default()
        };
        let kalman = Kalman {
            config: config.clone(),
        };

        let covariance = kalman.initialize_covariance();

        // Check diagonal elements
        assert_eq!(covariance[(0, 0)], config.initial_position_variance);
        assert_eq!(covariance[(1, 1)], config.initial_position_variance);
        assert_eq!(covariance[(2, 2)], config.initial_velocity_variance);
        assert_eq!(covariance[(3, 3)], config.initial_velocity_variance);

        // Check that off-diagonal elements are zero (no initial correlation)
        assert_eq!(covariance[(0, 1)], 0.0);
        assert_eq!(covariance[(0, 2)], 0.0);
        assert_eq!(covariance[(1, 3)], 0.0);
    }

    #[test]
    fn test_numerical_stability() {
        let config = Config {
            process_noise_position: 1e-10, // Very small noise
            process_noise_velocity: 1e-10,
            ..Default::default()
        };
        let kalman = Kalman { config };

        let mut track = Track {
            id: 1,
            state: TrackState {
                state: Vector4::new(0.0, 0.0, 0.0, 0.0),
                covariance: Matrix4::identity() * 1e-8, // Very small initial covariance
            },
            age: 0,
            confidence: 0.8,
        };

        // Run many predictions to test numerical stability
        for _ in 0..1000 {
            kalman.predict(&mut track);

            // Ensure diagonal elements don't go negative or become NaN
            for i in 0..4 {
                assert!(track.state.covariance[(i, i)] >= 1e-6);
                assert!(track.state.covariance[(i, i)].is_finite());
            }
        }
    }
}
