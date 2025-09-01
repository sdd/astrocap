use std::collections::HashMap;

use crate::model::{Detection, Track};
use crate::traits::{Configurable, ConfigurableConfig, Initiator};
use astrocap_core::AstrocapError;
use nalgebra::Vector2;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Minimum detection amplitude threshold (0-255 for u8 luma)
    pub min_amplitude: f32,

    /// Maximum distance for detection association in pixels
    pub max_association_distance: f32,

    /// Number of frames to accumulate evidence before track initiation
    pub confirmation_frames: usize,

    /// Maximum number of hypotheses to maintain per detection cluster
    pub max_hypotheses_per_cluster: usize,

    /// Minimum score for hypothesis promotion to track
    pub promotion_threshold: f32,

    /// Score decay factor per frame for unconfirmed hypotheses (0.0-1.0)
    pub score_decay: f32,

    /// Initial confidence for newly created tracks
    pub initial_confidence: f32,

    /// Maximum age in frames for unconfirmed hypotheses
    pub max_hypothesis_age: usize,

    /// Weight for amplitude consistency in scoring (0.0-1.0)
    pub amplitude_consistency_weight: f32,

    /// Weight for motion consistency in scoring (0.0-1.0)
    pub motion_consistency_weight: f32,

    /// Weight for temporal consistency in scoring (0.0-1.0)
    pub temporal_consistency_weight: f32,

    /// Weight for detection count in scoring (0.0-1.0)
    pub detection_count_weight: f32,

    /// Maximum expected motion between frames in pixels
    /// Used for motion consistency calculations
    pub max_expected_motion: f32,

    /// Amplitude variation tolerance as a fraction of mean amplitude
    /// Lower values require more consistent amplitudes
    pub amplitude_tolerance: f32,

    /// Minimum detection frequency for temporal scoring
    /// Detections per frame (0.0-1.0)
    pub min_detection_frequency: f32,

    /// Initial score multiplier for bright detections
    /// Applied to detections above bright_threshold
    pub bright_detection_bonus: f32,

    /// Amplitude threshold for bright detection bonus (0-255)
    pub bright_threshold: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            // Astronomical imaging appropriate defaults
            min_amplitude: 50.0, // Below typical detector threshold but allows flexibility
            max_association_distance: 4.0, // Conservative for star tracking
            confirmation_frames: 3, // Quick confirmation for astronomy
            max_hypotheses_per_cluster: 8, // Reasonable for star fields
            promotion_threshold: 0.65, // Moderate threshold for promotion
            score_decay: 0.96,   // Slow decay for persistent objects
            initial_confidence: 0.75, // Good starting confidence for confirmed tracks
            max_hypothesis_age: 15, // Allow for temporary disappearances

            // Scoring weights (must sum to approximately 1.0)
            amplitude_consistency_weight: 0.3, // Stars have consistent brightness
            motion_consistency_weight: 0.25,   // Smooth motion is important for stars
            temporal_consistency_weight: 0.25, // Regular detection pattern
            detection_count_weight: 0.2,       // More detections = more confidence

            // Astronomical motion and variation parameters
            max_expected_motion: 8.0,     // Reasonable for stars/planets
            amplitude_tolerance: 0.25,    // 25% amplitude variation allowed
            min_detection_frequency: 0.3, // Detect in at least 30% of frames

            // Brightness-based bonuses
            bright_detection_bonus: 1.2, // 20% bonus for bright objects
            bright_threshold: 140.0,     // Consistent with existing BRIGHT_THRESHOLD
        }
    }
}

impl ConfigurableConfig for Config {}

/// Represents a hypothesis about a potential track
#[derive(Debug, Clone)]
pub struct TrackHypothesis {
    /// Unique hypothesis ID
    pub id: u64,

    /// Detections that support this hypothesis (chronological order)
    pub supporting_detections: Vec<Detection>,

    /// Current hypothesis score [0.0, 1.0]
    pub score: f32,

    /// Age of this hypothesis in frames
    pub age: usize,

    /// Frame number when hypothesis was created
    pub birth_frame: u64,

    /// Last frame number when this hypothesis was updated
    pub last_update_frame: u64,
}

impl TrackHypothesis {
    /// Create a new hypothesis from an initial detection
    pub fn new(id: u64, detection: Detection, frame_number: u64, config: &Config) -> Self {
        let initial_score = Self::calculate_initial_score(&detection, config);

        Self {
            id,
            supporting_detections: vec![detection],
            score: initial_score,
            age: 0,
            birth_frame: frame_number,
            last_update_frame: frame_number,
        }
    }

    /// Calculate initial score based on detection properties
    fn calculate_initial_score(detection: &Detection, config: &Config) -> f32 {
        // Normalize amplitude to 0-1 range (u8 max is 255)
        let amplitude_score = (detection.amplitude / 255.0).clamp(0.0, 1.0);

        // Apply bright detection bonus
        let brightness_multiplier = if detection.amplitude >= config.bright_threshold {
            config.bright_detection_bonus
        } else {
            1.0
        };

        // Start with moderate confidence based on amplitude
        (amplitude_score * 0.4 * brightness_multiplier).clamp(0.0, 1.0)
    }

    /// Update hypothesis with a new supporting detection
    pub fn add_supporting_detection(
        &mut self,
        detection: Detection,
        frame_number: u64,
        config: &Config,
    ) {
        self.supporting_detections.push(detection);
        self.last_update_frame = frame_number;
        self.update_score(config);
    }

    /// Update hypothesis score based on accumulated evidence
    fn update_score(&mut self, config: &Config) {
        let detection_count_score = self.calculate_detection_count_score(config);
        let consistency_score = self.calculate_consistency_score(config);
        let temporal_score = self.calculate_temporal_score(config);
        let motion_score = self.calculate_motion_consistency(config);

        // Weighted combination of scoring factors
        self.score = detection_count_score * config.detection_count_weight
            + consistency_score * config.amplitude_consistency_weight
            + temporal_score * config.temporal_consistency_weight
            + motion_score * config.motion_consistency_weight;

        self.score = self.score.clamp(0.0, 1.0);
    }

    /// Calculate score based on detection count
    fn calculate_detection_count_score(&self, _config: &Config) -> f32 {
        // More detections = higher confidence, but with diminishing returns
        let count = self.supporting_detections.len() as f32;
        (count / (count + 3.0)) // Asymptotic to 1.0
    }

    /// Calculate score based on detection amplitude consistency
    fn calculate_consistency_score(&self, config: &Config) -> f32 {
        if self.supporting_detections.len() < 2 {
            return 0.6; // Neutral score for single detection
        }

        let amplitudes: Vec<f32> = self
            .supporting_detections
            .iter()
            .map(|d| d.amplitude)
            .collect();

        let mean_amplitude = amplitudes.iter().sum::<f32>() / amplitudes.len() as f32;

        if mean_amplitude <= 0.0 {
            return 0.0;
        }

        // Calculate coefficient of variation (std dev / mean)
        let variance = amplitudes
            .iter()
            .map(|&a| (a - mean_amplitude).powi(2))
            .sum::<f32>()
            / amplitudes.len() as f32;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = std_dev / mean_amplitude;

        // Score based on how much variation we allow
        if coefficient_of_variation <= config.amplitude_tolerance {
            1.0 - (coefficient_of_variation / config.amplitude_tolerance).powi(2)
        } else {
            0.0
        }
    }

    /// Calculate score based on motion consistency
    fn calculate_motion_consistency(&self, config: &Config) -> f32 {
        if self.supporting_detections.len() < 3 {
            return 0.7; // Assume good motion until proven otherwise
        }

        // Calculate motion vectors between consecutive detections
        let mut motions = Vec::new();
        for window in self.supporting_detections.windows(2) {
            let motion = window[1].position - window[0].position;
            motions.push(motion);
        }

        if motions.len() < 2 {
            return 0.7;
        }

        // Calculate motion consistency (how similar are consecutive motions?)
        let mut motion_deviations = Vec::new();
        for window in motions.windows(2) {
            let deviation = (window[1] - window[0]).norm();
            motion_deviations.push(deviation);
        }

        let mean_deviation = motion_deviations.iter().sum::<f32>() / motion_deviations.len() as f32;

        // Score based on motion consistency relative to expected motion
        if mean_deviation <= config.max_expected_motion {
            1.0 - (mean_deviation / config.max_expected_motion).powi(2)
        } else {
            0.0
        }
    }

    /// Calculate temporal score based on detection frequency
    fn calculate_temporal_score(&self, config: &Config) -> f32 {
        let frame_span = self.last_update_frame - self.birth_frame + 1;
        let detection_frequency = self.supporting_detections.len() as f32 / frame_span as f32;

        // Score based on detection frequency
        if detection_frequency >= config.min_detection_frequency {
            (detection_frequency / 1.0).min(1.0) // Cap at 1.0 for perfect detection
        } else {
            detection_frequency / config.min_detection_frequency
        }
    }

    /// Age the hypothesis (called each frame)
    pub fn age(&mut self, current_frame: u64, config: &Config) {
        self.age += 1;

        // Apply score decay if not updated this frame
        if self.last_update_frame < current_frame {
            self.score *= config.score_decay;
        }
    }

    /// Check if hypothesis is ready for promotion to track
    pub fn ready_for_promotion(&self, config: &Config) -> bool {
        self.supporting_detections.len() >= config.confirmation_frames
            && self.score >= config.promotion_threshold
    }

    /// Convert hypothesis to a track
    pub fn to_track(&self, config: &Config) -> Track {
        // Use the most recent detection as the track's current position
        let latest_detection = self.supporting_detections.last().unwrap();
        Track::from_detection(latest_detection, config.initial_confidence)
    }
}

/// MHT-based track initiator
pub struct MhtInitiator {
    config: Config,

    /// Active hypotheses being tracked
    active_hypotheses: HashMap<u64, TrackHypothesis>,

    /// Next hypothesis ID
    next_hypothesis_id: u64,

    /// Current frame number
    current_frame: u64,
}

impl Configurable for MhtInitiator {
    type Config = Config;

    fn from_config(config: Self::Config) -> Result<Box<Self>, AstrocapError> {
        Ok(Box::new(Self {
            config,
            active_hypotheses: HashMap::new(),
            next_hypothesis_id: 1,
            current_frame: 0,
        }))
    }
}

impl Initiator for MhtInitiator {
    fn initiate(&mut self, detections: &[&Detection]) -> Vec<Track> {
        self.current_frame += 1;

        // Filter detections by amplitude threshold
        let valid_detections: Vec<&Detection> = detections
            .iter()
            .filter(|d| d.amplitude >= self.config.min_amplitude)
            .copied()
            .collect();

        // Associate detections with existing hypotheses
        let (associated_detections, unassociated_detections) =
            self.associate_detections_to_hypotheses(&valid_detections);

        // Update existing hypotheses with associated detections
        self.update_hypotheses_with_detections(associated_detections);

        // Create new hypotheses from unassociated detections
        self.create_new_hypotheses(unassociated_detections);

        // Age all hypotheses and apply score decay
        self.age_hypotheses();

        // Prune old or low-scoring hypotheses
        self.prune_hypotheses();

        // Promote qualifying hypotheses to tracks
        self.promote_hypotheses_to_tracks()
    }
}

impl MhtInitiator {
    /// Associate detections with existing hypotheses
    fn associate_detections_to_hypotheses(
        &self,
        detections: &[&Detection],
    ) -> (Vec<(u64, Detection)>, Vec<Detection>) {
        let mut associated = Vec::new();
        let mut unassociated = Vec::new();

        for detection in detections {
            let mut best_hypothesis_id = None;
            let mut best_distance = f32::INFINITY;

            // Find the closest hypothesis within the association distance
            for (hypothesis_id, hypothesis) in &self.active_hypotheses {
                if let Some(latest_detection) = hypothesis.supporting_detections.last() {
                    let distance = (detection.position - latest_detection.position).norm();

                    if distance <= self.config.max_association_distance && distance < best_distance
                    {
                        best_distance = distance;
                        best_hypothesis_id = Some(*hypothesis_id);
                    }
                }
            }

            match best_hypothesis_id {
                Some(hypothesis_id) => {
                    associated.push((hypothesis_id, (*detection).clone()));
                }
                None => {
                    unassociated.push((*detection).clone());
                }
            }
        }

        (associated, unassociated)
    }

    /// Update existing hypotheses with associated detections
    fn update_hypotheses_with_detections(&mut self, associations: Vec<(u64, Detection)>) {
        for (hypothesis_id, detection) in associations {
            if let Some(hypothesis) = self.active_hypotheses.get_mut(&hypothesis_id) {
                hypothesis.add_supporting_detection(detection, self.current_frame, &self.config);
            }
        }
    }

    /// Create new hypotheses from unassociated detections
    fn create_new_hypotheses(&mut self, detections: Vec<Detection>) {
        for detection in detections {
            let hypothesis_id = self.next_hypothesis_id;
            self.next_hypothesis_id += 1;

            let hypothesis =
                TrackHypothesis::new(hypothesis_id, detection, self.current_frame, &self.config);
            self.active_hypotheses.insert(hypothesis_id, hypothesis);
        }
    }

    /// Age all hypotheses and apply score decay
    fn age_hypotheses(&mut self) {
        for hypothesis in self.active_hypotheses.values_mut() {
            hypothesis.age(self.current_frame, &self.config);
        }
    }

    /// Remove old or low-scoring hypotheses
    fn prune_hypotheses(&mut self) {
        // Remove hypotheses that are too old or have very low scores
        let min_score = 0.05; // Configurable minimum score threshold

        self.active_hypotheses.retain(|_id, hypothesis| {
            hypothesis.age <= self.config.max_hypothesis_age && hypothesis.score > min_score
        });

        // Limit total number of hypotheses to prevent explosion
        let max_total_hypotheses = self.config.max_hypotheses_per_cluster * 20;
        if self.active_hypotheses.len() > max_total_hypotheses {
            // Keep only the highest scoring hypotheses
            let mut hypothesis_scores: Vec<(u64, f32)> = self
                .active_hypotheses
                .iter()
                .map(|(id, h)| (*id, h.score))
                .collect();

            hypothesis_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let ids_to_keep: std::collections::HashSet<u64> = hypothesis_scores
                .into_iter()
                .take(max_total_hypotheses / 2) // Keep top half
                .map(|(id, _)| id)
                .collect();

            self.active_hypotheses
                .retain(|id, _| ids_to_keep.contains(id));
        }
    }

    /// Promote qualifying hypotheses to tracks
    fn promote_hypotheses_to_tracks(&mut self) -> Vec<Track> {
        let mut promoted_tracks = Vec::new();
        let mut hypotheses_to_remove = Vec::new();

        for (hypothesis_id, hypothesis) in &self.active_hypotheses {
            if hypothesis.ready_for_promotion(&self.config) {
                let track = hypothesis.to_track(&self.config);
                promoted_tracks.push(track);
                hypotheses_to_remove.push(*hypothesis_id);
            }
        }

        // Remove promoted hypotheses
        for id in hypotheses_to_remove {
            self.active_hypotheses.remove(&id);
        }

        promoted_tracks
    }

    /// Get the number of active hypotheses (for debugging/monitoring)
    pub fn active_hypothesis_count(&self) -> usize {
        self.active_hypotheses.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    fn create_test_detection(x: f32, y: f32, amplitude: f32) -> Detection {
        Detection::new(x, y, amplitude)
    }

    fn create_test_config() -> Config {
        Config::default()
    }

    #[test]
    fn test_hypothesis_creation() {
        let config = create_test_config();
        let detection = create_test_detection(10.0, 20.0, 100.0);
        let hypothesis = TrackHypothesis::new(1, detection, 0, &config);

        assert_eq!(hypothesis.id, 1);
        assert_eq!(hypothesis.supporting_detections.len(), 1);
        assert_eq!(hypothesis.birth_frame, 0);
        assert!(hypothesis.score > 0.0);
    }

    #[test]
    fn test_bright_detection_bonus() {
        let config = create_test_config();

        // Dim detection
        let dim_detection = create_test_detection(10.0, 20.0, 100.0);
        let dim_score = TrackHypothesis::calculate_initial_score(&dim_detection, &config);

        // Bright detection (above bright_threshold)
        let bright_detection = create_test_detection(10.0, 20.0, 200.0);
        let bright_score = TrackHypothesis::calculate_initial_score(&bright_detection, &config);

        assert!(bright_score > dim_score);
    }

    #[test]
    fn test_amplitude_consistency() {
        let config = create_test_config();
        let mut consistent_hypothesis =
            TrackHypothesis::new(1, create_test_detection(10.0, 20.0, 100.0), 0, &config);

        // Add consistent detections
        consistent_hypothesis.add_supporting_detection(
            create_test_detection(10.1, 20.1, 102.0),
            1,
            &config,
        );
        consistent_hypothesis.add_supporting_detection(
            create_test_detection(10.2, 20.2, 98.0),
            2,
            &config,
        );

        let consistent_score = consistent_hypothesis.calculate_consistency_score(&config);

        // Create inconsistent hypothesis
        let mut inconsistent_hypothesis =
            TrackHypothesis::new(2, create_test_detection(10.0, 20.0, 100.0), 0, &config);

        inconsistent_hypothesis.add_supporting_detection(
            create_test_detection(15.0, 25.0, 200.0),
            1,
            &config,
        );
        inconsistent_hypothesis.add_supporting_detection(
            create_test_detection(8.0, 18.0, 50.0),
            2,
            &config,
        );

        let inconsistent_score = inconsistent_hypothesis.calculate_consistency_score(&config);

        assert!(consistent_score > inconsistent_score);
    }

    #[test]
    fn test_configurable_parameters() {
        let mut config = create_test_config();
        config.min_amplitude = 80.0;
        config.promotion_threshold = 0.8;
        config.score_decay = 0.9;

        let mut initiator = MhtInitiator {
            config: config.clone(),
            active_hypotheses: HashMap::new(),
            next_hypothesis_id: 1,
            current_frame: 0,
        };

        // Test that min_amplitude is respected
        let low_amplitude_detection = create_test_detection(10.0, 20.0, 70.0);
        let high_amplitude_detection = create_test_detection(10.0, 20.0, 90.0);

        let detections = vec![&low_amplitude_detection, &high_amplitude_detection];
        let valid_detections: Vec<&Detection> = detections
            .iter()
            .filter(|d| d.amplitude >= config.min_amplitude)
            .copied()
            .collect();

        assert_eq!(valid_detections.len(), 1); // Only high amplitude should pass
        assert_eq!(valid_detections[0].amplitude, 90.0);
    }
}
