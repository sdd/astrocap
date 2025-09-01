use std::collections::{HashMap, HashSet};

use crate::config::Config;
use astrocap_core::annotations::{AnnotationSession, ObjectType, TrackedObject};
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::structs::DetectedPoint;
use astrocap_core::traits::FrameSink;
use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use astrocap_processor_tracker::model::{Detection, Track};
use chrono::{DateTime, Utc};
use nalgebra::Vector2;
use serde::Serialize;
use toml::Value;
use tracing::{info, warn};

/// Association between a track and an annotated object
#[derive(Debug, Clone)]
struct TrackAnnotationAssociation {
    track_id: u64,
    object_id: u32,
    first_associated_frame: u32,
    association_distance: f32,
}

/// Information about annotation visibility and track associations
#[derive(Debug, Clone)]
struct AnnotationTrackInfo {
    object_id: u32,
    first_visible_frame: u32,
    associated_track_id: Option<u64>,
    association_delay: Option<u32>, // frames between visibility and track initiation
    missed_opportunity: bool,       // true if no track was ever associated
}

/// Performance metrics for unassociated tracks
#[derive(Debug, Clone)]
struct UnassociatedTrackInfo {
    track_id: u64,
    initial_position: Vector2<f32>,
    initial_velocity: Vector2<f32>,
    final_velocity: Vector2<f32>,
    velocity_match_score: f32, // how well velocity matches expected annotation velocity
    likely_true_positive: bool, // good velocity match suggests true but unannotated object
}

/// Evaluation results structures
#[derive(Serialize, Debug)]
struct EvaluationResults {
    pub metadata: EvaluationMetadata,
    pub frame_results: Vec<FrameResult>,
    pub aggregate_metrics: AggregateMetrics,
    pub per_object_metrics: Vec<ObjectMetrics>,
    pub initiator_performance: InitiatorPerformanceMetrics,
}

#[derive(Serialize, Debug)]
struct EvaluationMetadata {
    pub annotations_file: String,
    pub distance_threshold: f32,
    pub evaluated_at: DateTime<Utc>,
    pub frame_count: u32,
    pub annotation_velocity_stats: VelocityStats,
}

#[derive(Serialize, Debug, Clone)]
struct VelocityStats {
    pub mean_velocity: Vector2<f32>,
    pub std_dev_velocity: Vector2<f32>,
    pub velocity_magnitude_mean: f32,
    pub velocity_magnitude_std: f32,
}

#[derive(Clone, Serialize, Debug)]
struct FrameResult {
    pub frame_number: u32,
    pub processing_time_us: Option<u64>,
    pub tracks_associated: usize,
    pub tracks_unassociated: usize,
    pub annotations_with_tracks: usize,
    pub annotations_without_tracks: usize,
}

#[derive(Serialize, Debug)]
struct AggregateMetrics {
    pub frames_processed: u32,
    pub total_annotations: usize,
    pub total_tracks_created: usize,
    pub successful_associations: usize,
    pub duplicate_initiations: usize, // tracks initiated for already-tracked annotations
    pub missed_initiations: usize,    // annotations that never got tracks
    pub mean_association_delay: f32,  // average frames between annotation visibility and track
    pub unassociated_likely_true_positives: usize,
}

#[derive(Serialize, Debug)]
struct InitiatorPerformanceMetrics {
    pub responsiveness: ResponsivenessMetrics,
    pub accuracy: AccuracyMetrics,
    pub duplicate_prevention: DuplicatePreventionMetrics,
    pub velocity_analysis: VelocityAnalysisMetrics,
}

#[derive(Serialize, Debug)]
struct ResponsivenessMetrics {
    pub mean_initiation_delay: f32,
    pub median_initiation_delay: f32,
    pub max_initiation_delay: u32,
}

#[derive(Serialize, Debug)]
struct AccuracyMetrics {
    pub mean_position_error: f32,
    pub mean_velocity_error: f32,
    pub position_error_std: f32,
    pub velocity_error_std: f32,
}

#[derive(Serialize, Debug)]
struct DuplicatePreventionMetrics {
    pub duplicate_initiations_count: usize,
    pub duplicate_rate: f32, // duplicates / total annotations
}

#[derive(Serialize, Debug)]
struct VelocityAnalysisMetrics {
    pub unassociated_tracks_analyzed: usize,
    pub velocity_matched_tracks: usize,
    pub velocity_match_rate: f32,
    pub mean_velocity_match_score: f32,
}

#[derive(Clone, Serialize, Debug)]
struct ObjectMetrics {
    pub object_id: u32,
    pub associated_track_id: Option<u64>,
    pub initiation_delay: Option<u32>,
    pub position_accuracy: Option<f32>,
    pub velocity_accuracy: Option<f32>,
    pub duplicate_tracks: Vec<u64>,
}

pub struct TrackerEvaluatorSink {
    config: Config,
    annotations: AnnotationSession,
    frame_results: Vec<FrameResult>,

    // Track associations and performance tracking
    track_associations: HashMap<u64, TrackAnnotationAssociation>,
    annotation_track_info: HashMap<u32, AnnotationTrackInfo>,
    unassociated_tracks: Vec<UnassociatedTrackInfo>,

    // Velocity statistics for annotations
    annotation_velocities: Vec<Vector2<f32>>,
    velocity_stats: Option<VelocityStats>,

    // Frame tracking
    processed_frames: u32,
}

impl TrackerEvaluatorSink {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: Config = config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!(
                "TrackerEvaluatorSink:: {}",
                e.to_string()
            ))
        })?;

        // Load annotations
        let annotations = AnnotationSession::load_from_file(
            config.annotations_file.to_str().ok_or_else(|| {
                AstrocapError::PluginInvalidConfigError(
                    "Invalid UTF-8 in annotations file path".to_string(),
                )
            })?,
        )
        .map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!("Failed to load annotations: {}", e))
        })?;

        info!(
            "Loaded {} tracked objects from annotations",
            annotations.objects.len()
        );

        let mut evaluator = Self {
            config,
            annotations,
            frame_results: Vec::new(),
            track_associations: HashMap::new(),
            annotation_track_info: HashMap::new(),
            unassociated_tracks: Vec::new(),
            annotation_velocities: Vec::new(),
            velocity_stats: None,
            processed_frames: 0,
        };

        // Calculate annotation velocity statistics up-front
        evaluator.calculate_annotation_velocities();

        Ok(evaluator)
    }

    fn calculate_annotation_velocities(&mut self) {
        let mut velocities = Vec::new();

        for object in &self.annotations.objects {
            if object.keyframes.len() >= 2 {
                // Calculate velocity between consecutive keyframes
                for window in object.keyframes.windows(2) {
                    let dt = (window[1].frame_number - window[0].frame_number) as f32;
                    if dt > 0.0 {
                        let dx = window[1].x - window[0].x;
                        let dy = window[1].y - window[0].y;
                        let velocity = Vector2::new(dx / dt, dy / dt);
                        velocities.push(velocity);
                    }
                }
            }
        }

        if !velocities.is_empty() {
            // Calculate mean and standard deviation
            let mean_vx = velocities.iter().map(|v| v.x).sum::<f32>() / velocities.len() as f32;
            let mean_vy = velocities.iter().map(|v| v.y).sum::<f32>() / velocities.len() as f32;
            let mean_velocity = Vector2::new(mean_vx, mean_vy);

            let variance_vx = velocities
                .iter()
                .map(|v| (v.x - mean_vx).powi(2))
                .sum::<f32>()
                / velocities.len() as f32;
            let variance_vy = velocities
                .iter()
                .map(|v| (v.y - mean_vy).powi(2))
                .sum::<f32>()
                / velocities.len() as f32;
            let std_dev_velocity = Vector2::new(variance_vx.sqrt(), variance_vy.sqrt());

            // Calculate velocity magnitude statistics
            let magnitudes: Vec<f32> = velocities.iter().map(|v| v.norm()).collect();
            let mean_magnitude = magnitudes.iter().sum::<f32>() / magnitudes.len() as f32;
            let variance_magnitude = magnitudes
                .iter()
                .map(|m| (m - mean_magnitude).powi(2))
                .sum::<f32>()
                / magnitudes.len() as f32;
            let std_magnitude = variance_magnitude.sqrt();

            self.velocity_stats = Some(VelocityStats {
                mean_velocity,
                std_dev_velocity,
                velocity_magnitude_mean: mean_magnitude,
                velocity_magnitude_std: std_magnitude,
            });

            self.annotation_velocities = velocities;

            info!("Annotation velocity stats: mean=({:.3}, {:.3}), std=({:.3}, {:.3}), magnitude_mean={:.3}±{:.3}",
                  mean_velocity.x, mean_velocity.y, std_dev_velocity.x, std_dev_velocity.y,
                  mean_magnitude, std_magnitude);
        } else {
            warn!("No velocity data available from annotations");
        }
    }

    fn evaluate_frame(
        &mut self,
        frame_number: u32,
        ground_truth: &[(u32, f32, f32)],
        tracks: &[Track],
        processing_time_us: Option<u64>,
    ) -> FrameResult {
        // Associate tracks to annotations
        let (associated_tracks, unassociated_tracks) =
            self.associate_tracks_to_annotations(frame_number, tracks, ground_truth);

        // Update annotation tracking info
        self.update_annotation_tracking_info(frame_number, ground_truth, &associated_tracks);

        // Analyze unassociated tracks for potential true positives
        self.analyze_unassociated_tracks(frame_number, &unassociated_tracks);

        // Count annotations with and without tracks
        let annotations_with_tracks = associated_tracks.len();
        let annotations_without_tracks = ground_truth.len().saturating_sub(annotations_with_tracks);

        FrameResult {
            frame_number,
            processing_time_us,
            tracks_associated: associated_tracks.len(),
            tracks_unassociated: unassociated_tracks.len(),
            annotations_with_tracks,
            annotations_without_tracks,
        }
    }

    fn associate_tracks_to_annotations(
        &mut self,
        frame_number: u32,
        tracks: &[Track],
        ground_truth: &[(u32, f32, f32)],
    ) -> (Vec<(u64, u32, f32)>, Vec<u64>) {
        let mut associated = Vec::new();
        let mut unassociated = Vec::new();
        let mut used_annotations: HashSet<u32> = HashSet::new();

        for track in tracks {
            let track_pos = Vector2::new(track.state.state.x, track.state.state.y);
            let mut best_match: Option<(u32, f32)> = None;

            // Find closest annotation within threshold
            for &(object_id, x, y) in ground_truth {
                if used_annotations.contains(&object_id) {
                    continue; // Already associated to another track
                }

                let annotation_pos = Vector2::new(x, y);
                let distance = (track_pos - annotation_pos).norm();

                if distance <= self.config.distance_threshold {
                    if best_match.is_none() || distance < best_match.unwrap().1 {
                        best_match = Some((object_id, distance));
                    }
                }
            }

            match best_match {
                Some((object_id, distance)) => {
                    // Check if this is a new association
                    if !self.track_associations.contains_key(&track.id) {
                        // New track association
                        let association = TrackAnnotationAssociation {
                            track_id: track.id,
                            object_id,
                            first_associated_frame: frame_number,
                            association_distance: distance,
                        };
                        self.track_associations.insert(track.id, association);

                        // Check if this annotation already has a track (duplicate initiation)
                        if let Some(existing_info) = self.annotation_track_info.get(&object_id) {
                            if existing_info.associated_track_id.is_some() {
                                warn!(
                                    "Duplicate track initiation for object {} at frame {}",
                                    object_id, frame_number
                                );
                            }
                        }
                    }

                    associated.push((track.id, object_id, distance));
                    used_annotations.insert(object_id);
                }
                None => {
                    unassociated.push(track.id);
                }
            }
        }

        (associated, unassociated)
    }

    fn update_annotation_tracking_info(
        &mut self,
        frame_number: u32,
        ground_truth: &[(u32, f32, f32)],
        associated_tracks: &[(u64, u32, f32)],
    ) {
        // Update info for visible annotations
        for &(object_id, _, _) in ground_truth {
            let info = self
                .annotation_track_info
                .entry(object_id)
                .or_insert(AnnotationTrackInfo {
                    object_id,
                    first_visible_frame: frame_number,
                    associated_track_id: None,
                    association_delay: None,
                    missed_opportunity: false,
                });

            // Update first visible frame (should be the minimum)
            if frame_number < info.first_visible_frame {
                info.first_visible_frame = frame_number;
            }
        }

        // Update association info
        for &(track_id, object_id, _) in associated_tracks {
            if let Some(info) = self.annotation_track_info.get_mut(&object_id) {
                if info.associated_track_id.is_none() {
                    info.associated_track_id = Some(track_id);
                    info.association_delay =
                        Some(frame_number.saturating_sub(info.first_visible_frame));
                }
            }
        }
    }

    fn analyze_unassociated_tracks(&mut self, frame_number: u32, unassociated_track_ids: &[u64]) {
        if let Some(ref velocity_stats) = self.velocity_stats {
            for &track_id in unassociated_track_ids {
                // For simplicity, we'll analyze tracks that are old enough to have velocity estimates
                // In a real implementation, you'd track track history and calculate velocities

                // Placeholder values - in practice you'd get these from track history
                let initial_position = Vector2::new(0.0, 0.0); // From track creation
                let initial_velocity = Vector2::new(0.0, 0.0); // From track state
                let final_velocity = Vector2::new(0.0, 0.0); // Average velocity over track lifetime

                // Calculate how well the track velocity matches expected annotation velocity
                let velocity_error = (final_velocity - velocity_stats.mean_velocity).norm();
                let expected_std = velocity_stats.velocity_magnitude_std;
                let velocity_match_score = 1.0 / (1.0 + velocity_error / expected_std.max(0.008));

                let likely_true_positive = velocity_match_score > 0.7; // Configurable threshold

                if likely_true_positive {
                    info!("Unassociated track {} has good velocity match (score: {:.3}) - likely true positive",
                          track_id, velocity_match_score);
                }

                let unassociated_info = UnassociatedTrackInfo {
                    track_id,
                    initial_position,
                    initial_velocity,
                    final_velocity,
                    velocity_match_score,
                    likely_true_positive,
                };

                self.unassociated_tracks.push(unassociated_info);
            }
        }
    }

    fn calculate_aggregate_metrics(&self) -> AggregateMetrics {
        let total_annotations = self.annotation_track_info.len();
        let successful_associations = self
            .annotation_track_info
            .values()
            .filter(|info| info.associated_track_id.is_some())
            .count();
        let missed_initiations = total_annotations - successful_associations;

        // Calculate duplicate initiations
        let mut object_track_counts: HashMap<u32, usize> = HashMap::new();
        for association in self.track_associations.values() {
            *object_track_counts
                .entry(association.object_id)
                .or_insert(0) += 1;
        }
        let duplicate_initiations = object_track_counts
            .values()
            .filter(|&&count| count > 1)
            .map(|&count| count - 1) // Count extras only
            .sum();

        // Calculate mean association delay
        let delays: Vec<u32> = self
            .annotation_track_info
            .values()
            .filter_map(|info| info.association_delay)
            .collect();
        let mean_association_delay = if delays.is_empty() {
            0.0
        } else {
            delays.iter().sum::<u32>() as f32 / delays.len() as f32
        };

        let unassociated_likely_true_positives = self
            .unassociated_tracks
            .iter()
            .filter(|info| info.likely_true_positive)
            .count();

        AggregateMetrics {
            frames_processed: self.processed_frames,
            total_annotations,
            total_tracks_created: self.track_associations.len(),
            successful_associations,
            duplicate_initiations,
            missed_initiations,
            mean_association_delay,
            unassociated_likely_true_positives,
        }
    }

    fn calculate_initiator_performance(&self) -> InitiatorPerformanceMetrics {
        // Responsiveness metrics
        let delays: Vec<u32> = self
            .annotation_track_info
            .values()
            .filter_map(|info| info.association_delay)
            .collect();

        let (mean_delay, median_delay, max_delay) = if delays.is_empty() {
            (0.0, 0.0, 0)
        } else {
            let mean = delays.iter().sum::<u32>() as f32 / delays.len() as f32;
            let mut sorted_delays = delays.clone();
            sorted_delays.sort();
            let median = sorted_delays[sorted_delays.len() / 2] as f32;
            let max = *sorted_delays.last().unwrap();
            (mean, median, max)
        };

        // Accuracy metrics (placeholder - would need track history)
        let accuracy = AccuracyMetrics {
            mean_position_error: 0.0,
            mean_velocity_error: 0.0,
            position_error_std: 0.0,
            velocity_error_std: 0.0,
        };

        // Duplicate prevention metrics
        let total_annotations = self.annotation_track_info.len().max(1);
        let duplicate_prevention = DuplicatePreventionMetrics {
            duplicate_initiations_count: self.calculate_aggregate_metrics().duplicate_initiations,
            duplicate_rate: self.calculate_aggregate_metrics().duplicate_initiations as f32
                / total_annotations as f32,
        };

        // Velocity analysis metrics
        let velocity_matched = self
            .unassociated_tracks
            .iter()
            .filter(|info| info.likely_true_positive)
            .count();
        let total_unassociated = self.unassociated_tracks.len().max(1);
        let mean_match_score = if self.unassociated_tracks.is_empty() {
            0.0
        } else {
            self.unassociated_tracks
                .iter()
                .map(|info| info.velocity_match_score)
                .sum::<f32>()
                / self.unassociated_tracks.len() as f32
        };

        let velocity_analysis = VelocityAnalysisMetrics {
            unassociated_tracks_analyzed: self.unassociated_tracks.len(),
            velocity_matched_tracks: velocity_matched,
            velocity_match_rate: velocity_matched as f32 / total_unassociated as f32,
            mean_velocity_match_score: mean_match_score,
        };

        InitiatorPerformanceMetrics {
            responsiveness: ResponsivenessMetrics {
                mean_initiation_delay: mean_delay,
                median_initiation_delay: median_delay,
                max_initiation_delay: max_delay,
            },
            accuracy,
            duplicate_prevention,
            velocity_analysis,
        }
    }

    fn save_results(&self) {
        let aggregate_metrics = self.calculate_aggregate_metrics();
        let initiator_performance = self.calculate_initiator_performance();

        let per_object_metrics: Vec<ObjectMetrics> = self
            .annotation_track_info
            .values()
            .map(|info| {
                ObjectMetrics {
                    object_id: info.object_id,
                    associated_track_id: info.associated_track_id,
                    initiation_delay: info.association_delay,
                    position_accuracy: None, // Would calculate from track history
                    velocity_accuracy: None, // Would calculate from track history
                    duplicate_tracks: Vec::new(), // Would track multiple associations
                }
            })
            .collect();

        let results = EvaluationResults {
            metadata: EvaluationMetadata {
                annotations_file: self.config.annotations_file.to_string_lossy().to_string(),
                distance_threshold: self.config.distance_threshold,
                evaluated_at: Utc::now(),
                frame_count: self.processed_frames,
                annotation_velocity_stats: self.velocity_stats.clone().unwrap_or(VelocityStats {
                    mean_velocity: Vector2::zeros(),
                    std_dev_velocity: Vector2::zeros(),
                    velocity_magnitude_mean: 0.0,
                    velocity_magnitude_std: 0.0,
                }),
            },
            frame_results: self.frame_results.clone(),
            aggregate_metrics,
            per_object_metrics,
            initiator_performance,
        };

        // Print comprehensive summary
        println!("\n=== TRACKER INITIATOR EVALUATION SUMMARY ===");
        println!(
            "Annotation velocity: mean=({:.3}, {:.3}) px/frame, magnitude={:.3}±{:.3} px/frame",
            results.metadata.annotation_velocity_stats.mean_velocity.x,
            results.metadata.annotation_velocity_stats.mean_velocity.y,
            results
                .metadata
                .annotation_velocity_stats
                .velocity_magnitude_mean,
            results
                .metadata
                .annotation_velocity_stats
                .velocity_magnitude_std
        );

        println!("\nOverall Performance:");
        println!(
            "  Total annotations: {}",
            results.aggregate_metrics.total_annotations
        );
        println!(
            "  Successful associations: {}",
            results.aggregate_metrics.successful_associations
        );
        println!(
            "  Missed initiations: {}",
            results.aggregate_metrics.missed_initiations
        );
        println!(
            "  Duplicate initiations: {}",
            results.aggregate_metrics.duplicate_initiations
        );
        println!(
            "  Mean association delay: {:.1} frames",
            results.aggregate_metrics.mean_association_delay
        );

        println!("\nInitiator Responsiveness:");
        println!(
            "  Mean initiation delay: {:.1} frames",
            results
                .initiator_performance
                .responsiveness
                .mean_initiation_delay
        );
        println!(
            "  Median initiation delay: {:.1} frames",
            results
                .initiator_performance
                .responsiveness
                .median_initiation_delay
        );
        println!(
            "  Max initiation delay: {} frames",
            results
                .initiator_performance
                .responsiveness
                .max_initiation_delay
        );

        println!("\nUnassociated Track Analysis:");
        println!(
            "  Unassociated tracks analyzed: {}",
            results
                .initiator_performance
                .velocity_analysis
                .unassociated_tracks_analyzed
        );
        println!(
            "  Velocity-matched tracks (likely true positives): {}",
            results
                .initiator_performance
                .velocity_analysis
                .velocity_matched_tracks
        );
        println!(
            "  Velocity match rate: {:.1}%",
            results
                .initiator_performance
                .velocity_analysis
                .velocity_match_rate
                * 100.0
        );
        println!(
            "  Mean velocity match score: {:.3}",
            results
                .initiator_performance
                .velocity_analysis
                .mean_velocity_match_score
        );

        println!("\nDuplicate Prevention:");
        println!(
            "  Duplicate rate: {:.1}%",
            results
                .initiator_performance
                .duplicate_prevention
                .duplicate_rate
                * 100.0
        );

        // Save detailed results to file
        let output_path = self
            .config
            .output_path
            .as_path()
            .to_string_lossy()
            .to_string();

        match serde_json::to_string_pretty(&results) {
            Ok(json_output) => match std::fs::write(&output_path, json_output) {
                Ok(_) => info!("Detailed results saved to: {}", output_path),
                Err(e) => warn!("Failed to save results to {}: {}", output_path, e),
            },
            Err(e) => warn!("Failed to serialize results: {}", e),
        }
    }
}

impl FrameSink for TrackerEvaluatorSink {
    fn consume(&mut self, frame_ctx: &mut FrameContext, _ctx: &mut PipelineContext) -> () {
        let frame_number = frame_ctx.frame_index as u32;
        self.processed_frames = frame_number + 1;

        // Get ground truth for this frame
        let ground_truth = self.annotations.objects_at_frame(frame_number);

        let Ok(tracks) = frame_ctx.try_get_as::<Vec<Track>>("tracker/tracks") else {
            warn!(
                "No detected tracks metadata found for frame {}",
                frame_number
            );
            return;
        };

        // Get processing time if available
        let processing_time_us = frame_ctx
            .metadata
            .get("processing_time_us")
            .and_then(|pt| pt.downcast_ref::<u64>())
            .copied();

        // Evaluate this frame
        let frame_result =
            self.evaluate_frame(frame_number, &ground_truth, &tracks, processing_time_us);

        // Log progress periodically
        if frame_number % 100 == 0 {
            info!(
                "Frame {}: {} tracks, {} annotations, {} associated",
                frame_number,
                tracks.len(),
                ground_truth.len(),
                frame_result.tracks_associated
            );
        }

        self.frame_results.push(frame_result);
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        info!("Finalizing tracker evaluation...");
        self.save_results();

        Ok(())
    }

    fn name(&self) -> &str {
        "tracker_evaluator"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
