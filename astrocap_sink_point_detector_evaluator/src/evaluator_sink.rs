use std::collections::HashMap;

use crate::config::Config;
use astrocap_core::annotations::{AnnotationSession, ObjectType, TrackedObject};
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::structs::Detection;
use astrocap_core::traits::FrameSink;
use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use chrono::{DateTime, Utc};
use serde::Serialize;
use toml::Value;
use tracing::{info, warn};

/// Evaluation results structures
#[derive(Serialize, Debug)]
struct EvaluationResults {
    pub metadata: EvaluationMetadata,
    pub frame_results: Vec<FrameResult>,
    pub aggregate_metrics: AggregateMetrics,
    pub per_object_metrics: Vec<ObjectMetrics>,
    pub amplitude_analysis: AmplitudeAnalysis,
}

#[derive(Serialize, Debug)]
struct EvaluationMetadata {
    pub annotations_file: String,
    pub distance_threshold: f32,
    pub evaluated_at: DateTime<Utc>,
    pub frame_count: u32,
}

#[derive(Clone, Serialize, Debug)]
struct FrameResult {
    pub frame_number: u32,
    pub ground_truth_count: u32,
    pub detections_count: u32,
    pub true_positives: u32,
    pub false_positives: u32,
    pub false_negatives: u32,
    pub processing_time_us: Option<u64>,
    pub amplitude_stats: FrameAmplitudeStats,
}

#[derive(Clone, Serialize, Debug)]
struct FrameAmplitudeStats {
    pub true_positive_amplitudes: Vec<f32>,
    pub false_positive_amplitudes: Vec<f32>,
    pub missed_object_info: Vec<MissedObjectInfo>,
}

#[derive(Clone, Serialize, Debug)]
struct MissedObjectInfo {
    pub object_id: u32,
    pub object_name: String,
    pub nearby_false_positive_count: u32,
    pub nearby_false_positive_amplitudes: Vec<f32>,
}

#[derive(Serialize, Debug)]
struct AggregateMetrics {
    pub total_ground_truth: u32,
    pub total_detections: u32,
    pub total_true_positives: u32,
    pub total_false_positives: u32,
    pub total_false_negatives: u32,
    pub precision: f32,
    pub recall: f32,
    pub f1_score: f32,
    pub frames_processed: u32,
}

#[derive(Clone, Serialize, Debug)]
struct ObjectMetrics {
    pub object_id: u32,
    pub object_name: String,
    pub object_type: ObjectType,
    pub frames_present: u32,
    pub detections_matched: u32,
    pub detection_rate: f32,
    pub frame_range: (u32, u32),
    pub avg_detection_distance: Option<f32>,
    pub avg_detection_amplitude: Option<f32>,
    pub amplitude_range: Option<(f32, f32)>,
    pub consecutive_detection_counts: HashMap<usize, usize>,
    pub consecutive_miss_counts: HashMap<usize, usize>,
}

#[derive(Serialize, Debug)]
struct AmplitudeAnalysis {
    pub true_positive_amplitude_stats: AmplitudeStats,
    pub false_positive_amplitude_stats: AmplitudeStats,
    pub amplitude_threshold_suggestions: Vec<ThresholdSuggestion>,
    pub per_object_amplitude_analysis: Vec<ObjectAmplitudeAnalysis>,
}

#[derive(Clone, Serialize, Debug)]
struct AmplitudeStats {
    pub mean: f32,
    pub median: f32,
    pub min: f32,
    pub max: f32,
    pub std_dev: f32,
    pub percentile_25: f32,
    pub percentile_75: f32,
    pub count: u32,
}

#[derive(Clone, Serialize, Debug)]
struct ThresholdSuggestion {
    pub threshold: f32,
    pub expected_precision: f32,
    pub expected_recall: f32,
    pub expected_f1: f32,
    pub tp_count: u32,
    pub fp_count: u32,
}

#[derive(Clone, Serialize, Debug)]
struct ObjectAmplitudeAnalysis {
    pub object_id: u32,
    pub object_name: String,
    pub detection_amplitude_stats: Option<AmplitudeStats>,
    pub missed_frames_with_brighter_fps: u32,
    pub total_missed_frames: u32,
    pub brightest_nearby_fp_amplitude: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct KalmanQAnalysis {
    pub position_variance_x: f32,
    pub position_variance_y: f32,
    pub velocity_variance_x: f32,
    pub velocity_variance_y: f32,
    pub position_velocity_covariance: f32,
    pub samples_used: usize,
    pub stars_included: usize,
}

#[derive(Debug)]
struct MotionNoiseStats {
    position_deltas_x: Vec<f32>,
    position_deltas_y: Vec<f32>,
    velocity_deltas_x: Vec<f32>,
    velocity_deltas_y: Vec<f32>,
}

fn calculate_variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32
}

fn calculate_covariance(x_values: &[f32], y_values: &[f32]) -> f32 {
    if x_values.len() != y_values.len() || x_values.is_empty() {
        return 0.0;
    }
    let x_mean = x_values.iter().sum::<f32>() / x_values.len() as f32;
    let y_mean = y_values.iter().sum::<f32>() / y_values.len() as f32;

    x_values
        .iter()
        .zip(y_values)
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f32>()
        / x_values.len() as f32
}

/// Internal tracking structure for per-object statistics
#[derive(Debug, Clone)]
struct ObjectTracker {
    pub object_id: u32,
    pub frames_present: u32,
    pub detections_matched: u32,
    pub total_distance: f32,
    pub distance_count: u32,
    pub detection_amplitudes: Vec<f32>,
    pub missed_frames_with_brighter_fps: u32,
    pub brightest_nearby_fp_amplitude: Option<f32>,
    pub consecutive_detections: Vec<usize>,
    pub consecutive_misses: Vec<usize>,
    pub current_detection_streak: usize,
    pub current_miss_streak: usize,
    pub last_frame_detected: Option<bool>,
}

impl ObjectTracker {
    fn new(object_id: u32) -> Self {
        Self {
            object_id,
            frames_present: 0,
            detections_matched: 0,
            total_distance: 0.0,
            distance_count: 0,
            detection_amplitudes: Vec::new(),
            missed_frames_with_brighter_fps: 0,
            brightest_nearby_fp_amplitude: None,
            consecutive_detections: Vec::new(),
            consecutive_misses: Vec::new(),
            current_detection_streak: 0,
            current_miss_streak: 0,
            last_frame_detected: None,
        }
    }

    fn add_frame(&mut self, detected: bool, distance: Option<f32>, amplitude: Option<f32>) {
        self.frames_present += 1;

        // Update consecutive detection/miss tracking
        match self.last_frame_detected {
            None => {
                // First frame for this object
                if detected {
                    self.current_detection_streak = 1;
                    self.current_miss_streak = 0;
                } else {
                    self.current_detection_streak = 0;
                    self.current_miss_streak = 1;
                }
            }
            Some(last_detected) => {
                if detected {
                    if last_detected {
                        // Continue detection streak
                        self.current_detection_streak += 1;
                    } else {
                        // End miss streak, start detection streak
                        if self.current_miss_streak > 0 {
                            self.consecutive_misses.push(self.current_miss_streak);
                        }
                        self.current_detection_streak = 1;
                        self.current_miss_streak = 0;
                    }
                } else {
                    if !last_detected {
                        // Continue miss streak
                        self.current_miss_streak += 1;
                    } else {
                        // End detection streak, start miss streak
                        if self.current_detection_streak > 0 {
                            self.consecutive_detections
                                .push(self.current_detection_streak);
                        }
                        self.current_detection_streak = 0;
                        self.current_miss_streak = 1;
                    }
                }
            }
        }

        self.last_frame_detected = Some(detected);

        if detected {
            self.detections_matched += 1;
            if let Some(dist) = distance {
                self.total_distance += dist;
                self.distance_count += 1;
            }
            if let Some(amp) = amplitude {
                self.detection_amplitudes.push(amp);
            }
        }
    }

    fn finalize_streaks(&mut self) {
        // Finalize any ongoing streaks when evaluation is complete
        if self.current_detection_streak > 0 {
            self.consecutive_detections
                .push(self.current_detection_streak);
        }
        if self.current_miss_streak > 0 {
            self.consecutive_misses.push(self.current_miss_streak);
        }
    }

    fn get_consecutive_detection_counts(&self) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();
        for &streak_length in &self.consecutive_detections {
            *counts.entry(streak_length).or_insert(0) += 1;
        }
        counts
    }

    fn get_consecutive_miss_counts(&self) -> HashMap<usize, usize> {
        let mut counts = HashMap::new();
        for &streak_length in &self.consecutive_misses {
            *counts.entry(streak_length).or_insert(0) += 1;
        }
        counts
    }

    fn add_missed_frame_info(
        &mut self,
        has_brighter_fps: bool,
        brightest_fp_amplitude: Option<f32>,
    ) {
        if has_brighter_fps {
            self.missed_frames_with_brighter_fps += 1;
        }
        if let Some(amp) = brightest_fp_amplitude {
            match self.brightest_nearby_fp_amplitude {
                None => self.brightest_nearby_fp_amplitude = Some(amp),
                Some(existing) => {
                    if amp > existing {
                        self.brightest_nearby_fp_amplitude = Some(amp);
                    }
                }
            }
        }
    }

    fn detection_rate(&self) -> f32 {
        if self.frames_present == 0 {
            0.0
        } else {
            self.detections_matched as f32 / self.frames_present as f32
        }
    }

    fn avg_detection_distance(&self) -> Option<f32> {
        if self.distance_count > 0 {
            Some(self.total_distance / self.distance_count as f32)
        } else {
            None
        }
    }

    fn avg_detection_amplitude(&self) -> Option<f32> {
        if self.detection_amplitudes.is_empty() {
            None
        } else {
            Some(
                self.detection_amplitudes.iter().sum::<f32>()
                    / self.detection_amplitudes.len() as f32,
            )
        }
    }

    fn amplitude_range(&self) -> Option<(f32, f32)> {
        if self.detection_amplitudes.is_empty() {
            None
        } else {
            let min = self
                .detection_amplitudes
                .iter()
                .cloned()
                .fold(f32::INFINITY, f32::min);
            let max = self
                .detection_amplitudes
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            Some((min, max))
        }
    }
}

pub struct PointDetectorEvaluatorSink {
    config: Config,
    annotations: AnnotationSession,
    frame_results: Vec<FrameResult>,
    object_trackers: HashMap<u32, ObjectTracker>,
    all_true_positive_amplitudes: Vec<f32>,
    all_false_positive_amplitudes: Vec<f32>,
    frame_detections: Vec<(u32, Vec<Detection>)>,
}

impl PointDetectorEvaluatorSink {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: Config = config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!("MedianProcessor:: {}", e.to_string()))
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

        // Initialize object trackers
        let mut object_trackers = HashMap::new();
        for obj in &annotations.objects {
            object_trackers.insert(obj.id, ObjectTracker::new(obj.id));
        }

        Ok(Self {
            config,
            annotations,
            frame_results: Vec::new(),
            object_trackers,
            all_true_positive_amplitudes: Vec::new(),
            all_false_positive_amplitudes: Vec::new(),
            frame_detections: Vec::new(),
        })
    }

    /// Evaluate detections against ground truth for a single frame
    fn evaluate_frame(
        &mut self,
        frame_number: u32,
        ground_truth: &[(u32, f32, f32)],
        detections: &[Detection],
        processing_time_us: Option<u64>,
    ) -> FrameResult {
        self.frame_detections
            .push((frame_number, detections.to_vec()));

        let mut true_positives = 0u32;
        let mut matched_detections = vec![false; detections.len()];
        let mut object_matches = HashMap::new();
        let mut true_positive_amplitudes = Vec::new();

        // For each ground truth point, find the closest detection within threshold
        for (gt_id, gt_x, gt_y) in ground_truth {
            let mut best_match_idx: Option<usize> = None;
            let mut best_distance = self.config.distance_threshold;

            for (det_idx, detected_point) in detections.iter().enumerate() {
                if matched_detections[det_idx] {
                    continue; // Already matched
                }

                let distance = ((gt_x - (detected_point.position[0])).powi(2)
                    + (gt_y - (detected_point.position[1])).powi(2))
                .sqrt();
                if distance < best_distance {
                    best_distance = distance;
                    best_match_idx = Some(det_idx);
                }
            }

            if let Some(idx) = best_match_idx {
                matched_detections[idx] = true;
                true_positives += 1;
                let amplitude = detections[idx].amplitude;
                object_matches.insert(*gt_id, (best_distance, amplitude));
                true_positive_amplitudes.push(amplitude);
                self.all_true_positive_amplitudes.push(amplitude);
            }
        }

        // Collect false positive amplitudes and analyze missed objects
        let mut false_positive_amplitudes = Vec::new();
        let mut missed_object_info = Vec::new();

        for (det_idx, detected_point) in detections.iter().enumerate() {
            if !matched_detections[det_idx] {
                false_positive_amplitudes.push(detected_point.amplitude);
                self.all_false_positive_amplitudes
                    .push(detected_point.amplitude);
            }
        }

        // Analyze missed objects - check if there are brighter false positives nearby
        for (gt_id, gt_x, gt_y) in ground_truth {
            if !object_matches.contains_key(gt_id) {
                let nearby_fps: Vec<f32> = detections
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| !matched_detections[*idx])
                    .map(|(_, det)| {
                        (
                            det,
                            ((gt_x - det.position.x).powi(2) + (gt_y - det.position.y).powi(2))
                                .sqrt(),
                        )
                    })
                    .filter(|(_, distance)| *distance <= self.config.distance_threshold * 2.0) // Search in wider radius
                    .map(|(det, _)| det.amplitude)
                    .collect();

                let has_brighter_fps = !nearby_fps.is_empty();
                let brightest_nearby_fp = nearby_fps.iter().fold(None, |acc, &x| match acc {
                    None => Some(x),
                    Some(current_max) => Some(current_max.max(x)),
                });

                if let Some(obj_name) = self
                    .annotations
                    .objects
                    .iter()
                    .find(|obj| obj.id == *gt_id)
                    .map(|obj| &obj.name)
                {
                    missed_object_info.push(MissedObjectInfo {
                        object_id: *gt_id,
                        object_name: obj_name.clone(),
                        nearby_false_positive_count: nearby_fps.len() as u32,
                        nearby_false_positive_amplitudes: nearby_fps.clone(),
                    });

                    // Update object tracker with missed frame info
                    if let Some(tracker) = self.object_trackers.get_mut(gt_id) {
                        tracker.add_missed_frame_info(has_brighter_fps, brightest_nearby_fp);
                    }
                }
            }
        }

        // Update object trackers
        for (gt_id, _gt_x, _gt_y) in ground_truth {
            if let Some(tracker) = self.object_trackers.get_mut(gt_id) {
                if let Some((distance, amplitude)) = object_matches.get(gt_id) {
                    tracker.add_frame(true, Some(*distance), Some(*amplitude));
                } else {
                    tracker.add_frame(false, None, None);
                }
            }
        }

        let false_positives = matched_detections
            .iter()
            .filter(|&&matched| !matched)
            .count() as u32;
        let false_negatives = ground_truth.len() as u32 - true_positives;

        FrameResult {
            frame_number,
            ground_truth_count: ground_truth.len() as u32,
            detections_count: detections.len() as u32,
            true_positives,
            false_positives,
            false_negatives,
            processing_time_us,
            amplitude_stats: FrameAmplitudeStats {
                true_positive_amplitudes,
                false_positive_amplitudes,
                missed_object_info,
            },
        }
    }

    /// Calculate amplitude statistics from a vector of amplitudes
    fn calculate_amplitude_stats(&self, amplitudes: &[f32]) -> AmplitudeStats {
        if amplitudes.is_empty() {
            return AmplitudeStats {
                mean: 0.0,
                median: 0.0,
                min: 0.0,
                max: 0.0,
                std_dev: 0.0,
                percentile_25: 0.0,
                percentile_75: 0.0,
                count: 0,
            };
        }

        let mut sorted_amplitudes = amplitudes.to_vec();
        sorted_amplitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count = amplitudes.len();
        let sum: f32 = amplitudes.iter().sum();
        let mean = sum / count as f32;

        let variance = amplitudes.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();

        let median = sorted_amplitudes[count / 2];
        let percentile_25 = sorted_amplitudes[count / 4];
        let percentile_75 = sorted_amplitudes[3 * count / 4];

        AmplitudeStats {
            mean,
            median,
            min: sorted_amplitudes[0],
            max: sorted_amplitudes[count - 1],
            std_dev,
            percentile_25,
            percentile_75,
            count: count as u32,
        }
    }

    /// Generate threshold suggestions based on ACTUAL detection pipeline behavior
    /// This should predict changes to point_threshold, not amplitude filtering
    fn generate_threshold_suggestions(&self) -> Vec<ThresholdSuggestion> {
        let mut suggestions = Vec::new();

        // Instead of amplitude-based filtering, we need to consider:
        // 1. The local maxima detection threshold (point_threshold in config)
        // 2. The complex star scoring that follows
        // 3. The model state filtering

        warn!("Threshold suggestions currently use simplified amplitude-only analysis");
        warn!("This does not reflect the actual detection pipeline behavior");
        warn!("Suggestions may not be accurate for point_threshold optimization");

        // For now, return empty suggestions or mark them as "preliminary"
        for threshold_percentile in [50, 60, 70, 80, 90] {
            let threshold_idx = (self.all_true_positive_amplitudes.len() * threshold_percentile
                / 100)
                .min(self.all_true_positive_amplitudes.len() - 1);

            if threshold_idx < self.all_true_positive_amplitudes.len() {
                let threshold = self.all_true_positive_amplitudes[threshold_idx];

                suggestions.push(ThresholdSuggestion {
                    threshold,
                    expected_precision: 0.0, // Mark as invalid
                    expected_recall: 0.0,    // Mark as invalid
                    expected_f1: 0.0,        // Mark as invalid
                    tp_count: 0,
                    fp_count: 0,
                });
            }
        }

        suggestions
    }

    /// Calculate aggregate metrics from all frame results
    fn calculate_aggregate_metrics(&self) -> AggregateMetrics {
        let total_ground_truth: u32 = self
            .frame_results
            .iter()
            .map(|fr| fr.ground_truth_count)
            .sum();
        let total_detections: u32 = self
            .frame_results
            .iter()
            .map(|fr| fr.detections_count)
            .sum();
        let total_true_positives: u32 = self.frame_results.iter().map(|fr| fr.true_positives).sum();
        let total_false_positives: u32 =
            self.frame_results.iter().map(|fr| fr.false_positives).sum();
        let total_false_negatives: u32 =
            self.frame_results.iter().map(|fr| fr.false_negatives).sum();

        let precision = if total_detections > 0 {
            total_true_positives as f32 / total_detections as f32
        } else {
            0.0
        };

        let recall = if total_ground_truth > 0 {
            total_true_positives as f32 / total_ground_truth as f32
        } else {
            0.0
        };

        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        AggregateMetrics {
            total_ground_truth,
            total_detections,
            total_true_positives,
            total_false_positives,
            total_false_negatives,
            precision,
            recall,
            f1_score,
            frames_processed: self.frame_results.len() as u32,
        }
    }

    /// Calculate per-object metrics
    fn calculate_per_object_metrics(&self) -> Vec<ObjectMetrics> {
        let mut metrics = Vec::new();

        for (object_id, mut tracker) in self.object_trackers.clone() {
            // Finalize any ongoing streaks
            tracker.finalize_streaks();

            // Find the frame range by looking through the annotations for this object
            let frame_range = self
                .annotations
                .objects
                .iter()
                .find(|obj| obj.id == object_id)
                .map(|obj| {
                    if obj.keyframes.is_empty() {
                        (0, 0)
                    } else {
                        let frame_numbers: Vec<u32> =
                            obj.keyframes.iter().map(|kf| kf.frame_number).collect();
                        let min_frame = *frame_numbers.iter().min().unwrap_or(&0);
                        let max_frame = *frame_numbers.iter().max().unwrap_or(&0);
                        (min_frame, max_frame)
                    }
                })
                .unwrap_or((0, 0));

            let object_name = self
                .annotations
                .objects
                .iter()
                .find(|obj| obj.id == object_id)
                .map(|obj| obj.name.clone())
                .unwrap_or_else(|| format!("Object_{}", object_id));

            let object_type = self
                .annotations
                .objects
                .iter()
                .find(|obj| obj.id == object_id)
                .map(|obj| obj.object_type.clone())
                .unwrap_or(ObjectType::Unknown);

            metrics.push(ObjectMetrics {
                object_id,
                object_name,
                object_type,
                frames_present: tracker.frames_present,
                detections_matched: tracker.detections_matched,
                detection_rate: tracker.detection_rate(),
                frame_range,
                avg_detection_distance: tracker.avg_detection_distance(),
                avg_detection_amplitude: tracker.avg_detection_amplitude(),
                amplitude_range: tracker.amplitude_range(),
                consecutive_detection_counts: tracker.get_consecutive_detection_counts(),
                consecutive_miss_counts: tracker.get_consecutive_miss_counts(),
            });
        }

        metrics.sort_by_key(|m| m.object_id);
        metrics
    }

    /// Calculate per-object amplitude analysis
    fn calculate_per_object_amplitude_analysis(&self) -> Vec<ObjectAmplitudeAnalysis> {
        let mut analysis = Vec::new();

        for obj in &self.annotations.objects {
            let tracker = &self.object_trackers[&obj.id];

            let detection_amplitude_stats = if !tracker.detection_amplitudes.is_empty() {
                Some(self.calculate_amplitude_stats(&tracker.detection_amplitudes))
            } else {
                None
            };

            analysis.push(ObjectAmplitudeAnalysis {
                object_id: obj.id,
                object_name: obj.name.clone(),
                detection_amplitude_stats,
                missed_frames_with_brighter_fps: tracker.missed_frames_with_brighter_fps,
                total_missed_frames: tracker.frames_present - tracker.detections_matched,
                brightest_nearby_fp_amplitude: tracker.brightest_nearby_fp_amplitude,
            });
        }

        analysis.sort_by_key(|a| a.missed_frames_with_brighter_fps);
        analysis.reverse(); // Most problematic first

        analysis
    }

    /// Save evaluation results to JSON file
    fn save_results(&self) -> Result<(), AstrocapError> {
        let aggregate_metrics = self.calculate_aggregate_metrics();
        let per_object_metrics = self.calculate_per_object_metrics();
        let threshold_suggestions = self.generate_threshold_suggestions();
        let per_object_amplitude_analysis = self.calculate_per_object_amplitude_analysis();

        let amplitude_analysis = AmplitudeAnalysis {
            true_positive_amplitude_stats: self
                .calculate_amplitude_stats(&self.all_true_positive_amplitudes),
            false_positive_amplitude_stats: self
                .calculate_amplitude_stats(&self.all_false_positive_amplitudes),
            amplitude_threshold_suggestions: threshold_suggestions.clone(),
            per_object_amplitude_analysis: per_object_amplitude_analysis.clone(),
        };

        let results = EvaluationResults {
            metadata: EvaluationMetadata {
                annotations_file: self.config.annotations_file.to_string_lossy().to_string(),
                distance_threshold: self.config.distance_threshold,
                evaluated_at: Utc::now(),
                frame_count: self.frame_results.len() as u32,
            },
            frame_results: self.frame_results.clone(),
            aggregate_metrics,
            per_object_metrics: per_object_metrics.clone(),
            amplitude_analysis,
        };

        let results_json = serde_json::to_string_pretty(&results).map_err(|e| {
            AstrocapError::GeneralPluginError(format!("Failed to serialize results: {}", e))
        })?;

        std::fs::write(&self.config.output_path, results_json).map_err(|e| {
            AstrocapError::GeneralPluginError(format!("Failed to write results file: {}", e))
        })?;

        info!("Evaluation results saved to: {:?}", self.config.output_path);

        // Print enhanced summary to console
        self.print_console_summary(
            &results.aggregate_metrics,
            &per_object_metrics,
            &results.amplitude_analysis,
        );

        Ok(())
    }

    fn print_console_summary(
        &self,
        aggregate: &AggregateMetrics,
        per_object: &[ObjectMetrics],
        amplitude_analysis: &AmplitudeAnalysis,
    ) {
        println!("\n=== Point Detector Evaluation Results ===");
        println!(
            "Distance threshold: {:.1} pixels",
            self.config.distance_threshold
        );
        println!("Frames processed: {}", aggregate.frames_processed);
        println!();

        println!("=== Aggregate Metrics ===");
        println!(
            "Total ground truth objects: {}",
            aggregate.total_ground_truth
        );
        println!("Total detections: {}", aggregate.total_detections);
        println!("True positives: {}", aggregate.total_true_positives);
        println!("False positives: {}", aggregate.total_false_positives);
        println!("False negatives: {}", aggregate.total_false_negatives);
        println!("Precision: {:.3}", aggregate.precision);
        println!("Recall: {:.3}", aggregate.recall);
        println!("F1 Score: {:.3}", aggregate.f1_score);
        println!();

        println!("=== Per-Object Analysis ({} objects) ===", per_object.len());
        for obj in per_object {
            println!(
                "Object {} ({}): {}/{} frames detected ({:.1}%)",
                obj.object_id,
                obj.object_name,
                obj.detections_matched,
                obj.frames_present,
                obj.detection_rate * 100.0
            );

            if let Some(avg_dist) = obj.avg_detection_distance {
                println!("  Average detection distance: {:.2} pixels", avg_dist);
            }

            if let Some(avg_amp) = obj.avg_detection_amplitude {
                println!("  Average detection amplitude: {:.1}", avg_amp);
            }

            if let Some((min_amp, max_amp)) = obj.amplitude_range {
                println!("  Amplitude range: {:.1} - {:.1}", min_amp, max_amp);
            }

            // Print consecutive detection statistics
            // if !obj.consecutive_detection_counts.is_empty() {
            //     print!("  Consecutive detections: ");
            //     let mut detection_pairs: Vec<_> = obj.consecutive_detection_counts.iter().collect();
            //     detection_pairs.sort_by_key(|(length, _)| *length);
            //     for (i, (length, count)) in detection_pairs.iter().enumerate() {
            //         if i > 0 {
            //             print!(", ");
            //         }
            //         print!("{}×{}", length, count);
            //     }
            //     println!();
            // }

            // Print consecutive miss statistics
            if !obj.consecutive_miss_counts.is_empty() {
                print!("  Consecutive misses: ");
                let mut miss_pairs: Vec<_> = obj.consecutive_miss_counts.iter().collect();
                miss_pairs.sort_by_key(|(length, _)| *length);
                for (i, (length, count)) in miss_pairs.iter().enumerate() {
                    if i > 0 {
                        print!(", ");
                    }
                    print!("{}×{}", length, count);
                }
                println!();
            }

            println!();
        }

        println!("=== Amplitude Analysis ===");
        println!("True Positive Amplitudes:");
        let tp_stats = &amplitude_analysis.true_positive_amplitude_stats;
        println!(
            "  Count: {}, Mean: {:.1}, Median: {:.1}, Std Dev: {:.1}",
            tp_stats.count, tp_stats.mean, tp_stats.median, tp_stats.std_dev
        );
        println!(
            "  Range: {:.1} - {:.1}, IQR: {:.1} - {:.1}",
            tp_stats.min, tp_stats.max, tp_stats.percentile_25, tp_stats.percentile_75
        );

        println!("False Positive Amplitudes:");
        let fp_stats = &amplitude_analysis.false_positive_amplitude_stats;
        println!(
            "  Count: {}, Mean: {:.1}, Median: {:.1}, Std Dev: {:.1}",
            fp_stats.count, fp_stats.mean, fp_stats.median, fp_stats.std_dev
        );
        println!(
            "  Range: {:.1} - {:.1}, IQR: {:.1} - {:.1}",
            fp_stats.min, fp_stats.max, fp_stats.percentile_25, fp_stats.percentile_75
        );

        if !amplitude_analysis
            .amplitude_threshold_suggestions
            .is_empty()
        {
            println!();
            println!("=== Amplitude Threshold Suggestions ===");
            for suggestion in &amplitude_analysis.amplitude_threshold_suggestions
                [..3.min(amplitude_analysis.amplitude_threshold_suggestions.len())]
            {
                println!(
                    "Threshold {:.1}: Precision {:.3}, Recall {:.3}, F1 {:.3}",
                    suggestion.threshold,
                    suggestion.expected_precision,
                    suggestion.expected_recall,
                    suggestion.expected_f1
                );
            }
        }

        println!("\nResults saved to: {}", self.config.output_path.display());
    }

    pub fn analyze_kalman_q_matrix(&self, min_detection_rate: f32) -> KalmanQAnalysis {
        let mut position_deltas_x = Vec::new();
        let mut position_deltas_y = Vec::new();
        let mut velocity_deltas_x = Vec::new();
        let mut velocity_deltas_y = Vec::new();

        println!("\n=== FILTERING STARS FOR Q MATRIX ANALYSIS ===");

        // Only analyze high-reliability stars
        for object in &self.annotations.objects {
            if let Some(tracker) = self.object_trackers.get(&object.id) {
                let detection_rate = tracker.detection_rate();

                if detection_rate >= min_detection_rate {
                    println!(
                        "Including Object {}: {:.1}% detection rate",
                        object.id,
                        detection_rate * 100.0
                    );

                    let motion_stats = self.analyze_object_motion_noise(object);
                    position_deltas_x.extend(motion_stats.position_deltas_x);
                    position_deltas_y.extend(motion_stats.position_deltas_y);
                    velocity_deltas_x.extend(motion_stats.velocity_deltas_x);
                    velocity_deltas_y.extend(motion_stats.velocity_deltas_y);
                } else {
                    println!(
                        "Excluding Object {}: {:.1}% detection rate (< {:.1}%)",
                        object.id,
                        detection_rate * 100.0,
                        min_detection_rate * 100.0
                    );
                }
            }
        }

        println!(
            "Using {} position samples and {} velocity samples for Q matrix estimation",
            position_deltas_x.len(),
            velocity_deltas_x.len()
        );

        KalmanQAnalysis {
            position_variance_x: calculate_variance(&position_deltas_x),
            position_variance_y: calculate_variance(&position_deltas_y),
            velocity_variance_x: calculate_variance(&velocity_deltas_x),
            velocity_variance_y: calculate_variance(&velocity_deltas_y),
            position_velocity_covariance: calculate_covariance(
                &position_deltas_x,
                &velocity_deltas_x,
            ),
            samples_used: position_deltas_x.len(),
            stars_included: self.count_high_detection_rate_stars(min_detection_rate),
        }
    }

    fn count_high_detection_rate_stars(&self, min_detection_rate: f32) -> usize {
        self.object_trackers
            .values()
            .filter(|tracker| tracker.detection_rate() >= min_detection_rate)
            .count()
    }

    fn analyze_object_motion_noise(&self, object: &TrackedObject) -> MotionNoiseStats {
        let mut position_deltas_x = Vec::new();
        let mut position_deltas_y = Vec::new();
        let mut velocity_deltas_x = Vec::new();
        let mut velocity_deltas_y = Vec::new();

        // Need at least 2 keyframes for linear interpolation
        if object.keyframes.len() < 2 {
            return MotionNoiseStats {
                position_deltas_x,
                position_deltas_y,
                velocity_deltas_x,
                velocity_deltas_y,
            };
        }

        // Get frame range for this object
        let (start_frame, end_frame) = match object.frame_range() {
            Some(range) => range,
            None => {
                return MotionNoiseStats {
                    position_deltas_x,
                    position_deltas_y,
                    velocity_deltas_x,
                    velocity_deltas_y,
                }
            }
        };

        println!(
            "  Analyzing Object {} from frame {} to {}",
            object.id, start_frame, end_frame
        );

        // Track the star's actual detected positions vs expected positions
        let mut previous_detection: Option<(u32, f32, f32)> = None;
        let mut sample_count = 0;

        for (frame_number, detections) in &self.frame_detections {
            // Skip frames outside this object's range
            if *frame_number < start_frame || *frame_number > end_frame {
                continue;
            }

            // Get expected position from linear interpolation
            if let Some(expected_pos) = object.position_at_frame(*frame_number) {
                // Find the closest detection to expected position
                let mut best_match: Option<&Detection> = None;
                let mut best_distance = 5.0; // Maximum matching distance

                for detection in detections {
                    let distance = ((detection.position.x - expected_pos.0).powi(2)
                        + (detection.position.y - expected_pos.1).powi(2))
                    .sqrt();

                    if distance < best_distance {
                        best_distance = distance;
                        best_match = Some(detection);
                    }
                }

                if let Some(detection) = best_match {
                    // Calculate position error (actual - expected)
                    let pos_error_x = detection.position.x - expected_pos.0;
                    let pos_error_y = detection.position.y - expected_pos.1;

                    position_deltas_x.push(pos_error_x);
                    position_deltas_y.push(pos_error_y);

                    // Calculate velocity error if we have a previous detection
                    if let Some((prev_frame, prev_x, prev_y)) = previous_detection {
                        let frame_gap = *frame_number - prev_frame;

                        if frame_gap > 0 && frame_gap <= 10 {
                            // Reasonable frame gap
                            // Actual velocity
                            let actual_vel_x = (detection.position.x - prev_x) / frame_gap as f32;
                            let actual_vel_y = (detection.position.y - prev_y) / frame_gap as f32;

                            // Expected velocity from linear interpolation
                            if let Some(prev_expected) = object.position_at_frame(prev_frame) {
                                let expected_vel_x =
                                    (expected_pos.0 - prev_expected.0) / frame_gap as f32;
                                let expected_vel_y =
                                    (expected_pos.1 - prev_expected.1) / frame_gap as f32;

                                // Velocity error
                                let vel_error_x = actual_vel_x - expected_vel_x;
                                let vel_error_y = actual_vel_y - expected_vel_y;

                                velocity_deltas_x.push(vel_error_x);
                                velocity_deltas_y.push(vel_error_y);
                            }
                        }
                    }

                    // Update previous detection
                    previous_detection =
                        Some((*frame_number, detection.position.x, detection.position.y));
                    sample_count += 1;
                }
            }
        }

        println!(
            "  Object {}: {} position samples, {} velocity samples",
            object.id,
            position_deltas_x.len(),
            velocity_deltas_x.len()
        );

        MotionNoiseStats {
            position_deltas_x,
            position_deltas_y,
            velocity_deltas_x,
            velocity_deltas_y,
        }
    }

    fn print_q_matrix_comparison(&self, analyses: &[(&str, KalmanQAnalysis)]) {
        for (label, analysis) in analyses {
            println!(
                "\n=== {} ({} stars, {} samples) ===",
                label, analysis.stars_included, analysis.samples_used
            );
            println!(
                "Position noise: x={:.4}, y={:.4}",
                analysis.position_variance_x, analysis.position_variance_y
            );
            println!(
                "Velocity noise: x={:.4}, y={:.4}",
                analysis.velocity_variance_x, analysis.velocity_variance_y
            );
        }
    }
}

impl FrameSink for PointDetectorEvaluatorSink {
    fn consume(&mut self, frame_ctx: &mut FrameContext, _ctx: &mut PipelineContext) -> () {
        let frame_number = frame_ctx.frame_index as u32;

        // Get ground truth for this frame
        let ground_truth = self.annotations.objects_at_frame(frame_number);

        let Ok(detections) = frame_ctx.try_get_as::<Vec<Detection>>("detected_points") else {
            warn!(
                "No detected points metadata found for frame {}",
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
            self.evaluate_frame(frame_number, &ground_truth, &detections, processing_time_us);

        // Log progress periodically
        if frame_number % 100 == 0
            || frame_result.ground_truth_count > 0
            || frame_result.detections_count > 0
        {
            info!(
                "Frame {}: GT={}, Det={}, TP={}, FP={}, FN={}",
                frame_number,
                frame_result.ground_truth_count,
                frame_result.detections_count,
                frame_result.true_positives,
                frame_result.false_positives,
                frame_result.false_negatives
            );
        }

        self.frame_results.push(frame_result);
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        info!("Finalizing point detector evaluation...");
        self.save_results();

        // Analyze Q matrix with different thresholds
        println!("\n=== KALMAN Q MATRIX ANALYSIS ===");

        let q_analysis_90 = self.analyze_kalman_q_matrix(0.90);
        let q_analysis_80 = self.analyze_kalman_q_matrix(0.80);
        let q_analysis_all = self.analyze_kalman_q_matrix(0.0);

        self.print_q_matrix_comparison(&[
            ("90%+ detection rate", q_analysis_90),
            ("80%+ detection rate", q_analysis_80),
            ("All stars", q_analysis_all),
        ]);

        Ok(())
    }

    fn name(&self) -> &str {
        "point_detector_evaluator"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
