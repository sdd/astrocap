use std::collections::HashMap;

use crate::config::Config;
use astrocap_core::annotations::{AnnotationSession, ObjectType, TrackedObject};
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::structs::DetectedPoint;
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
#[derive(Debug)]
struct ObjectTracker {
    pub object_id: u32,
    pub frames_present: u32,
    pub detections_matched: u32,
    pub total_distance: f32,
    pub distance_count: u32,
    pub detection_amplitudes: Vec<f32>,
    pub missed_frames_with_brighter_fps: u32,
    pub brightest_nearby_fp_amplitude: Option<f32>,
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
        }
    }

    fn add_frame(&mut self, detected: bool, distance: Option<f32>, amplitude: Option<f32>) {
        self.frames_present += 1;
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

    fn add_missed_frame_info(
        &mut self,
        has_brighter_fps: bool,
        brightest_fp_amplitude: Option<f32>,
    ) {
        if has_brighter_fps {
            self.missed_frames_with_brighter_fps += 1;
        }

        if let Some(fp_amp) = brightest_fp_amplitude {
            self.brightest_nearby_fp_amplitude = Some(
                self.brightest_nearby_fp_amplitude
                    .map(|current| current.max(fp_amp))
                    .unwrap_or(fp_amp),
            );
        }
    }

    fn detection_rate(&self) -> f32 {
        if self.frames_present > 0 {
            self.detections_matched as f32 / self.frames_present as f32
        } else {
            0.0
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
        if !self.detection_amplitudes.is_empty() {
            Some(
                self.detection_amplitudes.iter().sum::<f32>()
                    / self.detection_amplitudes.len() as f32,
            )
        } else {
            None
        }
    }

    fn amplitude_range(&self) -> Option<(f32, f32)> {
        if !self.detection_amplitudes.is_empty() {
            let min = self
                .detection_amplitudes
                .iter()
                .fold(f32::INFINITY, |a, &b| a.min(b));
            let max = self
                .detection_amplitudes
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            Some((min, max))
        } else {
            None
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
    frame_detections: Vec<(u32, Vec<DetectedPoint>)>,
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
        detections: &[DetectedPoint],
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

                let distance = ((gt_x - (detected_point.x as f32)).powi(2)
                    + (gt_y - (detected_point.y as f32)).powi(2))
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
                            ((gt_x - det.x as f32).powi(2) + (gt_y - det.y as f32).powi(2)).sqrt(),
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

        for obj in &self.annotations.objects {
            let tracker = &self.object_trackers[&obj.id];
            let frame_range = obj.frame_range().unwrap_or((0, 0));

            metrics.push(ObjectMetrics {
                object_id: obj.id,
                object_name: obj.name.clone(),
                object_type: obj.object_type.clone(),
                frames_present: tracker.frames_present,
                detections_matched: tracker.detections_matched,
                detection_rate: tracker.detection_rate(),
                frame_range,
                avg_detection_distance: tracker.avg_detection_distance(),
                avg_detection_amplitude: tracker.avg_detection_amplitude(),
                amplitude_range: tracker.amplitude_range(),
            });
        }

        // Sort by detection rate (worst first for easier spotting of problems)
        metrics.sort_by(|a, b| a.detection_rate.partial_cmp(&b.detection_rate).unwrap());

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
        println!("\n=== POINT DETECTOR EVALUATION RESULTS ===");
        println!("Frames processed: {}", aggregate.frames_processed);
        println!(
            "Total ground truth points: {}",
            aggregate.total_ground_truth
        );
        println!("Total detections: {}", aggregate.total_detections);
        println!("True positives: {}", aggregate.total_true_positives);
        println!("False positives: {}", aggregate.total_false_positives);
        println!("False negatives: {}", aggregate.total_false_negatives);
        println!("Precision: {:.3}", aggregate.precision);
        println!("Recall: {:.3}", aggregate.recall);
        println!("F1 Score: {:.3}", aggregate.f1_score);
        println!(
            "Distance threshold: {} pixels",
            self.config.distance_threshold
        );

        // Amplitude statistics
        println!("\n=== AMPLITUDE ANALYSIS ===");
        let tp_stats = &amplitude_analysis.true_positive_amplitude_stats;
        let fp_stats = &amplitude_analysis.false_positive_amplitude_stats;

        println!(
            "TRUE POSITIVES: count={}, mean={:.1}, median={:.1}, range={:.1}-{:.1}",
            tp_stats.count, tp_stats.mean, tp_stats.median, tp_stats.min, tp_stats.max
        );
        println!(
            "FALSE POSITIVES: count={}, mean={:.1}, median={:.1}, range={:.1}-{:.1}",
            fp_stats.count, fp_stats.mean, fp_stats.median, fp_stats.min, fp_stats.max
        );

        // Threshold suggestions
        println!("\n=== THRESHOLD SUGGESTIONS ===");
        println!(
            "{:<10} {:<10} {:<8} {:<8} {:<8}",
            "Threshold", "F1 Score", "Precision", "Recall", "TP/FP"
        );
        println!("{}", "-".repeat(50));
        for suggestion in amplitude_analysis
            .amplitude_threshold_suggestions
            .iter()
            .take(5)
        {
            println!(
                "{:<10.1} {:<10.3} {:<8.3} {:<8.3} {:<8}",
                suggestion.threshold,
                suggestion.expected_f1,
                suggestion.expected_precision,
                suggestion.expected_recall,
                format!("{}/{}", suggestion.tp_count, suggestion.fp_count)
            );
        }

        // Per-object breakdown with amplitude info
        println!("\n=== PER-OBJECT DETECTION RATES WITH AMPLITUDE ===");
        println!(
            "{:<4} {:<12} {:<8} {:<8} {:<8} {:<12} {:<12} {:<12}",
            "ID", "Name", "Present", "Detected", "Rate", "Avg Dist", "Avg Amp", "Missed w/ FPs"
        );
        println!("{}", "-".repeat(90));

        for obj in per_object {
            let avg_dist_str = if let Some(dist) = obj.avg_detection_distance {
                format!("{:.2}", dist)
            } else {
                "N/A".to_string()
            };

            let avg_amp_str = if let Some(amp) = obj.avg_detection_amplitude {
                format!("{:.1}", amp)
            } else {
                "N/A".to_string()
            };

            let rate_str = format!("{:.1}%", obj.detection_rate * 100.0);

            // Find corresponding amplitude analysis
            let missed_with_fps = amplitude_analysis
                .per_object_amplitude_analysis
                .iter()
                .find(|a| a.object_id == obj.object_id)
                .map(|a| a.missed_frames_with_brighter_fps)
                .unwrap_or(0);

            println!(
                "{:<4} {:<12} {:<8} {:<8} {:<8} {:<12} {:<12} {:<12}",
                obj.object_id,
                obj.object_name.chars().take(12).collect::<String>(),
                obj.frames_present,
                obj.detections_matched,
                rate_str,
                avg_dist_str,
                avg_amp_str,
                missed_with_fps
            );
        }

        // Highlight objects with threshold issues
        let problematic_objects: Vec<_> = amplitude_analysis
            .per_object_amplitude_analysis
            .iter()
            .filter(|obj| obj.missed_frames_with_brighter_fps > 0)
            .collect();

        if !problematic_objects.is_empty() {
            println!("\n⚠️  OBJECTS MISSING DUE TO THRESHOLD ISSUES:");
            for obj in problematic_objects {
                let total_missed = obj.total_missed_frames;
                let missed_with_fps = obj.missed_frames_with_brighter_fps;
                let percentage = if total_missed > 0 {
                    (missed_with_fps as f32 / total_missed as f32) * 100.0
                } else {
                    0.0
                };

                println!(
                    "   {} ({}): {}/{} missed frames ({:.1}%) had brighter false positives nearby",
                    obj.object_name, obj.object_id, missed_with_fps, total_missed, percentage
                );

                if let Some(brightest_fp) = obj.brightest_nearby_fp_amplitude {
                    println!("     Brightest nearby FP amplitude: {:.1}", brightest_fp);
                }
            }
        }

        println!("{}", "=".repeat(50));
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
                let mut best_match: Option<&DetectedPoint> = None;
                let mut best_distance = 5.0; // Maximum matching distance

                for detection in detections {
                    let distance = ((detection.x - expected_pos.0).powi(2)
                        + (detection.y - expected_pos.1).powi(2))
                    .sqrt();

                    if distance < best_distance {
                        best_distance = distance;
                        best_match = Some(detection);
                    }
                }

                if let Some(detection) = best_match {
                    // Calculate position error (actual - expected)
                    let pos_error_x = detection.x - expected_pos.0;
                    let pos_error_y = detection.y - expected_pos.1;

                    position_deltas_x.push(pos_error_x);
                    position_deltas_y.push(pos_error_y);

                    // Calculate velocity error if we have a previous detection
                    if let Some((prev_frame, prev_x, prev_y)) = previous_detection {
                        let frame_gap = *frame_number - prev_frame;

                        if frame_gap > 0 && frame_gap <= 10 {
                            // Reasonable frame gap
                            // Actual velocity
                            let actual_vel_x = (detection.x - prev_x) / frame_gap as f32;
                            let actual_vel_y = (detection.y - prev_y) / frame_gap as f32;

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
                    previous_detection = Some((*frame_number, detection.x, detection.y));
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

        let Ok(detections) = frame_ctx.try_get_as::<Vec<DetectedPoint>>("detected_points") else {
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
