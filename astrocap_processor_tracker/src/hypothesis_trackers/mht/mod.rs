pub mod config;
pub mod kalman_filter;

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

use kiddo::{immutable::float::kdtree::ImmutableKdTree, SquaredEuclidean};
use ordered_float::OrderedFloat;
use rerun::RecordingStream;
use toml::Value;

use astrocap_core::AstrocapError;

use crate::model::Detection;
use crate::traits::HypothesisTracker;

use config::MhtConfig;
use kalman_filter::KalmanFilter;

// TODO: include amplitude as a dimension in the trees
type DetectionTree = ImmutableKdTree<f32, u32, 2, 32>;
// type TrackTree = ImmutableKdTree<f32, u32, 2, 32>;

static NEXT_TRACK_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Default)]
pub struct TrackNodeStore {
    pub track_nodes: HashMap<u64, MhtTrackNode>,
}

#[derive(Debug)]
pub struct MhtTrackNode {
    pub id: u64,
    pub age: usize,
    pub parent_id: Option<u64>,
    pub detection_id: Option<u64>,
    pub state: KalmanFilter,
    pub log_likelihood: f32,
    pub cum_log_likelihood: f32,

    // TODO: this may vary, but we just take the original detection amplitude for now
    pub original_amplitude: f32,
}

impl MhtTrackNode {
    pub fn spawn_association(&self, detection: &Detection, log_likelihood: f32) -> Self {
        let mut state = self.state.predict();
        state.update(detection);

        Self {
            id: NEXT_TRACK_ID.fetch_add(1, Ordering::Relaxed),
            age: self.age + 1,
            parent_id: Some(self.id),
            detection_id: Some(detection.id),
            state,
            log_likelihood,
            cum_log_likelihood: self.cum_log_likelihood + log_likelihood,
            original_amplitude: detection.amplitude,
        }
    }

    pub fn spawn_predicted(&self, log_likelihood: f32) -> Self {
        Self {
            id: NEXT_TRACK_ID.fetch_add(1, Ordering::Relaxed),
            age: self.age + 1,
            parent_id: Some(self.id),
            detection_id: None,
            state: self.state.predict(),
            log_likelihood,
            cum_log_likelihood: self.cum_log_likelihood + log_likelihood,
            original_amplitude: self.original_amplitude,
        }
    }
}

pub struct MultiHypothesisTracker {
    config: MhtConfig,
    leaf_ids: HashSet<u64>,
    track_node_store: TrackNodeStore,
}

impl MultiHypothesisTracker {
    pub fn new(raw_config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(raw_config) = raw_config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: MhtConfig = raw_config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!(
                "MhtTracker config error: {}",
                e.to_string()
            ))
        })?;

        let leaf_ids = HashSet::default();
        let track_node_store = TrackNodeStore::default();

        Ok(Self {
            config,
            leaf_ids,
            track_node_store,
        })
    }
}

impl HypothesisTracker for MultiHypothesisTracker {
    fn process_frame(&mut self, detections: &[Detection]) {
        // build detection k-d tree
        let detection_coords: Vec<[f32; 2]> = detections
            .iter()
            .map(|d| [d.position.x, d.position.y])
            .collect();
        let detections_tree: DetectionTree =
            DetectionTree::new_from_slice(detection_coords.as_slice());

        // let track_coords: Vec<[f32; 2]> = self
        //     .tracks
        //     .iter()
        //     .map(|t| [t.state.state.x, t.state.state.y])
        //     .collect();
        // let tracks_tree: TrackTree = TrackTree::new_from_slice(track_coords.as_slice());

        let mut new_leaves: HashSet<u64> = HashSet::default();
        let mut new_tracks: Vec<MhtTrackNode> = Vec::default();

        // for each existing leaf:
        for leaf_id in self.leaf_ids.iter() {
            let track = &self.track_node_store.track_nodes[leaf_id];

            //  * find all detections within distance gate
            let gated_detections = detections_tree.within_unsorted::<SquaredEuclidean>(
                &[track.state.x(), track.state.y()],
                self.config.association_gating_distance,
            );

            //  * branch a child track for each detection
            for nn in gated_detections {
                let detection = &detections[nn.item as usize];

                let pd = Self::pd_from_amplitude(
                    detection.amplitude,
                    self.config.threshold,
                    self.config.sigma,
                );

                let child_track_node = track
                    .spawn_association(detection, track.state.assoc_log_likelihood(detection, pd));

                new_leaves.insert(child_track_node.id);
                new_tracks.push(child_track_node);
            }

            //  * spawn a miss branch
            let prediction_track_node =
                track.spawn_predicted(self.calc_miss_log_likelihood(Self::pd_from_amplitude(
                    track.original_amplitude,
                    self.config.threshold,
                    self.config.sigma,
                )));

            new_leaves.insert(prediction_track_node.id);
            self.track_node_store
                .track_nodes
                .insert(prediction_track_node.id, prediction_track_node);
        }

        new_tracks.into_iter().for_each(|t| {
            self.track_node_store.track_nodes.insert(t.id, t);
        });

        // birth new tracks
        // TODO:
        //  * Cap births per frame (e.g. max 10).
        //  * Per-tile cap (e.g. max 1–2 births per 64×64 tile).
        //  * Promotion rule: new track stays “tentative” until it gets M-of-N associations
        for (idx, detection) in detections.iter().enumerate() {
            if !self.should_spawn_birth(detection) {
                continue;
            }

            let new_track = self.spawn_track_from_detection(detection, idx as u64);
            new_leaves.insert(new_track.id);
            self.track_node_store
                .track_nodes
                .insert(new_track.id, new_track);
        }

        self.leaf_ids = new_leaves;

        self.prune_hypotheses();
    }

    fn log_to_rerun(&self, rec: &RecordingStream) {
        if self.leaf_ids.is_empty() {
            return;
        }

        let tracks: Vec<_> = self
            .leaf_ids
            .iter()
            .map(|id| self.track_node_store.track_nodes.get(id).unwrap())
            .collect();

        rec.log(
            "model/tracks".to_string(),
            &rerun::Points2D::new(
                tracks
                    .iter()
                    .map(|track| (track.state.x(), track.state.y())),
            )
            .with_colors(tracks.iter().map(|track| {
                // Color tracks based on their confidence
                let normalized_confidence = track.cum_log_likelihood.clamp(-40.0, 40.0);
                let hue = normalized_confidence * 1.5; // 0° = red (low confidence), 120° = green (high confidence)
                let (r, g, b) = crate::processor::hsv_to_rgb(hue, 1.0, 1.0);
                rerun::Color::from_rgb(r, g, b)
            }))
            .with_labels(tracks.iter().map(|track| {
                format!(
                    "ID: {}, Age: {}, CLL: {:.2}",
                    track.id, track.age, track.cum_log_likelihood
                )
            })),
        )
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to log tracks to rerun: {}", e);
        });
    }
}

impl MultiHypothesisTracker {
    fn prune_hypotheses(&mut self) {
        // Gather (id, score) for all leaves above the threshold
        let mut candidates: Vec<(u64, f32)> = self
            .leaf_ids
            .iter()
            .filter_map(|id| {
                self.track_node_store.track_nodes.get(id).and_then(|n| {
                    let score = n.cum_log_likelihood; // <-- use cumulative
                    if score > self.config.min_leaf_log_likelihood {
                        Some((*id, score))
                    } else {
                        None
                    }
                })
            })
            .collect();

        // If more than max_leaf_count, select the N best
        if candidates.len() > self.config.max_leaf_count {
            let n = self.config.max_leaf_count;
            candidates.select_nth_unstable_by_key(n, |&(_, score)| OrderedFloat(-score));
            candidates.truncate(n);
        }

        self.leaf_ids = candidates.into_iter().map(|(id, _)| id).collect();
    }

    fn spawn_track_from_detection(&self, detection: &Detection, idx: u64) -> MhtTrackNode {
        let log_likelihood = self.calc_birth_log_likelihood(detection);
        MhtTrackNode {
            id: NEXT_TRACK_ID.fetch_add(1, Ordering::AcqRel),
            age: 0,
            parent_id: None,
            detection_id: Some(idx),
            state: KalmanFilter::new_from_detection(&detection),
            log_likelihood,
            cum_log_likelihood: log_likelihood,
            original_amplitude: detection.amplitude,
        }
    }

    fn should_spawn_birth(&self, detection: &Detection) -> bool {
        let birth_ll = self.calc_birth_log_likelihood(detection);
        let clutter_ll = self.calc_clutter_log_likelihood(detection);

        (birth_ll - clutter_ll) > self.config.birth_tau
    }

    /// PD is the probability that a track will be detected in any given frame.
    /// We derive it from the amplitude of the track, the detection threshold
    /// of the detector, and the noise scale (standard deviation) of the background.
    /// This mapping function uses a logistic-like curve centered at the threshold.
    pub fn pd_from_amplitude(amplitude: f32, threshold: f32, sigma: f32) -> f32 {
        // Gaussian CDF approximation
        let z = (amplitude - threshold) / sigma;
        0.5 * (1.0 + (z / (1.0 + 0.2316419 * z.abs())).tanh()) // crude fast erf approx
    }

    fn calc_birth_log_likelihood(&self, detection: &Detection) -> f32 {
        let pd = Self::pd_from_amplitude(
            detection.amplitude,
            self.config.threshold,
            self.config.sigma,
        )
        .clamp(1e-6, 1.0 - 1e-6);

        self.config.track_birth_rate.ln() + pd.ln()
    }

    fn calc_clutter_log_likelihood(&self, _detection: &Detection) -> f32 {
        self.config.clutter_rate.ln()
    }

    fn calc_miss_log_likelihood(&self, pd: f32) -> f32 {
        (1.0 - pd).ln()
    }

    pub fn print_leaf_chains(&self) {
        for leaf_id in &self.leaf_ids {
            let mut chain = Vec::new();
            let mut current_id = Some(*leaf_id);

            while let Some(id) = current_id {
                if let Some(node) = self.track_node_store.track_nodes.get(&id) {
                    chain.push(format!(
                        "({id}: pos=({}, {}), ll={:.2}, cum={:.2})",
                        node.state.x(),
                        node.state.y(),
                        node.log_likelihood,
                        node.cum_log_likelihood
                    ));
                    current_id = node.parent_id;
                } else {
                    break;
                }
            }

            // chain is leaf → root, which is what we want now
            println!("Leaf chain: {}", chain.join(" <- "));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn toy_two_frames_with_branching() {
        use std::collections::HashSet;

        // Simple config: low threshold, generous pruning
        let config = MhtConfig {
            association_gating_distance: 5.0,
            track_birth_rate: 2e-8,
            clutter_rate: 3.69e-5,
            min_leaf_log_likelihood: -100.0,
            max_leaf_count: 100,
            threshold: 50.0,
            sigma: 5.0,
            birth_tau: -10.0,
            r_x: 1.0,
            r_y: 1.0,
        };

        let mut tracker = MultiHypothesisTracker {
            config,
            leaf_ids: HashSet::default(),
            track_node_store: TrackNodeStore::default(),
        };

        // --- Frame 1: two detections ---
        let frame1 = vec![
            Detection {
                id: 0,
                position: Vector2::new(10.0, 10.0),
                amplitude: 80.0,
            },
            Detection {
                id: 1,
                position: Vector2::new(30.0, 30.0),
                amplitude: 85.0,
            },
        ];

        tracker.process_frame(&frame1);

        println!("After frame 1:");
        tracker.print_leaf_chains();

        // Expect: 2 leaf tracks (one per detection)
        assert_eq!(tracker.leaf_ids.len(), 2);

        // --- Frame 2: one continuation near (10,10), plus a new detection at (50,50) ---
        let frame2 = vec![
            Detection {
                id: 2,
                position: Vector2::new(11.0, 11.0),
                amplitude: 82.0,
            },
            Detection {
                id: 3,
                position: Vector2::new(50.0, 50.0),
                amplitude: 90.0,
            },
        ];

        tracker.process_frame(&frame2);

        println!("After frame 2:");
        tracker.print_leaf_chains();

        // After frame 2, we expect:
        // - 1 association branch continuing the (10,10) track near (11,11)
        // - 1 miss branch for the (30,30) track (no detection nearby)
        // - 1 new birth for the new (50,50) detection
        // So roughly 3 leaves.
        assert!(tracker.leaf_ids.len() >= 3);

        // Also sanity check that one of the leaves is close to (11,11)
        let mut found_assoc = false;
        for id in &tracker.leaf_ids {
            if let Some(node) = tracker.track_node_store.track_nodes.get(id) {
                let dx = node.state.x() - 11.0;
                let dy = node.state.y() - 11.0;
                if dx.abs() < 2.0 && dy.abs() < 2.0 {
                    found_assoc = true;
                }
            }
        }
        assert!(found_assoc, "Expected to find an association near (11,11)");
    }

    #[test]
    fn toy_miss_only_case() {
        use std::collections::HashSet;

        let config = MhtConfig {
            association_gating_distance: 5.0,
            track_birth_rate: 2e-8,
            clutter_rate: 3.69e-5,
            min_leaf_log_likelihood: -100.0,
            max_leaf_count: 100,
            threshold: 50.0,
            sigma: 5.0,
            birth_tau: -10.0,
            r_x: 1.0,
            r_y: 1.0,
        };

        let mut tracker = MultiHypothesisTracker {
            config,
            leaf_ids: HashSet::default(),
            track_node_store: TrackNodeStore::default(),
        };

        // --- Frame 1: one detection at (10,10) ---
        let frame1 = vec![Detection {
            id: 0,
            position: Vector2::new(10.0, 10.0),
            amplitude: 80.0,
        }];

        tracker.process_frame(&frame1);

        println!("After frame 1:");
        tracker.print_leaf_chains();
        assert_eq!(tracker.leaf_ids.len(), 1);

        // Grab the track ID from frame 1
        let first_leaf_id = *tracker.leaf_ids.iter().next().unwrap();
        let first_node = tracker
            .track_node_store
            .track_nodes
            .get(&first_leaf_id)
            .unwrap();
        let x1 = first_node.state.x();
        let y1 = first_node.state.y();

        // --- Frame 2: no detections ---
        let frame2: Vec<Detection> = vec![];
        tracker.process_frame(&frame2);

        println!("After frame 2 (miss only):");
        tracker.print_leaf_chains();
        assert_eq!(
            tracker.leaf_ids.len(),
            1,
            "Should still have 1 leaf after a miss"
        );

        // The new leaf should be a child of the old one
        let second_leaf_id = *tracker.leaf_ids.iter().next().unwrap();
        let second_node = tracker
            .track_node_store
            .track_nodes
            .get(&second_leaf_id)
            .unwrap();
        assert_eq!(second_node.parent_id, Some(first_leaf_id));

        // State should have propagated forward (predict-only step)
        // For now vx,vy=0, so (x,y) should stay the same.
        assert!((second_node.state.x() - x1).abs() < 1e-5);
        assert!((second_node.state.y() - y1).abs() < 1e-5);
    }

    #[test]
    fn toy_miss_with_velocity() {
        use nalgebra::{Matrix4, Vector4};
        use std::collections::HashSet;

        let config = MhtConfig {
            association_gating_distance: 5.0,
            track_birth_rate: 2e-8,
            clutter_rate: 3.69e-5,
            min_leaf_log_likelihood: -100.0,
            max_leaf_count: 100,
            threshold: 50.0,
            sigma: 5.0,
            birth_tau: -10.0,
            r_x: 1.0,
            r_y: 1.0,
        };

        let mut tracker = MultiHypothesisTracker {
            config,
            leaf_ids: HashSet::default(),
            track_node_store: TrackNodeStore::default(),
        };

        // Manually create a node with velocity (2,3)
        let init_state = Vector4::new(0.0, 0.0, 2.0, 3.0);
        let init_kf = KalmanFilter::new_with_covariance(init_state, Matrix4::identity());

        let node = MhtTrackNode {
            id: NEXT_TRACK_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            age: 0,
            parent_id: None,
            detection_id: None,
            state: init_kf,
            log_likelihood: 0.0,
            cum_log_likelihood: 0.0,
            original_amplitude: 100.0,
        };

        let id = node.id;
        tracker.track_node_store.track_nodes.insert(id, node);
        tracker.leaf_ids.insert(id);

        // --- Frame with no detections ---
        let frame: Vec<Detection> = vec![];
        tracker.process_frame(&frame);

        println!("After miss with velocity:");
        tracker.print_leaf_chains();

        // We should still have one leaf
        assert_eq!(tracker.leaf_ids.len(), 1);

        // The new leaf should be a child of the old one
        let new_leaf_id = *tracker.leaf_ids.iter().next().unwrap();
        let new_node = tracker
            .track_node_store
            .track_nodes
            .get(&new_leaf_id)
            .unwrap();
        assert_eq!(new_node.parent_id, Some(id));

        // Position should have advanced by velocity: (0,0) -> (2,3)
        assert!((new_node.state.x() - 2.0).abs() < 1e-5);
        assert!((new_node.state.y() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn toy_long_gap_then_reassoc() {
        use nalgebra::{Matrix4, Vector4};
        use std::collections::HashSet;

        let config = MhtConfig {
            association_gating_distance: 5.0,
            track_birth_rate: 2e-8,
            clutter_rate: 3.69e-5,
            min_leaf_log_likelihood: -100.0,
            max_leaf_count: 100,
            threshold: 50.0,
            sigma: 5.0,
            birth_tau: -10.0,
            r_x: 1.0,
            r_y: 1.0,
        };

        let mut tracker = MultiHypothesisTracker {
            config,
            leaf_ids: HashSet::default(),
            track_node_store: TrackNodeStore::default(),
        };

        // Seed a node at (0,0) with velocity (1,0)
        let init_state = Vector4::new(0.0, 0.0, 1.0, 0.0);
        let init_kf = KalmanFilter::new_with_covariance(init_state, Matrix4::identity());

        let node = MhtTrackNode {
            id: NEXT_TRACK_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            age: 0,
            parent_id: None,
            detection_id: None,
            state: init_kf,
            log_likelihood: 0.0,
            cum_log_likelihood: 0.0,
            original_amplitude: 100.0,
        };

        let mut id = node.id;
        tracker.track_node_store.track_nodes.insert(id, node);
        tracker.leaf_ids.insert(id);

        // --- 5 frames with no detections ---
        for _ in 0..5 {
            let frame: Vec<Detection> = vec![];
            tracker.process_frame(&frame);

            // update id to the new leaf each time
            id = *tracker.leaf_ids.iter().next().unwrap();
        }

        // The track should have propagated to ~x=5, y=0
        let node = tracker.track_node_store.track_nodes.get(&id).unwrap();
        println!("After 5 misses: {:?}", node.state);
        assert!((node.state.x() - 5.0).abs() < 1.0);
        assert!((node.state.y() - 0.0).abs() < 1.0);

        // --- Frame with a reappearing detection near (5.2,0.1) ---
        let frame = vec![Detection {
            id: 999,
            position: Vector2::new(5.2, 0.1),
            amplitude: 90.0,
        }];
        tracker.process_frame(&frame);

        println!("After reassociation:");
        tracker.print_leaf_chains();

        // New leaf should exist, near (5.2,0.1)
        let reassoc_id = *tracker.leaf_ids.iter().next().unwrap();
        let reassoc_node = tracker
            .track_node_store
            .track_nodes
            .get(&reassoc_id)
            .unwrap();

        assert!((reassoc_node.state.x() - 5.2).abs() < 1.0);
        assert!((reassoc_node.state.y() - 0.1).abs() < 1.0);
    }
}
