use crate::model::Track;
use crate::traits::{Associations, Associator, Configurable, ConfigurableConfig};
use astrocap_core::{structs::Detection, AstrocapError};
use kiddo::{MutableKdTree, SquaredEuclidean};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct Config {
    max_distance: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_distance: 3.0, // pixels
        }
    }
}

impl ConfigurableConfig for Config {}

pub struct NearestAssociator {
    config: Config,
}

impl Configurable for NearestAssociator {
    type Config = Config;

    fn from_config(config: Self::Config) -> Result<Box<Self>, AstrocapError> {
        Ok(Box::new(Self { config }))
    }
}

impl Associator for NearestAssociator {
    fn associate(
        &self,
        detections: &[Detection],
        tracks: &[Track],
    ) -> (Associations, HashSet<usize>) {
        let mut associations = HashMap::new();
        let mut used_detections = HashSet::new();

        // Build KD-tree of detections
        let mut detection_tree: MutableKdTree<f32, 2> = MutableKdTree::builder()
            .build_from_entries(&[])
            .expect("empty detection tree construction failed");
        for (idx, detection) in detections.iter().enumerate() {
            detection_tree
                .add(&[detection.position.x, detection.position.y], idx as u32)
                .expect("detection tree insertion failed");
        }

        // For each track, find nearest detection within gate using KD-tree
        for (track_idx, track) in tracks.iter().enumerate() {
            let track_pos = [track.state.state[0], track.state.state[1]];

            // Find all detections within the gate radius
            let candidates = detection_tree
                .query(&track_pos)
                .within::<SquaredEuclidean<f32>>(
                    self.config.max_distance * self.config.max_distance,
                )
                .execute();

            // Find the closest unused detection
            let mut best_detection_idx = None;
            let mut best_distance_sq = f32::INFINITY;

            for candidate in candidates {
                let detection_idx = candidate.item as usize;

                // Skip already used detections
                if used_detections.contains(&detection_idx) {
                    continue;
                }

                if candidate.distance < best_distance_sq {
                    best_distance_sq = candidate.distance;
                    best_detection_idx = Some(detection_idx);
                }
            }

            // If we found a valid association, record it
            if let Some(detection_idx) = best_detection_idx {
                associations.insert(track_idx, detection_idx);
                used_detections.insert(detection_idx);
            }
        }

        // Find unassociated detections
        let unassociated_detections: HashSet<usize> = (0..detections.len())
            .filter(|idx| !used_detections.contains(idx))
            .collect();

        (associations, unassociated_detections)
    }
}
