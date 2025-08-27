use std::collections::{HashMap, HashSet};

use astrocap_core::structs::DetectedPoint;

use crate::model::Track;

/// maps track indices to associated detection index
pub type Associations = HashMap<usize, usize>;

/// Initializes new tracks from unassociated detections
pub trait Initiator: Send + Sync + 'static {
    fn initiate(&self, detections: &[&DetectedPoint]) -> Vec<Track>;
}

/// Terminates tracks based on various criteria
pub trait Terminator: Send + Sync + 'static {
    fn should_terminate(&self, track: &Track) -> bool;
}

/// Handles data association between tracks and detections
pub trait Associator: Send + Sync + 'static {
    fn associate(
        &self,
        detections: &[DetectedPoint],
        tracks: &[Track],
    ) -> (Associations, HashSet<usize>);
}

/// Updates track state with new measurements
pub trait Updater: Send + Sync + 'static {
    fn update(&self, track: &mut Track, detection: &DetectedPoint);
}

/// Predicts track state forward in time
pub trait Predictor: Send + Sync + 'static {
    fn predict(&self, track: &mut Track);
}
