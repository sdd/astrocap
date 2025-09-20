use super::factory::SingleHypothesisTrackerElementFactory;
use crate::model::Track;
use crate::processor::hsv_to_rgb;
use crate::traits::{Associator, HypothesisTracker, Initiator, Predictor, Terminator, Updater};
use astrocap_core::{AstrocapError, FrameContext};
use astrocap_core::structs::Detection;
use rerun::RecordingStream;
use toml::Value;

pub struct SingleHypothesisTracker {
    initiator: Box<dyn Initiator>,
    terminator: Box<dyn Terminator>,
    associator: Box<dyn Associator>,
    updater: Box<dyn Updater>,
    predictor: Box<dyn Predictor>,

    tracks: Vec<Track>,
}

impl SingleHypothesisTracker {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(raw_config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        // Create components using the raw config Value
        let initiator = SingleHypothesisTrackerElementFactory::create_initiator(raw_config)?;
        let terminator = SingleHypothesisTrackerElementFactory::create_terminator(raw_config)?;
        let associator = SingleHypothesisTrackerElementFactory::create_associator(raw_config)?;
        let updater = SingleHypothesisTrackerElementFactory::create_updater(raw_config)?;
        let predictor = SingleHypothesisTrackerElementFactory::create_predictor(raw_config)?;

        Ok(Self {
            initiator,
            terminator,
            associator,
            updater,
            predictor,
            tracks: Vec::new(),
        })
    }
}

impl HypothesisTracker for SingleHypothesisTracker {
    fn process_frame(&mut self, detections: &[Detection], _frame_index: usize) {
        // associate detections with existing tracks
        let (associations, unassociated_detections) =
            self.associator.associate(&detections, &self.tracks);

        for (track_idx, mut track) in self.tracks.iter_mut().enumerate() {
            track.age += 1;

            if let Some(detection_idx) = associations.get(&track_idx) {
                // for associated tracks: update the track with the detection via the updater
                self.updater.update(&mut track, &detections[*detection_idx])
            } else {
                // for unassociated tracks: update the track with the predictor
                track.confidence -= 0.01;
                self.predictor.predict(&mut track);
            }
        }

        // pass unassociated detections to the initiator
        let unassociated_detections = unassociated_detections
            .iter()
            .map(|idx| &detections[*idx])
            .collect::<Vec<_>>();

        let new_tracks = self.initiator.initiate(unassociated_detections.as_slice());

        self.tracks.extend(new_tracks);

        // run all tracks through the reaper
        self.tracks
            .retain(|track| !self.terminator.should_terminate(track));
    }

    fn log_to_rerun(&self, rec: &RecordingStream, _frame_ctx: &mut FrameContext) {
        if self.tracks.is_empty() {
            return;
        }

        rec.log(
            "tracker/tracks".to_string(),
            &rerun::Points2D::new(
                self.tracks
                    .iter()
                    .map(|track| (track.state.state.x, track.state.state.y)),
            )
            .with_colors(self.tracks.iter().map(|track| {
                // Color tracks based on their confidence
                let normalized_confidence = track.confidence.min(1.0).max(0.0);
                let hue = normalized_confidence * 120.0; // 0° = red (low confidence), 120° = green (high confidence)
                let (r, g, b) = hsv_to_rgb(hue, 1.0, 1.0);
                rerun::Color::from_rgb(r, g, b)
            }))
            .with_labels(self.tracks.iter().map(|track| {
                format!(
                    "ID: {}, Age: {}, Conf: {:.2}",
                    track.id, track.age, track.confidence
                )
            })),
        )
        .unwrap_or_else(|e| {
            tracing::warn!("Failed to log tracks to rerun: {}", e);
        });
    }
}
