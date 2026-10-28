use super::factory::SingleHypothesisTrackerElementFactory;
use crate::model::Track;
use crate::processor::hsv_to_rgb;
use crate::traits::{Associator, HypothesisTracker, Initiator, Predictor, Terminator, Updater};
use astrocap_core::structs::Detection;
use astrocap_core::traits::{TrackSummarize, TrackSummary};
use astrocap_core::{AstrocapError, FrameContext};
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

        let factory = SingleHypothesisTrackerElementFactory::new(raw_config)?;
        let initiator = factory.create_initiator()?;
        let terminator = factory.create_terminator()?;
        let associator = factory.create_associator()?;
        let updater = factory.create_updater()?;
        let predictor = factory.create_predictor()?;

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
    fn process_frame(&mut self, detections: &[Detection], frame_index: u64) {
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

        let mut new_tracks = self.initiator.initiate(unassociated_detections.as_slice());
        for track in &mut new_tracks {
            track.first_seen = frame_index;
        }

        self.tracks.extend(new_tracks);

        // run all tracks through the reaper
        self.tracks
            .retain(|track| !self.terminator.should_terminate(track));
    }

    fn track_summaries(&self) -> Vec<TrackSummary> {
        self.tracks.iter().map(TrackSummarize::summarize).collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn preserves_large_frame_indices_in_track_summaries() {
        let config = Value::Table(Default::default());
        let mut tracker = SingleHypothesisTracker::new(Some(&config)).unwrap();
        let frame_index = u64::from(u32::MAX) + 1;
        let detection = Detection {
            id: 1,
            position: Vector2::new(10.0, 20.0),
            amplitude: 80.0,
        };
        tracker.process_frame(&[detection], frame_index);
        let tracks = tracker.track_summaries();
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].first_seen, frame_index);
    }
}
