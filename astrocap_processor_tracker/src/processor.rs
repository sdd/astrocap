use crate::config::Config;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::structs::DetectedPoint;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use rerun::RecordingStream;
use toml::Value;

use crate::factory::Factory;
use crate::model::{Detection, Track};
use crate::traits::{Associator, Initiator, Predictor, Terminator, Updater};

pub struct TrackerProcessor {
    initiator: Box<dyn Initiator>,
    terminator: Box<dyn Terminator>,
    associator: Box<dyn Associator>,
    updater: Box<dyn Updater>,
    predictor: Box<dyn Predictor>,
    config: Config,

    tracks: Vec<Track>,
}

impl TrackerProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(raw_config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        // Parse the main config struct for any processor-level configuration
        let config: Config = raw_config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!("Tracker:: {}", e.to_string()))
        })?;

        // Create components using the raw config Value
        let initiator = Factory::create_initiator(raw_config)?;
        let terminator = Factory::create_terminator(raw_config)?;
        let associator = Factory::create_associator(raw_config)?;
        let updater = Factory::create_updater(raw_config)?;
        let predictor = Factory::create_predictor(raw_config)?;

        let tracks = Vec::new();

        Ok(Self {
            config,
            tracks,
            initiator,
            terminator,
            associator,
            updater,
            predictor,
        })
    }

    fn log_to_rerun(&self, rec: &RecordingStream) {
        if self.tracks.is_empty() {
            return;
        }

        rec.log(
            "model/tracks".to_string(),
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

impl FrameProcessor for TrackerProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let detections = frame_ctx.get_as::<Vec<DetectedPoint>>("detected_points");

        // map DetectedPoint to Detection
        let detections = detections
            .iter()
            .map(|d| Detection::new(d.x, d.y, d.amplitude))
            .collect::<Vec<_>>();

        frame_ctx.put("tracker/detections", detections.clone());

        // associate detections with existing tracks
        let (associations, unassociated_detections) =
            self.associator.associate(&detections, &self.tracks);

        for (track_idx, track) in self.tracks.iter_mut().enumerate() {
            track.age += 1;

            if let Some(detection_idx) = associations.get(&track_idx) {
                // for associated tracks: update the track with the detection via the updater
                self.updater.update(track, &detections[*detection_idx])
            } else {
                // for unassociated tracks: update the track with the predictor
                track.confidence -= 0.01;
                self.predictor.predict(track);
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

        frame_ctx.put("tracker/tracks", self.tracks.clone());

        // Log to rerun if available
        if let Ok(rec) = ctx.try_get_as::<RecordingStream>("rerun") {
            self.log_to_rerun(&rec);
        }

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "tracker"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

// Helper function for HSV to RGB conversion (same as in model processor)
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let h = h % 360.0; // Wrap hue to [0, 360)
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);

    let c = v * s; // Chroma
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r_prime, g_prime, b_prime) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        300..=359 => (c, 0.0, x),
        _ => (0.0, 0.0, 0.0), // Should never happen due to modulo
    };

    let r = ((r_prime + m) * 255.0).round() as u8;
    let g = ((g_prime + m) * 255.0).round() as u8;
    let b = ((b_prime + m) * 255.0).round() as u8;

    (r, g, b)
}
