use crate::config::Config;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::structs::DetectedPoint;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use toml::Value;

use crate::factory::Factory;
use crate::model::Track;
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
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: Config = config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!("Tracker:: {}", e.to_string()))
        })?;

        let initiator = Factory::create_initiator(&config)?;
        let terminator = Factory::create_terminator(&config)?;
        let associator = Factory::create_associator(&config)?;
        let updater = Factory::create_updater(&config)?;
        let predictor = Factory::create_predictor(&config)?;

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
}

impl FrameProcessor for TrackerProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let detections = frame_ctx.get_as::<Vec<DetectedPoint>>("model/detected_points");

        // associate detections with existing tracks
        let (associations, unassociated_detections) =
            self.associator.associate(detections, &self.tracks);

        for (track_idx, track) in self.tracks.iter_mut().enumerate() {
            track.age += 1;

            if let Some(detection_idx) = associations.get(&track_idx) {
                // for associated tracks: update the track with the detection via the updater
                self.updater.update(track, &detections[*detection_idx])
            } else {
                // for unassociated tracks: update the track with the predictor
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
        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "tracker"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
