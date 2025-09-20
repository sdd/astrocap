use rerun::RecordingStream;
use toml::Value;

use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::structs::Detection;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::{AstrocapError, DumpManager, FrameContext, FrameProcessorResult};

use crate::config::Config;
use crate::model::Track;
use crate::traits::HypothesisTracker;

pub struct TrackerProcessor {
    tracker: Box<dyn HypothesisTracker>,
}

impl TrackerProcessor {
    pub fn new(raw_config: Option<&Value>) -> Result<Self, AstrocapError> {
        let config: Config = match raw_config {
            Some(config_value) => config_value.clone().try_into().map_err(|e| {
                AstrocapError::PluginInvalidConfigError(format!("Tracker: {}", e.to_string()))
            })?,
            None => Config::default(),
        };

        let tracker = config.tracker_strategy.create_tracker(raw_config)?;

        Ok(Self { tracker })
    }
}

impl FrameProcessor for TrackerProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let detections = frame_ctx.get_as::<Vec<Detection>>("detected_points");

        // map DetectedPoint to Detection
        let detections = detections
            .iter()
            .map(|d| Detection::new(d.position.x, d.position.y, d.amplitude))
            .collect::<Vec<_>>();

        frame_ctx.put("tracker/detections", detections.clone());

        tracing::debug!("processing frame");

        self.tracker
            .process_frame(&detections, frame_ctx.frame_index);

        // TODO: ensure that tracks are reportable for any tracker
        if let Ok(rec) = ctx.try_get_as::<RecordingStream>("rerun") {
            self.tracker.log_to_rerun(&rec, frame_ctx);
        }

        // Get the dump manager from the pipeline context
        let entry = ctx.entry("dump_manager");
        if let std::collections::hash_map::Entry::Occupied(mut entry) = entry {
            let dump_manager = entry
                .get_mut()
                .downcast_mut::<DumpManager>()
                .expect("dump_manager is not a DumpManager");

            if let Err(e) = self.tracker.dump(dump_manager, frame_ctx.frame_index) {
                tracing::error!("Error dumping: {}", e);
            }
        }

        FrameProcessorResult::Continue
    }

    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        ctx.put("model/tracks", Vec::<Track>::new());

        Ok(())
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        Ok(())
    }

    fn name(&self) -> &str {
        "tracker"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

// Helper function for HSV to RGB conversion (same as in model processor)
#[allow(unused)]
pub(crate) fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
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
