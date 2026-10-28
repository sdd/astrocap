use std::fs::File;

use crate::config::Config;
use astrocap_core::frame_metadata::DetectedPoints;
use astrocap_core::structs::Detection;
use astrocap_core::{AstrocapError, FrameContext, FrameSource};
use vyd::frame::pool::WaitPolicy;
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

pub struct RecordedDetectionsSource {
    config: Config,
    frame_index: usize,
    detections: Vec<Vec<Detection>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use vyd::frame::CpuAllocationOrigin;
    use vyd::frame::factory::FrameFactory;
    use vyd::frame::pool::{BufferPoolManager, PoolLimits};

    #[test]
    fn configured_dummy_frames_use_pool_and_retry_after_exhaustion() {
        let path =
            std::env::temp_dir().join(format!("recorded-detections-{}.json", std::process::id()));
        std::fs::write(&path, "[[],[]]").unwrap();
        let config: toml::Value = toml::from_str(&format!(
            "input_path = '{}'\nframe_width = 3\nframe_height = 2",
            path.display()
        ))
        .unwrap();
        let mut source = RecordedDetectionsSource::new(Some(&config)).unwrap();
        let mut ctx = PipelineContext::new();
        ctx.set_frame_factory(FrameFactory::new(BufferPoolManager::new(
            PoolLimits::new(Some(1), Some(6)).unwrap(),
        )));
        let first = source.try_next_frame(&mut ctx).unwrap().unwrap();
        assert_eq!(first.get_typed::<DetectedPoints>().unwrap().0.len(), 0);
        assert!(first.metadata.get("detected_points").is_none());
        assert_eq!(first.frame.dimensions(), (3, 2));
        assert_eq!(first.frame.as_cpu_ref().unwrap(), &[0; 6]);
        assert_eq!(
            first
                .frame
                .to_cpu(None)
                .unwrap()
                .img
                .as_raw()
                .allocation_origin(),
            CpuAllocationOrigin::Pool
        );
        assert!(source.try_next_frame(&mut ctx).is_err());
        assert_eq!(source.frame_index, 1);
        drop(first);
        assert!(source.try_next_frame(&mut ctx).unwrap().is_some());
        assert!(source.try_next_frame(&mut ctx).unwrap().is_none());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_zero_dimensions() {
        let config: toml::Value = toml::from_str("frame_width = 0\nframe_height = 2").unwrap();
        assert!(RecordedDetectionsSource::new(Some(&config)).is_err());
    }
}

impl RecordedDetectionsSource {
    pub fn new(params: Option<&toml::Value>) -> Result<Self, AstrocapError> {
        let config = match params {
            Some(params_value) => params_value.clone().try_into()?,
            None => Config::default(),
        };
        if config.frame_width == 0 || config.frame_height == 0 {
            return Err(AstrocapError::PluginInvalidConfigError(
                "recorded detections frame dimensions must be positive".into(),
            ));
        }
        if let Some(pixels) = &config.frame_pixels {
            let expected = usize::try_from(config.frame_width)
                .ok()
                .and_then(|width| {
                    usize::try_from(config.frame_height)
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .ok_or_else(|| {
                    AstrocapError::PluginInvalidConfigError(
                        "recorded frame dimensions overflow".into(),
                    )
                })?;
            if pixels.len() != expected {
                return Err(AstrocapError::PluginInvalidConfigError(format!(
                    "recorded frame_pixels has {} bytes; expected {expected} for {}x{} packed GRAY8",
                    pixels.len(),
                    config.frame_width,
                    config.frame_height
                )));
            }
        }

        // Load the detections from the JSON file
        let detections = Self::load_detections(&config.input_path)?;
        tracing::info!(
            "Loaded {} frames of detections from {:?}",
            detections.len(),
            config.input_path
        );

        Ok(Self {
            config,
            frame_index: 0,
            detections,
        })
    }

    fn load_detections(
        input_path: &std::path::PathBuf,
    ) -> Result<Vec<Vec<Detection>>, AstrocapError> {
        let file = File::open(input_path).map_err(|e| {
            AstrocapError::GeneralPluginError(format!(
                "Failed to open detections file {:?}: {}",
                input_path, e
            ))
        })?;

        let detections: Vec<Vec<Detection>> = serde_json::from_reader(file).map_err(|e| {
            AstrocapError::GeneralPluginError(format!(
                "Failed to parse detections from {:?}: {}",
                input_path, e
            ))
        })?;

        Ok(detections)
    }
}

impl FrameSource for RecordedDetectionsSource {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext> {
        match self.try_next_frame(ctx) {
            Ok(frame) => frame,
            Err(error) => {
                tracing::error!(%error, "recorded detections source failed");
                None
            }
        }
    }

    fn try_next_frame(
        &mut self,
        ctx: &mut PipelineContext,
    ) -> Result<Option<FrameContext>, AstrocapError> {
        // Check if we've reached the end of our recorded detections
        if self.frame_index >= self.detections.len() {
            return Ok(None);
        }

        // Create a dummy frame
        let dummy_frame = ctx
            .frame_factory()
            .create_luma8_with(
                self.config.frame_width,
                self.config.frame_height,
                WaitPolicy::Immediate,
                |output| {
                    if let Some(pixels) = &self.config.frame_pixels {
                        output.copy_from_slice(pixels);
                    } else {
                        output.fill(0);
                    }
                    Ok(())
                },
            )
            .map_err(|error| {
                AstrocapError::GeneralPluginError(format!(
                    "recorded detections frame {}: {error}",
                    self.frame_index
                ))
            })?;

        // Create frame context
        let mut frame_ctx = FrameContext::new(dummy_frame, self.frame_index);

        // Convert our Detection format to DetectedPoint format (what the pipeline expects)
        let detected_points: Vec<Detection> =
            self.detections[self.frame_index].iter().cloned().collect();

        // Inject the detections into the frame context metadata
        // This is the same key that the point detector processor uses
        frame_ctx.put_typed(DetectedPoints(detected_points));

        self.frame_index += 1;

        Ok(Some(frame_ctx))
    }

    fn name(&self) -> &str {
        "recorded_detections_source"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
