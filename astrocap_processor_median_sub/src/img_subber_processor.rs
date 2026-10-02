use astrocap_core::traits::FrameProcessor;
use astrocap_core::FrameProcessorResult::Skip;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use std::sync::Arc;
use toml::Value;
use vyd::frame::CpuFrame;
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

pub struct ImgSubberProcessor {
    subtractand_key: String,
}

impl ImgSubberProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let subtractand_key = config
            .and_then(|c| c.get("subtractand_key"))
            .and_then(|d| d.as_str())
            .map(|d| d.to_string());

        let Some(subtractand_key) = subtractand_key else {
            return Err(AstrocapError::GeneralPluginError(
                "subtractand_key not present in ImgSubber Config".to_string(),
            ));
        };

        Ok(Self { subtractand_key })
    }
}

impl FrameProcessor for ImgSubberProcessor {
    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        _ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let start = std::time::Instant::now();

        tracing::trace!(
            "ImgSubberProcessor received frame with metadata keys: {:?}",
            frame_ctx.metadata.keys().collect::<Vec<_>>()
        );

        let Ok(frame) = frame_ctx.take_frame().to_cpu(None) else {
            tracing::warn!("No frame to process");
            return Skip;
        };

        let metadata_key = format!("video/{}", self.subtractand_key);
        let Some(subtractand) = frame_ctx.metadata.get(&metadata_key) else {
            tracing::error!("{metadata_key} not present in FrameContext!");
            return Skip;
        };

        let Some(subtractand) = subtractand.downcast_ref::<Arc<Frame>>() else {
            tracing::error!("{metadata_key} not downcastable to Frame!");
            return Skip;
        };

        let Some(subtractand_img) = subtractand.as_cpu_image() else {
            tracing::error!("{metadata_key} not CPU Frame!");
            return Skip;
        };

        // Get raw pixel slices for high-performance processing
        let frame_pixels = frame.img.as_raw();
        let subtractand_pixels = subtractand_img.as_raw();
        let pixel_count = frame_pixels.len();

        // Pre-allocate Vec with uninitialized memory
        let mut result_data = Vec::with_capacity(pixel_count);
        unsafe {
            result_data.set_len(pixel_count);
        }

        let elapsed = start.elapsed();
        tracing::debug!("ImgSubber setup took {:?}", elapsed);
        let start = std::time::Instant::now();

        // Ultra-fast iterator-based subtraction with saturation
        result_data
            .iter_mut()
            .zip(frame_pixels.iter())
            .zip(subtractand_pixels.iter())
            .for_each(|((result, &frame_pixel), &subtractand_pixel)| {
                *result = frame_pixel.saturating_sub(subtractand_pixel);
            });

        let elapsed = start.elapsed();
        tracing::debug!("ImgSubber processing took {:?}", elapsed);
        let start = std::time::Instant::now();

        // Direct Vec to CpuFrame - avoid Arc overhead
        let subtracted_frame =
            CpuFrame::from_vec(frame.width(), frame.height(), result_data).unwrap();
        frame_ctx.frame = Frame::Cpu(subtracted_frame);

        let elapsed = start.elapsed();
        tracing::debug!("ImgSubber post-processing took {:?}", elapsed);

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "img_subber"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
