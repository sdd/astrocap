use astrocap_core::traits::FrameProcessor;
use astrocap_core::FrameProcessorResult::Skip;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use std::sync::Arc;
use toml::Value;
use vyd::frame::CpuFrame;
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

pub struct MaskProcessor {
    file_path: String,
}

impl MaskProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(file_path) = config
            .and_then(|c| c.get("file_path"))
            .and_then(|d| d.as_str())
        else {
            return Err(AstrocapError::PluginInvalidConfigError(
                "missing file_path in MaskProcessor Config".to_string(),
            ));
        };

        Ok(Self {
            file_path: file_path.to_string(),
        })
    }
}

impl FrameProcessor for MaskProcessor {
    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        let img = image::open(&self.file_path)
            .map_err(|e| AstrocapError::GeneralPluginError(e.to_string()))?;

        let gray_img = img.to_luma8();
        let storage: Arc<[u8]> = Arc::from(gray_img.into_raw());
        let mask = CpuFrame::from_shared(img.width(), img.height(), storage).unwrap();

        tracing::info!(
            width = mask.img.width(),
            height = mask.img.height(),
            file_path = self.file_path,
            "mask loaded"
        );
        let mask = Frame::Cpu(mask);

        ctx.put("image/mask", mask);

        Ok(())
    }

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        let start = std::time::Instant::now();

        let Ok(frame) = frame_ctx.take_frame().to_cpu(None) else {
            tracing::warn!("could not get CPU Frame");
            return Skip;
        };

        let mask = ctx.try_get_as::<Frame>("image/mask").unwrap();
        let mask = match mask {
            Frame::Cpu(mask) => mask,
            _ => {
                tracing::warn!("mask is not a CPU Frame");
                return Skip;
            }
        };

        // Get raw pixel slices for high-performance processing
        let frame_pixels = frame.img.as_raw();
        let mask_pixels = mask.img.as_raw();
        let pixel_count = frame_pixels.len();

        // Pre-allocate Vec with uninitialized memory
        #[allow(clippy::uninit_vec)]
        let mut result_data = Vec::with_capacity(pixel_count);
        unsafe {
            result_data.set_len(pixel_count);
        }

        let elapsed = start.elapsed();
        tracing::debug!("Mask setup took {:?}", elapsed);
        let start = std::time::Instant::now();

        result_data
            .iter_mut()
            .zip(frame_pixels.iter())
            .zip(mask_pixels.iter())
            .for_each(|((result, &frame_pixel), &mask_pixel)| {
                *result = frame_pixel * (mask_pixel != 0) as u8;
            });

        let elapsed = start.elapsed();
        tracing::debug!("Mask processing took {:?}", elapsed);
        let start = std::time::Instant::now();

        let masked_frame = CpuFrame::from_vec(frame.width(), frame.height(), result_data).unwrap();
        frame_ctx.frame = Frame::Cpu(masked_frame);

        let elapsed = start.elapsed();
        tracing::debug!("Mask post-processing took {:?}", elapsed);

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "mask"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
