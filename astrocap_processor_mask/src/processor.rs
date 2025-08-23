use crate::map_colors::map_colors;
use astrocap_core::frame::CpuFrame;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::statistics::ProcessingType;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::FrameProcessorResult::Skip;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use image::Luma;
use std::sync::Arc;
use toml::Value;

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
                "missing file_ath in MaskProcessor Config".to_string(),
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

        let masked_frame = map_colors(&frame.img, &mask.img, |p, q| {
            Luma([(p[0]) * (q[0] != 0) as u8])
        });

        let cpu_storage: Arc<[u8]> = Arc::from(masked_frame.into_raw());
        let integration_frame =
            CpuFrame::from_shared(frame.width(), frame.height(), cpu_storage).unwrap();

        frame_ctx.frame = Frame::Cpu(integration_frame);

        FrameProcessorResult::Continue
    }

    fn name(&self) -> &str {
        "mask"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}
