use astrocap_core::pipeline_services::MaskFrame;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::FrameProcessorResult::Skip;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use std::sync::Arc;
use toml::Value;
use vyd::caps::{FrameMemory, FramePortSpec, FramePortSpecs, ProcessorPortSpecs};
#[cfg(feature = "gpu-wgpu")]
use vyd::frame::layout::FrameSpec;
use vyd::frame::pool::WaitPolicy;
use vyd::frame::CpuFrame;
use vyd::frame::PixelFormat;
#[cfg(feature = "gpu-wgpu")]
use vyd::gpu::recording::{GpuBindingSpec, GpuComputeProgram, GpuPlaneBinding, GpuRecording};
#[cfg(feature = "gpu-wgpu")]
use vyd::gpu::submission::submit;
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

#[cfg(feature = "gpu-wgpu")]
const MASK_SHADER: &str = r#"
@group(0) @binding(0) var input_image: texture_2d<u32>;
@group(0) @binding(1) var mask_image: texture_2d<u32>;
@group(0) @binding(2) var output_image: texture_storage_2d<rgba8uint, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output_image);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let xy = vec2<i32>(id.xy);
    let input = textureLoad(input_image, xy, 0).r;
    let mask = textureLoad(mask_image, xy, 0).r;
    textureStore(output_image, xy, vec4<u32>(select(0u, input, mask != 0u), 0u, 0u, 255u));
}
"#;

pub struct MaskProcessor {
    file_path: String,
    gpu_enabled: bool,
    #[cfg(feature = "gpu-wgpu")]
    gpu_mask: Option<Frame>,
    #[cfg(feature = "gpu-wgpu")]
    gpu_program: Option<GpuComputeProgram>,
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

        let gpu_enabled = match config.and_then(|c| c.get("gpu")) {
            None => false,
            Some(Value::Boolean(enabled)) => *enabled,
            Some(_) => {
                return Err(AstrocapError::PluginInvalidConfigError(
                    "MaskProcessor gpu must be a boolean".to_string(),
                ))
            }
        };
        #[cfg(not(feature = "gpu-wgpu"))]
        if gpu_enabled {
            return Err(AstrocapError::PluginInvalidConfigError(
                "MaskProcessor gpu = true requires the gpu-wgpu feature".to_string(),
            ));
        }
        Ok(Self {
            file_path: file_path.to_string(),
            gpu_enabled,
            #[cfg(feature = "gpu-wgpu")]
            gpu_mask: None,
            #[cfg(feature = "gpu-wgpu")]
            gpu_program: None,
        })
    }

    #[cfg(feature = "gpu-wgpu")]
    fn try_process_gpu(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, AstrocapError> {
        use std::time::Duration;

        let mask_service = ctx
            .services()
            .latest::<MaskFrame>()
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
            .ok_or_else(|| {
                AstrocapError::GeneralPluginError("mask service has not been initialized".into())
            })?;
        let (width, height) = frame_ctx.frame.dimensions();
        let (mask_width, mask_height) = mask_service.0.dimensions();
        if (mask_width, mask_height) != (width, height) {
            return Err(AstrocapError::GeneralPluginError(format!(
                "mask dimensions do not match frame: {mask_width}x{mask_height} versus {width}x{height}"
            )));
        }
        let mask = self.gpu_mask.as_mut().ok_or_else(|| {
            AstrocapError::GeneralPluginError("GPU mask has not been initialized".into())
        })?;
        let program = self.gpu_program.as_ref().ok_or_else(|| {
            AstrocapError::GeneralPluginError("GPU mask program has not been initialized".into())
        })?;
        let transfer = ctx
            .gpu_transfer()
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let gpu = ctx
            .wgpu_context()
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
            .clone();
        let wait = WaitPolicy::Timeout(Duration::from_secs(10));
        let spec = FrameSpec::new(width, height, PixelFormat::Luma8)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let input = frame_ctx
            .frame
            .try_read_gpu_as(ctx.frame_factory(), transfer, PixelFormat::Luma8, wait)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let mask = mask
            .try_read_gpu_as(ctx.frame_factory(), transfer, PixelFormat::Luma8, wait)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let output = frame_ctx
            .frame
            .try_write_gpu_as(transfer, PixelFormat::Luma8, wait)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let mut recording = GpuRecording::new(gpu);
        let input_id = recording
            .add_read(input, spec)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?[0];
        let mask_id = recording
            .add_read(mask, spec)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?[0];
        let output_id = recording
            .add_write(output, spec)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?[0];
        let group = recording
            .bind_group(
                program,
                &[
                    (0, GpuPlaneBinding::Read(input_id)),
                    (1, GpuPlaneBinding::Read(mask_id)),
                    (2, GpuPlaneBinding::Write(output_id)),
                ],
            )
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        recording
            .dispatch(program, group, [width.div_ceil(8), height.div_ceil(8), 1])
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let receipt = submit(recording)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        receipt
            .try_wait(wait)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        Ok(FrameProcessorResult::Continue)
    }
}

impl FrameProcessor for MaskProcessor {
    fn ports(&self) -> ProcessorPortSpecs {
        if self.gpu_enabled {
            ProcessorPortSpecs::passthrough(FramePortSpecs::one(FramePortSpec::new(
                PixelFormat::Luma8,
                FrameMemory::Wgpu,
            )))
        } else {
            ProcessorPortSpecs::default()
        }
    }

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

        #[cfg(feature = "gpu-wgpu")]
        if self.gpu_enabled {
            let gpu = ctx
                .wgpu_context()
                .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
                .clone();
            let transfer = ctx
                .gpu_transfer()
                .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
            let program = GpuComputeProgram::new(
                &gpu,
                MASK_SHADER,
                "main",
                &[
                    GpuBindingSpec::read(0, PixelFormat::Luma8, 0),
                    GpuBindingSpec::read(1, PixelFormat::Luma8, 0),
                    GpuBindingSpec::write(2, PixelFormat::Luma8, 0),
                ],
            )
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
            let mut gpu_mask = mask.clone_shared();
            gpu_mask
                .ensure_gpu_with(
                    ctx.frame_factory(),
                    transfer,
                    WaitPolicy::Timeout(std::time::Duration::from_secs(10)),
                )
                .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
            self.gpu_mask = Some(gpu_mask);
            self.gpu_program = Some(program);
        }

        ctx.services()
            .publish(Arc::new(MaskFrame(mask)))
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;

        Ok(())
    }

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        match self.try_process(frame_ctx, ctx) {
            Ok(result) => result,
            Err(error) => {
                tracing::error!(%error, "mask processing failed");
                Skip
            }
        }
    }

    fn try_process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, AstrocapError> {
        #[cfg(feature = "gpu-wgpu")]
        if self.gpu_enabled {
            return self.try_process_gpu(frame_ctx, ctx);
        }
        let start = std::time::Instant::now();

        let Ok(frame) = frame_ctx.frame.to_cpu(None) else {
            tracing::warn!("could not get CPU Frame");
            return Ok(Skip);
        };

        let mask = ctx
            .services()
            .latest::<MaskFrame>()
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
            .ok_or_else(|| {
                AstrocapError::GeneralPluginError("mask service has not been initialized".into())
            })?;
        let mask = match &mask.0 {
            Frame::Cpu(mask) => mask,
            _ => {
                tracing::warn!("mask is not a CPU Frame");
                return Ok(Skip);
            }
        };

        // Get raw pixel slices for high-performance processing
        let frame_pixels = frame.img.as_raw();
        let mask_pixels = mask.img.as_raw();
        let pixel_count = frame_pixels.len();
        if mask.width() != frame.width()
            || mask.height() != frame.height()
            || mask_pixels.len() != pixel_count
        {
            return Err(AstrocapError::GeneralPluginError(format!(
                "mask dimensions do not match frame: {}x{} versus {}x{}",
                mask.width(),
                mask.height(),
                frame.width(),
                frame.height()
            )));
        }

        let elapsed = start.elapsed();
        tracing::debug!("Mask setup took {:?}", elapsed);
        let start = std::time::Instant::now();

        let masked_frame = ctx
            .frame_factory()
            .create_luma8_with(
                frame.width(),
                frame.height(),
                WaitPolicy::Immediate,
                |result| {
                    for ((output, &input), &mask) in result
                        .iter_mut()
                        .zip(frame_pixels.iter())
                        .zip(mask_pixels.iter())
                    {
                        *output = input * (mask != 0) as u8;
                    }
                    Ok(())
                },
            )
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;

        let elapsed = start.elapsed();
        tracing::debug!("Mask processing took {:?}", elapsed);
        let start = std::time::Instant::now();

        frame_ctx.frame = masked_frame;

        let elapsed = start.elapsed();
        tracing::debug!("Mask post-processing took {:?}", elapsed);

        Ok(FrameProcessorResult::Continue)
    }

    fn name(&self) -> &str {
        "mask"
    }

    fn processing_type(&self) -> ProcessingType {
        if self.gpu_enabled {
            ProcessingType::Gpu
        } else {
            ProcessingType::Cpu
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_processor() -> MaskProcessor {
        MaskProcessor::new(Some(&toml::from_str("file_path = 'unused.png'").unwrap())).unwrap()
    }

    #[test]
    fn mask_processor_uses_pooled_output_and_initializes_every_pixel() {
        let mut processor = cpu_processor();
        let mut context = PipelineContext::new();
        context
            .services()
            .publish(Arc::new(MaskFrame(
                Frame::from_raw(2, 2, vec![255, 0, 255, 0]).unwrap(),
            )))
            .unwrap();
        let mut frame = FrameContext::new(Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap(), 0);
        assert!(matches!(
            processor.try_process(&mut frame, &mut context),
            Ok(FrameProcessorResult::Continue)
        ));
        assert_eq!(frame.frame.as_cpu_ref().unwrap(), &[1, 0, 3, 0]);
        assert_eq!(
            context
                .frame_factory()
                .cpu_pool()
                .snapshot()
                .unwrap()
                .active
                .len(),
            1
        );
        drop(frame);
        assert_eq!(
            context
                .frame_factory()
                .cpu_pool()
                .snapshot()
                .unwrap()
                .active
                .len(),
            0
        );
    }

    #[test]
    fn mask_size_mismatch_fails_without_replacing_input_frame() {
        let mut processor = cpu_processor();
        let mut context = PipelineContext::new();
        context
            .services()
            .publish(Arc::new(MaskFrame(
                Frame::from_raw(1, 1, vec![255]).unwrap(),
            )))
            .unwrap();
        let mut frame = FrameContext::new(Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap(), 0);
        let error = match processor.try_process(&mut frame, &mut context) {
            Ok(_) => panic!("mismatched mask should fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("mask dimensions do not match"));
        assert_eq!(frame.frame.as_cpu_ref().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn equal_pixel_count_with_different_shape_is_rejected() {
        let mut processor = cpu_processor();
        let mut context = PipelineContext::new();
        context
            .services()
            .publish(Arc::new(MaskFrame(
                Frame::from_raw(1, 4, vec![255; 4]).unwrap(),
            )))
            .unwrap();
        let mut frame = FrameContext::new(Frame::from_raw(2, 2, vec![1, 2, 3, 4]).unwrap(), 0);
        assert!(processor.try_process(&mut frame, &mut context).is_err());
        assert_eq!(frame.frame.as_cpu_ref().unwrap(), &[1, 2, 3, 4]);
    }

    #[test]
    fn missing_mask_service_fails_without_replacing_frame() {
        let mut processor = cpu_processor();
        let mut context = PipelineContext::new();
        let mut frame = FrameContext::new(Frame::from_raw(1, 1, vec![8]).unwrap(), 0);
        let error = match processor.try_process(&mut frame, &mut context) {
            Ok(_) => panic!("mask service must be initialized"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("mask service has not been initialized"));
        assert_eq!(frame.frame.as_cpu_ref().unwrap(), &[8]);
    }

    #[cfg(not(feature = "gpu-wgpu"))]
    #[test]
    fn gpu_setting_requires_feature() {
        let config: Value = toml::from_str("file_path = 'mask.png'\ngpu = true").unwrap();
        assert!(MaskProcessor::new(Some(&config))
            .err()
            .unwrap()
            .to_string()
            .contains("gpu-wgpu feature"));
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn gpu_port_and_input_errors_preserve_frame() {
        let config: Value = toml::from_str("file_path = 'mask.png'\ngpu = true").unwrap();
        let mut processor = MaskProcessor::new(Some(&config)).unwrap();
        let relation = processor.ports().relations()[0];
        assert_eq!(relation.input.memory(), FrameMemory::Wgpu);
        assert_eq!(relation.output.pixel_format(), PixelFormat::Luma8);
        let mut context = PipelineContext::new();
        let mut frame = FrameContext::new(Frame::from_raw(1, 4, vec![1, 2, 3, 4]).unwrap(), 0);
        let error = processor
            .try_process(&mut frame, &mut context)
            .err()
            .unwrap();
        assert!(error
            .to_string()
            .contains("mask service has not been initialized"));
        assert_eq!(frame.frame.as_cpu_ref().unwrap(), &[1, 2, 3, 4]);
        context
            .services()
            .publish(Arc::new(MaskFrame(
                Frame::from_raw(2, 2, vec![255; 4]).unwrap(),
            )))
            .unwrap();
        let error = processor
            .try_process(&mut frame, &mut context)
            .err()
            .unwrap();
        assert!(error.to_string().contains("mask dimensions do not match"));
        assert_eq!(frame.frame.as_cpu_ref().unwrap(), &[1, 2, 3, 4]);
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn gpu_mask_matches_cpu_and_uploads_mask_once() {
        use vyd::gpu::{GpuContextError, WgpuContext, WgpuContextOptions};
        use vyd::pipeline::{Pipeline, PipelineSession, Stage};
        use vyd::stages::{DummySink, DummySource};
        use vyd::statistics::PipelineStatistics;

        let gpu = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
            Ok(gpu) => Arc::new(gpu),
            Err(GpuContextError::Unavailable) => {
                eprintln!("GPU mask comparison unverified: no compatible adapter");
                return;
            }
            Err(error) => panic!("GPU mask hardware gate failed: {error}"),
        };
        eprintln!(
            "GPU mask adapter={} backend={:?}",
            gpu.adapter_info().name,
            gpu.adapter_info().backend
        );
        let path = std::env::temp_dir().join(format!(
            "astrocap-mask-gpu-{}-{}.png",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mask_bytes: Vec<u8> = (0..21).map(|i| if i % 3 == 0 { 0 } else { 255 }).collect();
        image::GrayImage::from_raw(7, 3, mask_bytes)
            .unwrap()
            .save(&path)
            .unwrap();
        let make_config = |gpu: bool| -> Value {
            let mut fields = toml::map::Map::new();
            fields.insert(
                "file_path".into(),
                Value::String(path.to_string_lossy().into_owned()),
            );
            fields.insert("gpu".into(), Value::Boolean(gpu));
            Value::Table(fields)
        };
        let mut cpu = MaskProcessor::new(Some(&make_config(false))).unwrap();
        let mut cpu_ctx = PipelineContext::new();
        cpu.pipeline_ctx_init(&mut cpu_ctx).unwrap();

        let stats = Arc::new(PipelineStatistics::new());
        let mut gpu_ctx = PipelineContext::new();
        gpu_ctx.set_wgpu_context(gpu).unwrap();
        let pipeline = Pipeline {
            source: Stage::new(Box::new(DummySource::new(None).unwrap()), "source"),
            stages: vec![Stage::new(
                Box::new(MaskProcessor::new(Some(&make_config(true))).unwrap()),
                "mask",
            )],
            sink: Stage::new(Box::new(DummySink::new(None).unwrap()), "sink"),
        };
        let mut session = PipelineSession::new(gpu_ctx, pipeline, stats.clone());
        session.start().unwrap();
        for frame_index in 0..2 {
            let source: Vec<u8> = (0..21).map(|i| (i * 11 + frame_index * 7) as u8).collect();
            let mut cpu_frame =
                FrameContext::new(Frame::from_raw(7, 3, source.clone()).unwrap(), frame_index);
            cpu.try_process(&mut cpu_frame, &mut cpu_ctx).unwrap();
            let expected = cpu_frame.frame.as_cpu_ref().unwrap().to_vec();

            let mut gpu_frame =
                FrameContext::new(Frame::from_raw(7, 3, source).unwrap(), frame_index);
            gpu_frame.put("probe", 42u32);
            assert!(matches!(
                session.process(0, &mut gpu_frame),
                Ok(FrameProcessorResult::Continue)
            ));
            assert_eq!(gpu_frame.frame_index, frame_index);
            assert_eq!(*gpu_frame.get_as::<u32>("probe"), 42);
            let actual = gpu_frame
                .frame
                .try_read_cpu_as_with_transfer(
                    session.frame_factory(),
                    session.gpu_transfer().unwrap(),
                    PixelFormat::Luma8,
                    WaitPolicy::Timeout(std::time::Duration::from_secs(10)),
                )
                .unwrap()
                .luma8_bytes()
                .unwrap()
                .to_vec();
            assert_eq!(actual, expected);
        }
        let snapshot = stats.snapshot();
        let uploads: u64 = snapshot
            .frame_operations
            .values()
            .flat_map(|operations| operations.iter())
            .filter(|(name, _)| name.starts_with("upload_luma8@"))
            .map(|(_, counters)| counters.completed)
            .sum();
        let downloads: u64 = snapshot
            .frame_operations
            .values()
            .flat_map(|operations| operations.iter())
            .filter(|(name, _)| name.starts_with("download_luma8@"))
            .map(|(_, counters)| counters.completed)
            .sum();
        assert_eq!(
            uploads, 3,
            "one mask upload and one input upload per frame: {snapshot:?}"
        );
        assert_eq!(
            downloads, 2,
            "one requested output readback per frame: {snapshot:?}"
        );
        session.finish(Ok(())).unwrap();
        std::fs::remove_file(path).unwrap();
    }
}
