use astrocap_core::frame_metadata::MedianFrame;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::FrameProcessorResult::Skip;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use std::sync::Arc;
use toml::Value;
use vyd::caps::{FrameMemory, FramePortSpec, FramePortSpecs, ProcessorPortSpecs};
#[cfg(feature = "gpu-wgpu")]
use vyd::frame::layout::FrameSpec;
use vyd::frame::pool::WaitPolicy;
use vyd::frame::PixelFormat;
#[cfg(feature = "gpu-wgpu")]
use vyd::gpu::recording::{GpuBindingSpec, GpuComputeProgram, GpuPlaneBinding, GpuRecording};
#[cfg(feature = "gpu-wgpu")]
use vyd::gpu::submission::submit;
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;

#[cfg(feature = "gpu-wgpu")]
const SUBTRACT_SHADER: &str = r#"
@group(0) @binding(0) var input_image: texture_2d<u32>;
@group(0) @binding(1) var background_image: texture_2d<u32>;
@group(0) @binding(2) var output_image: texture_storage_2d<rgba8uint, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output_image);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let xy = vec2<i32>(id.xy);
    let input = textureLoad(input_image, xy, 0).r;
    let background = textureLoad(background_image, xy, 0).r;
    let difference = select(0u, input - background, input >= background);
    textureStore(output_image, xy, vec4<u32>(difference, 0u, 0u, 255u));
}
"#;

pub struct ImgSubberProcessor {
    subtractand_key: String,
    gpu_enabled: bool,
    #[cfg(feature = "gpu-wgpu")]
    gpu_program: Option<GpuComputeProgram>,
    #[cfg(feature = "gpu-wgpu")]
    gpu_subtractand: Option<(Arc<Frame>, Frame)>,
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

        let gpu_enabled = match config.and_then(|c| c.get("gpu")) {
            None => false,
            Some(Value::Boolean(enabled)) => *enabled,
            Some(_) => {
                return Err(AstrocapError::PluginInvalidConfigError(
                    "ImgSubberProcessor gpu must be a boolean".into(),
                ))
            }
        };
        #[cfg(not(feature = "gpu-wgpu"))]
        if gpu_enabled {
            return Err(AstrocapError::PluginInvalidConfigError(
                "ImgSubberProcessor gpu = true requires the gpu-wgpu feature".into(),
            ));
        }

        Ok(Self {
            subtractand_key,
            gpu_enabled,
            #[cfg(feature = "gpu-wgpu")]
            gpu_program: None,
            #[cfg(feature = "gpu-wgpu")]
            gpu_subtractand: None,
        })
    }

    fn subtractand<'a>(&self, frame_ctx: &'a FrameContext) -> Option<&'a Arc<Frame>> {
        if self.subtractand_key == "median" {
            if let Some(median) = frame_ctx.get_typed::<MedianFrame>() {
                return Some(&median.0);
            }
        }
        let metadata_key = format!("video/{}", self.subtractand_key);
        let Some(value) = frame_ctx.metadata.get(&metadata_key) else {
            tracing::error!("{metadata_key} not present in FrameContext!");
            return None;
        };
        let Some(subtractand) = value.downcast_ref::<Arc<Frame>>() else {
            tracing::error!("{metadata_key} not downcastable to Frame!");
            return None;
        };
        Some(subtractand)
    }

    #[cfg(feature = "gpu-wgpu")]
    fn try_process_gpu(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, AstrocapError> {
        use std::time::Duration;

        let Some(subtractand) = self.subtractand(frame_ctx).cloned() else {
            return Ok(Skip);
        };
        let (width, height) = frame_ctx.frame.dimensions();
        if subtractand.dimensions() != (width, height) {
            return Err(AstrocapError::GeneralPluginError(format!(
                "subtractand dimensions do not match frame: {:?} versus {:?}",
                subtractand.dimensions(),
                (width, height)
            )));
        }
        let program = self.gpu_program.as_ref().ok_or_else(|| {
            AstrocapError::GeneralPluginError(
                "GPU subtraction program has not been initialized".into(),
            )
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

        // Async median publishes one Arc for many input frames. Keep a frame with
        // its GPU replica across calls; cloning the CPU Arc's frame each time
        // would discard the replica cache and upload the same median repeatedly.
        if !self
            .gpu_subtractand
            .as_ref()
            .is_some_and(|(cached, _)| Arc::ptr_eq(cached, &subtractand))
        {
            let mut cached = subtractand.clone_shared();
            cached
                .try_read_gpu_as(ctx.frame_factory(), transfer, PixelFormat::Luma8, wait)
                .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
            self.gpu_subtractand = Some((subtractand, cached));
        }
        let background = self
            .gpu_subtractand
            .as_mut()
            .unwrap()
            .1
            .try_read_gpu_as(ctx.frame_factory(), transfer, PixelFormat::Luma8, wait)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let input = frame_ctx
            .frame
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
        let background_id = recording
            .add_read(background, spec)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?[0];
        let output_id = recording
            .add_write(output, spec)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?[0];
        let group = recording
            .bind_group(
                program,
                &[
                    (0, GpuPlaneBinding::Read(input_id)),
                    (1, GpuPlaneBinding::Read(background_id)),
                    (2, GpuPlaneBinding::Write(output_id)),
                ],
            )
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        recording
            .dispatch(program, group, [width.div_ceil(8), height.div_ceil(8), 1])
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        submit(recording)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
            .try_wait(wait)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        Ok(FrameProcessorResult::Continue)
    }
}

impl FrameProcessor for ImgSubberProcessor {
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
        #[cfg(feature = "gpu-wgpu")]
        if self.gpu_enabled {
            let gpu = ctx
                .wgpu_context()
                .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
            self.gpu_program = Some(
                GpuComputeProgram::new(
                    gpu,
                    SUBTRACT_SHADER,
                    "main",
                    &[
                        GpuBindingSpec::read(0, PixelFormat::Luma8, 0),
                        GpuBindingSpec::read(1, PixelFormat::Luma8, 0),
                        GpuBindingSpec::write(2, PixelFormat::Luma8, 0),
                    ],
                )
                .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?,
            );
            self.gpu_subtractand = None;
        }
        #[cfg(not(feature = "gpu-wgpu"))]
        let _ = ctx;
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
                tracing::error!(%error, "image subtraction failed");
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

        tracing::trace!(
            "ImgSubberProcessor received frame with metadata keys: {:?}",
            frame_ctx.metadata.keys().collect::<Vec<_>>()
        );

        let Ok(frame) = frame_ctx.frame.to_cpu(None) else {
            tracing::warn!("No frame to process");
            return Ok(Skip);
        };

        let Some(subtractand) = self.subtractand(frame_ctx) else {
            return Ok(Skip);
        };

        let Some(subtractand_img) = subtractand.as_cpu_image() else {
            tracing::error!("subtractand is not a CPU Frame!");
            return Ok(Skip);
        };

        // Get raw pixel slices for high-performance processing
        let frame_pixels = frame.img.as_raw();
        let subtractand_pixels = subtractand_img.as_raw();
        if subtractand.dimensions() != frame.dimensions()
            || subtractand_pixels.len() != frame_pixels.len()
        {
            return Err(AstrocapError::GeneralPluginError(format!(
                "subtractand dimensions do not match frame: {:?} versus {:?}",
                subtractand.dimensions(),
                frame.dimensions()
            )));
        }

        let elapsed = start.elapsed();
        tracing::debug!("ImgSubber setup took {:?}", elapsed);
        let start = std::time::Instant::now();

        let subtracted_frame = ctx
            .frame_factory()
            .create_luma8_with(
                frame.width(),
                frame.height(),
                WaitPolicy::Immediate,
                |result| {
                    for ((output, &input), &subtractand) in result
                        .iter_mut()
                        .zip(frame_pixels.iter())
                        .zip(subtractand_pixels.iter())
                    {
                        *output = input.saturating_sub(subtractand);
                    }
                    Ok(())
                },
            )
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;

        let elapsed = start.elapsed();
        tracing::debug!("ImgSubber processing took {:?}", elapsed);
        let start = std::time::Instant::now();

        frame_ctx.frame = subtracted_frame;

        let elapsed = start.elapsed();
        tracing::debug!("ImgSubber post-processing took {:?}", elapsed);

        Ok(FrameProcessorResult::Continue)
    }

    fn name(&self) -> &str {
        "img_subber"
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

    fn processor() -> ImgSubberProcessor {
        let config: Value = toml::from_str("subtractand_key = 'median'").unwrap();
        ImgSubberProcessor::new(Some(&config)).unwrap()
    }

    #[test]
    fn subtraction_uses_pooled_output_and_saturates() {
        let mut context = PipelineContext::new();
        let mut frame = FrameContext::new(Frame::from_raw(2, 2, vec![9, 3, 0, 255]).unwrap(), 0);
        frame.put_typed(MedianFrame(Arc::new(
            Frame::from_raw(2, 2, vec![2, 5, 0, 100]).unwrap(),
        )));
        assert!(matches!(
            processor().try_process(&mut frame, &mut context),
            Ok(FrameProcessorResult::Continue)
        ));
        assert_eq!(frame.frame.as_cpu_ref().unwrap(), &[7, 0, 0, 155]);
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
    fn equal_area_shape_mismatch_preserves_input() {
        let mut context = PipelineContext::new();
        let mut frame = FrameContext::new(Frame::from_raw(2, 2, vec![9, 3, 0, 255]).unwrap(), 0);
        frame.put(
            "video/median",
            Arc::new(Frame::from_raw(1, 4, vec![2, 5, 0, 100]).unwrap()),
        );
        let error = match processor().try_process(&mut frame, &mut context) {
            Ok(_) => panic!("shape mismatch should fail"),
            Err(error) => error,
        };
        assert!(error
            .to_string()
            .contains("subtractand dimensions do not match"));
        assert_eq!(frame.frame.as_cpu_ref().unwrap(), &[9, 3, 0, 255]);
        assert!(context
            .frame_factory()
            .cpu_pool()
            .snapshot()
            .unwrap()
            .active
            .is_empty());
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn gpu_subtraction_matches_cpu_and_reuses_median_upload() {
        use vyd::gpu::{GpuContextError, WgpuContext, WgpuContextOptions};
        use vyd::pipeline::{Pipeline, PipelineSession, Stage};
        use vyd::stages::{DummySink, DummySource};
        use vyd::statistics::PipelineStatistics;

        let gpu = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
            Ok(gpu) => Arc::new(gpu),
            Err(GpuContextError::Unavailable) => {
                eprintln!("GPU subtraction comparison unverified: no compatible adapter");
                return;
            }
            Err(error) => panic!("GPU subtraction hardware gate failed: {error}"),
        };
        eprintln!(
            "GPU subtraction adapter={} backend={:?}",
            gpu.adapter_info().name,
            gpu.adapter_info().backend
        );
        let make_config = |gpu: bool| -> Value {
            let mut fields = toml::map::Map::new();
            fields.insert("subtractand_key".into(), Value::String("median".into()));
            fields.insert("gpu".into(), Value::Boolean(gpu));
            Value::Table(fields)
        };
        let mut cpu = ImgSubberProcessor::new(Some(&make_config(false))).unwrap();
        let mut cpu_ctx = PipelineContext::new();
        cpu.pipeline_ctx_init(&mut cpu_ctx).unwrap();

        let stats = Arc::new(PipelineStatistics::new());
        let mut gpu_ctx = PipelineContext::new();
        gpu_ctx.set_wgpu_context(gpu).unwrap();
        let pipeline = Pipeline {
            source: Stage::new(Box::new(DummySource::new(None).unwrap()), "source"),
            stages: vec![Stage::new(
                Box::new(ImgSubberProcessor::new(Some(&make_config(true))).unwrap()),
                "img_subber",
            )],
            sink: Stage::new(Box::new(DummySink::new(None).unwrap()), "sink"),
        };
        let mut session = PipelineSession::new(gpu_ctx, pipeline, stats.clone());
        session.start().unwrap();

        let medians = [
            Arc::new(Frame::from_raw(7, 3, (0..21).map(|i| (i * 13) as u8).collect()).unwrap()),
            Arc::new(Frame::from_raw(7, 3, (0..21).map(|i| (i * 7) as u8).collect()).unwrap()),
        ];
        for frame_index in 0..3 {
            let source: Vec<u8> = (0..21).map(|i| (i * 11 + frame_index * 7) as u8).collect();
            let median = medians[usize::from(frame_index == 2)].clone();
            let mut cpu_frame =
                FrameContext::new(Frame::from_raw(7, 3, source.clone()).unwrap(), frame_index);
            cpu_frame.put_typed(MedianFrame(median.clone()));
            cpu.try_process(&mut cpu_frame, &mut cpu_ctx).unwrap();
            let expected = cpu_frame.frame.as_cpu_ref().unwrap().to_vec();

            let mut gpu_frame =
                FrameContext::new(Frame::from_raw(7, 3, source).unwrap(), frame_index);
            gpu_frame.put_typed(MedianFrame(median));
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
        assert_eq!(uploads, 5, "two medians and three inputs: {snapshot:?}");
        assert_eq!(downloads, 3, "three requested readbacks: {snapshot:?}");
        session.finish(Ok(())).unwrap();
    }
}
