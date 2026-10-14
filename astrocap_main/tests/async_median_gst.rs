use std::sync::{Arc, Mutex};
use std::time::Duration;

use astrocap_core::{AstrocapError, FrameContext, FrameProcessorResult};
use astrocap_processor_median_sub::{ImgSubberProcessor, MedianProcessor};
use vyd::frame::pool::WaitPolicy;
use vyd::pipeline::{Pipeline, PipelineContext, Stage};
use vyd::statistics::{PipelineStatistics, ProcessingType};
use vyd::{FrameProcessor, FrameSink, FrameSource};

struct ValuesSource(usize);

impl FrameSource for ValuesSource {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext> {
        self.try_next_frame(ctx).unwrap()
    }

    fn try_next_frame(
        &mut self,
        ctx: &mut PipelineContext,
    ) -> Result<Option<FrameContext>, AstrocapError> {
        let values = [3, 8, 12];
        let Some(&value) = values.get(self.0) else {
            return Ok(None);
        };
        let frame = ctx
            .frame_factory()
            .create_luma8_with(4, 1, WaitPolicy::Immediate, |bytes| {
                bytes.fill(value);
                Ok(())
            })
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let frame = FrameContext::new(frame, self.0);
        self.0 += 1;
        Ok(Some(frame))
    }

    fn name(&self) -> &str {
        "values"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

struct SpaceAdmissions(Duration);

impl FrameProcessor for SpaceAdmissions {
    fn process(&mut self, _: &mut FrameContext, _: &mut PipelineContext) -> FrameProcessorResult {
        std::thread::sleep(self.0);
        FrameProcessorResult::Continue
    }
    fn name(&self) -> &str {
        "space-admissions"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

struct CheckSink(Arc<Mutex<Vec<(usize, u8)>>>);

impl FrameSink for CheckSink {
    fn consume(&mut self, frame: &mut FrameContext, _: &mut PipelineContext) {
        let pixels = frame
            .frame
            .try_read_cpu_luma8(WaitPolicy::Immediate)
            .unwrap();
        self.0.lock().unwrap().push((frame.frame_index, pixels[0]));
    }
    fn name(&self) -> &str {
        "check"
    }
    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

#[test]
fn async_median_reaches_subtractor_with_earlier_pixels_through_gst() {
    let config: vyd::toml::Value = vyd::toml::from_str("async = true\nwindow_size = 0").unwrap();
    let subtract_config: vyd::toml::Value =
        vyd::toml::from_str("subtractand_key = 'median'").unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stats = Arc::new(PipelineStatistics::new());
    let pipeline = Pipeline {
        source: Stage::new(Box::new(ValuesSource(0)), "values"),
        stages: vec![
            Stage::new(
                Box::new(MedianProcessor::new(Some(&config)).unwrap()),
                "median",
            ),
            Stage::new(
                Box::new(SpaceAdmissions(Duration::from_millis(50))),
                "space",
            ),
            Stage::new(
                Box::new(ImgSubberProcessor::new(Some(&subtract_config)).unwrap()),
                "subtract",
            ),
        ],
        sink: Stage::new(Box::new(CheckSink(seen.clone())), "check"),
    };
    vyd_gstreamer::runtime::run_pipeline(PipelineContext::new(), pipeline, Some(stats.clone()))
        .unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![(1, 5), (2, 4)]);
    let snapshot = stats.snapshot();
    assert_eq!(snapshot.frame_count, 3);
    assert_eq!(snapshot.stages["check"].call_count, 2);
    assert!(snapshot.stages["background/Median generation"].call_count >= 1);
    assert_eq!(snapshot.memory_operations["gst-adapter/pack_zero_copy"], 11);
    assert_eq!(
        snapshot.memory_operations["gst-adapter/unpack_zero_copy"],
        11
    );
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn async_median_reaches_gpu_subtractor_through_gst() {
    use vyd::gpu::{GpuContextError, WgpuContext, WgpuContextOptions};

    let gpu = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
        Ok(gpu) => Arc::new(gpu),
        Err(GpuContextError::Unavailable) => {
            eprintln!("GPU GST subtraction integration unverified: no compatible adapter");
            return;
        }
        Err(error) => panic!("GPU GST subtraction hardware gate failed: {error}"),
    };
    let median_config: vyd::toml::Value =
        vyd::toml::from_str("async = true\nwindow_size = 0").unwrap();
    let subtract_config: vyd::toml::Value =
        vyd::toml::from_str("subtractand_key = 'median'\ngpu = true").unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stats = Arc::new(PipelineStatistics::new());
    let pipeline = Pipeline {
        source: Stage::new(Box::new(ValuesSource(0)), "values"),
        stages: vec![
            Stage::new(
                Box::new(MedianProcessor::new(Some(&median_config)).unwrap()),
                "median",
            ),
            Stage::new(
                Box::new(SpaceAdmissions(Duration::from_millis(50))),
                "space",
            ),
            Stage::new(
                Box::new(ImgSubberProcessor::new(Some(&subtract_config)).unwrap()),
                "subtract",
            ),
        ],
        sink: Stage::new(Box::new(CheckSink(seen.clone())), "check"),
    };
    let mut ctx = PipelineContext::new();
    ctx.set_wgpu_context(gpu).unwrap();
    vyd_gstreamer::runtime::run_pipeline(ctx, pipeline, Some(stats.clone())).unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![(1, 5), (2, 4)]);
    assert_eq!(stats.snapshot().frame_count, 3);
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_async_median_reaches_cpu_subtractor_through_gst() {
    use vyd::gpu::{GpuContextError, WgpuContext, WgpuContextOptions};

    let gpu = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
        Ok(gpu) => Arc::new(gpu),
        Err(GpuContextError::Unavailable) => {
            eprintln!("GPU median integration unverified: no compatible adapter");
            return;
        }
        Err(error) => panic!("GPU median hardware gate failed: {error}"),
    };
    let median_config: vyd::toml::Value =
        vyd::toml::from_str("async = true\ngpu = true\nwindow_size = 0").unwrap();
    let subtract_config: vyd::toml::Value =
        vyd::toml::from_str("subtractand_key = 'median'").unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stats = Arc::new(PipelineStatistics::new());
    let pipeline = Pipeline {
        source: Stage::new(Box::new(ValuesSource(0)), "values"),
        stages: vec![
            Stage::new(
                Box::new(MedianProcessor::new(Some(&median_config)).unwrap()),
                "median",
            ),
            Stage::new(
                Box::new(SpaceAdmissions(Duration::from_millis(500))),
                "space",
            ),
            Stage::new(
                Box::new(ImgSubberProcessor::new(Some(&subtract_config)).unwrap()),
                "subtract",
            ),
        ],
        sink: Stage::new(Box::new(CheckSink(seen.clone())), "check"),
    };
    let mut ctx = PipelineContext::new();
    ctx.set_wgpu_context(gpu).unwrap();
    vyd_gstreamer::runtime::run_pipeline(ctx, pipeline, Some(stats.clone())).unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![(1, 5), (2, 4)]);
    let snapshot = stats.snapshot();
    assert!(snapshot.stages["background/Median generation"].call_count >= 2);
}

#[cfg(feature = "gpu-wgpu")]
#[test]
fn gpu_mask_and_subtractor_run_as_one_segment() {
    use astrocap_processor_mask::MaskProcessor;
    use vyd::gpu::{GpuContextError, WgpuContext, WgpuContextOptions};

    let gpu = match WgpuContext::request_blocking(WgpuContextOptions::default()) {
        Ok(gpu) => Arc::new(gpu),
        Err(GpuContextError::Unavailable) => {
            eprintln!("GPU mask/subtraction segment unverified: no compatible adapter");
            return;
        }
        Err(error) => panic!("GPU mask/subtraction hardware gate failed: {error}"),
    };
    let mask_path = std::env::temp_dir().join(format!(
        "astrocap-gpu-mask-sub-{}-{}.pgm",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::write(&mask_path, b"P5\n4 1\n255\n\xff\0\xff\xff").unwrap();
    let median_config: vyd::toml::Value =
        vyd::toml::from_str("async = true\nwindow_size = 0").unwrap();
    let mask_config: vyd::toml::Value = vyd::toml::from_str(&format!(
        "file_path = {:?}\ngpu = true",
        mask_path.to_string_lossy()
    ))
    .unwrap();
    let subtract_config: vyd::toml::Value =
        vyd::toml::from_str("subtractand_key = 'median'\ngpu = true").unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let stats = Arc::new(PipelineStatistics::new());
    let pipeline = Pipeline {
        source: Stage::new(Box::new(ValuesSource(0)), "values"),
        stages: vec![
            Stage::new(
                Box::new(MedianProcessor::new(Some(&median_config)).unwrap()),
                "median",
            ),
            Stage::new(
                Box::new(SpaceAdmissions(Duration::from_millis(50))),
                "space",
            ),
            Stage::new(
                Box::new(MaskProcessor::new(Some(&mask_config)).unwrap()),
                "mask",
            ),
            Stage::new(
                Box::new(ImgSubberProcessor::new(Some(&subtract_config)).unwrap()),
                "subtract",
            ),
        ],
        sink: Stage::new(Box::new(CheckSink(seen.clone())), "check"),
    };
    let mut ctx = PipelineContext::new();
    ctx.set_wgpu_context(gpu).unwrap();
    vyd_gstreamer::runtime::run_pipeline(ctx, pipeline, Some(stats.clone())).unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![(1, 5), (2, 4)]);
    let snapshot = stats.snapshot();
    assert_eq!(snapshot.stages["boundary/GPU upload"].call_count, 3);
    assert_eq!(snapshot.stages["boundary/GPU readback"].call_count, 3);
    let uploads: u64 = snapshot
        .frame_operations
        .values()
        .flat_map(|operations| operations.iter())
        .filter(|(name, _)| name.starts_with("upload_luma8@"))
        .map(|(_, counters)| counters.completed)
        .sum();
    assert_eq!(
        uploads, 6,
        "one mask, two medians, three source frames: {snapshot:?}"
    );
    let uploads_for = |stage: &str| -> u64 {
        snapshot.frame_operations[stage]
            .iter()
            .filter(|(name, _)| name.starts_with("upload_luma8@"))
            .map(|(_, counters)| counters.completed)
            .sum()
    };
    assert_eq!(uploads_for("mask"), 1);
    assert_eq!(uploads_for("subtract"), 2);
    assert_eq!(uploads_for("boundary/GPU upload"), 3);
    let downloads: u64 = snapshot.frame_operations["boundary/GPU readback"]
        .iter()
        .filter(|(name, _)| name.starts_with("download_luma8@"))
        .map(|(_, counters)| counters.completed)
        .sum();
    assert_eq!(downloads, 3);
    std::fs::remove_file(mask_path).unwrap();
}
