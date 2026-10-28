use crate::median_config::MedianConfig;
use crate::median_filter::median_filter_into_with_cancel;
#[cfg(feature = "gpu-wgpu")]
use crate::median_gpu::GpuMedian;
use astrocap_core::frame_metadata::MedianFrame;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use image::{ImageBuffer, Luma};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use toml::Value;
use vyd::frame::pool::{PoolError, WaitPolicy};
use vyd::frame::FrameError;
use vyd::pipeline::PipelineContext;
use vyd::statistics::ProcessingType;
use vyd::statistics::{PipelineStatistics, StatsContext};

pub struct MedianProcessor {
    config: MedianConfig,
    worker: Option<AsyncWorker>,
}

// The worker owns at most one active input and one replaceable pending input.
// Results are monotonic by admission sequence; a current frame can only use a
// completed median from an earlier frame of the same dimensions.
#[derive(Default)]
struct WorkerState {
    pending: Option<(u64, Frame)>,
    latest: Option<(u64, Arc<Frame>)>,
    next_sequence: u64,
    active: bool,
    replaced: u64,
    completed: u64,
    failure: Option<String>,
    closing: bool,
}

impl WorkerState {
    fn admit(&mut self, frame: Frame) -> Result<Option<Arc<Frame>>, AstrocapError> {
        let sequence = self.next_sequence;
        self.next_sequence = sequence.checked_add(1).ok_or_else(|| {
            AstrocapError::GeneralPluginError("median admission sequence exhausted".into())
        })?;
        let previous = self
            .latest
            .as_ref()
            .and_then(|(completed, median)| (*completed < sequence).then(|| median.clone()));
        if self.pending.replace((sequence, frame)).is_some() {
            self.replaced += 1;
        }
        Ok(previous)
    }
}

struct WorkerShared {
    state: Mutex<WorkerState>,
    wake: Condvar,
}

struct AsyncWorker {
    shared: Arc<WorkerShared>,
    handle: Option<JoinHandle<()>>,
    stats: Option<Arc<PipelineStatistics>>,
    cancelled: Arc<AtomicBool>,
}

const MEDIAN_WORKER_TIMING: &str = "background/Median generation";

impl AsyncWorker {
    fn start(ctx: &PipelineContext, config: &MedianConfig) -> Result<Self, AstrocapError> {
        let shared = Arc::new(WorkerShared {
            state: Mutex::new(WorkerState::default()),
            wake: Condvar::new(),
        });
        let stats = ctx.services().latest::<PipelineStatistics>()?;
        if let Some(stats) = &stats {
            stats.register_stage(
                MEDIAN_WORKER_TIMING,
                if config.gpu {
                    ProcessingType::Gpu
                } else {
                    ProcessingType::Cpu
                },
            );
        }
        #[cfg(feature = "gpu-wgpu")]
        let gpu_median = if config.gpu {
            Some(GpuMedian::new(
                ctx.wgpu_context()
                    .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
                    .clone(),
                ctx.gpu_transfer()
                    .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?
                    .clone(),
                config.window_size,
            )?)
        } else {
            None
        };
        let window_size = config.window_size;
        let work_shared = shared.clone();
        let work_stats = stats.clone();
        let cancelled = Arc::new(AtomicBool::new(false));
        let work_cancelled = cancelled.clone();
        let factory = ctx.frame_factory().clone();
        let attribution = stats
            .as_ref()
            .map(|stats| StatsContext::new(stats.clone()).for_stage(MEDIAN_WORKER_TIMING));
        let handle = thread::Builder::new()
            .name("astrocap-median".into())
            .spawn(move || loop {
                let (sequence, input) = {
                    let mut state = work_shared
                        .state
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    while state.pending.is_none() && !state.closing {
                        state = work_shared
                            .wake
                            .wait(state)
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                    }
                    if state.closing || work_stats.as_ref().is_some_and(|s| !s.is_running()) {
                        return;
                    }
                    state.active = true;
                    work_shared.wake.notify_all();
                    state.pending.take().expect("pending checked above")
                };
                let _scope = attribution.as_ref().map(StatsContext::enter);
                let compute_started = Instant::now();
                let cancelled = || {
                    work_cancelled.load(Ordering::Acquire)
                        || work_stats.as_ref().is_some_and(|s| !s.is_running())
                };
                #[cfg(feature = "gpu-wgpu")]
                let result = if let Some(engine) = &gpu_median {
                    let mut input = input;
                    engine.filter(&mut input, &factory, cancelled)
                } else {
                    MedianProcessor::make_median(&input, &factory, window_size, cancelled)
                };
                #[cfg(not(feature = "gpu-wgpu"))]
                let result = MedianProcessor::make_median(&input, &factory, window_size, cancelled);
                let compute_elapsed = compute_started.elapsed();
                let mut state = work_shared
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                state.active = false;
                work_shared.wake.notify_all();
                if state.closing || work_stats.as_ref().is_some_and(|s| !s.is_running()) {
                    return;
                }
                match result {
                    Ok(median) => {
                        if let Some(stats) = &work_stats {
                            stats.record_stage_timing(
                                MEDIAN_WORKER_TIMING,
                                compute_elapsed.as_micros() as u64,
                            );
                        }
                        state.latest = Some((sequence, Arc::new(median)));
                        state.completed += 1;
                        work_shared.wake.notify_all();
                    }
                    Err(error) => {
                        state.failure = Some(error.to_string());
                        state.pending = None;
                        state.closing = true;
                        work_shared.wake.notify_all();
                        return;
                    }
                }
            })
            .map_err(|error| {
                AstrocapError::GeneralPluginError(format!("median worker spawn: {error}"))
            })?;
        Ok(Self {
            shared,
            handle: Some(handle),
            stats,
            cancelled,
        })
    }

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
    ) -> Result<FrameProcessorResult, AstrocapError> {
        frame_ctx
            .frame
            .try_read_cpu_luma8(WaitPolicy::Immediate)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(error) = &state.failure {
            return Err(AstrocapError::GeneralPluginError(format!(
                "median worker: {error}"
            )));
        }
        if state.closing
            || self.cancelled.load(Ordering::Acquire)
            || self.stats.as_ref().is_some_and(|s| !s.is_running())
        {
            return Err(AstrocapError::GeneralPluginError(
                "median worker is stopping".into(),
            ));
        }
        if self.handle.as_ref().is_some_and(JoinHandle::is_finished) {
            return Err(AstrocapError::GeneralPluginError(
                "median worker exited unexpectedly".into(),
            ));
        }
        // Snapshot before admission, so this frame cannot consume its own result.
        let previous = state
            .admit(frame_ctx.frame.clone())?
            .filter(|median| median.dimensions() == frame_ctx.frame.dimensions());
        self.shared.wake.notify_one();
        drop(state);
        if self.stats.as_ref().is_some_and(|s| !s.is_running()) {
            return Err(AstrocapError::GeneralPluginError(
                "median worker is stopping".into(),
            ));
        }
        if let Some(median) = previous {
            MedianProcessor::publish_median(frame_ctx, median);
        }
        Ok(FrameProcessorResult::Continue)
    }

    fn shutdown(&mut self) -> Result<(), AstrocapError> {
        {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // The worker must observe closing before it can observe cancellation.
            // Otherwise a cancelled calculation can be recorded as a failure.
            state.closing = true;
            self.cancelled.store(true, Ordering::Release);
            state.pending = None;
            state.latest = None;
            self.shared.wake.notify_one();
        }
        if self
            .handle
            .take()
            .is_some_and(|handle| handle.join().is_err())
        {
            return Err(AstrocapError::GeneralPluginError(
                "median worker panicked".into(),
            ));
        }
        let state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        tracing::debug!(
            admitted = state.next_sequence,
            completed = state.completed,
            replaced = state.replaced,
            "median worker stopped"
        );
        if let Some(error) = &state.failure {
            return Err(AstrocapError::GeneralPluginError(format!(
                "median worker: {error}"
            )));
        }
        Ok(())
    }
}

impl Drop for AsyncWorker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

impl MedianProcessor {
    fn publish_median(frame_ctx: &mut FrameContext, median: Arc<Frame>) {
        frame_ctx.put_typed(MedianFrame(median.clone()));
        // Keep the migration-era key for consumers not yet using typed metadata.
        frame_ctx.put("video/median", median);
    }

    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: MedianConfig = config.clone().try_into()?;
        if config.window_size > 32_767 {
            return Err(AstrocapError::PluginInvalidConfigError(
                "MedianProcessor: window_size must be at most 32767".into(),
            ));
        }
        if config.gpu && !config.r#async {
            return Err(AstrocapError::PluginInvalidConfigError(
                "MedianProcessor gpu = true requires async = true".into(),
            ));
        }
        #[cfg(not(feature = "gpu-wgpu"))]
        if config.gpu {
            return Err(AstrocapError::PluginInvalidConfigError(
                "MedianProcessor gpu = true requires the gpu-wgpu feature".into(),
            ));
        }
        if config.gpu && config.window_size > 64 {
            return Err(AstrocapError::PluginInvalidConfigError(
                "MedianProcessor GPU window_size must be at most 64".into(),
            ));
        }
        Ok(Self {
            config,
            worker: None,
        })
    }

    fn make_median(
        frame: &Frame,
        factory: &vyd::frame::factory::FrameFactory,
        window_size: u32,
        cancelled: impl Fn() -> bool,
    ) -> Result<Frame, AstrocapError> {
        let (width, height) = frame.dimensions();
        let pixels = frame
            .try_read_cpu_luma8(WaitPolicy::Immediate)
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))?;
        let img = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, pixels).ok_or_else(|| {
            AstrocapError::GeneralPluginError("invalid packed median input layout".into())
        })?;
        factory
            .create_luma8_with(width, height, WaitPolicy::Immediate, |bytes| {
                let len = bytes.len();
                let mut out = ImageBuffer::<Luma<u8>, _>::from_raw(width, height, bytes).ok_or(
                    FrameError::InvalidDimensions {
                        width,
                        height,
                        data_len: len,
                    },
                )?;
                if median_filter_into_with_cancel(
                    &img,
                    &mut out,
                    window_size,
                    window_size,
                    cancelled,
                ) {
                    Ok(())
                } else {
                    Err(FrameError::Pool(PoolError::Cancelled))
                }
            })
            .map_err(|error| AstrocapError::GeneralPluginError(error.to_string()))
    }

    fn process_sync(
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
        config: &MedianConfig,
    ) -> Result<FrameProcessorResult, AstrocapError> {
        let median = Self::make_median(
            &frame_ctx.frame,
            ctx.frame_factory(),
            config.window_size,
            || false,
        )?;
        Self::publish_median(frame_ctx, Arc::new(median));
        Ok(FrameProcessorResult::Continue)
    }
}

impl FrameProcessor for MedianProcessor {
    fn pipeline_ctx_init(&mut self, ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        if self.config.r#async {
            self.worker = Some(AsyncWorker::start(ctx, &self.config)?);
        }
        Ok(())
    }

    fn pipeline_finished(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        if let Some(mut worker) = self.worker.take() {
            worker.shutdown()?;
        }
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
                tracing::error!(%error, "median processing failed");
                FrameProcessorResult::Skip
            }
        }
    }

    fn try_process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> Result<FrameProcessorResult, AstrocapError> {
        if self.config.r#async {
            self.worker
                .as_mut()
                .ok_or_else(|| {
                    AstrocapError::GeneralPluginError("median worker was not initialized".into())
                })?
                .process(frame_ctx)
        } else {
            Self::process_sync(frame_ctx, ctx, &self.config)
        }
    }

    fn name(&self) -> &str {
        "median"
    }

    fn processing_type(&self) -> ProcessingType {
        ProcessingType::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::median_filter::median_filter;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    use vyd::frame::factory::FrameFactory;
    use vyd::frame::pool::{BufferPoolManager, PoolLimits};

    fn processor() -> MedianProcessor {
        MedianProcessor {
            config: MedianConfig {
                r#async: false,
                gpu: false,
                window_size: 1,
            },
            worker: None,
        }
    }

    fn async_processor(window_size: u32) -> MedianProcessor {
        MedianProcessor {
            config: MedianConfig {
                r#async: true,
                gpu: false,
                window_size,
            },
            worker: None,
        }
    }

    fn wait_for(worker: &AsyncWorker, predicate: impl Fn(&WorkerState) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut state = worker.shared.state.lock().unwrap();
        while !predicate(&state) {
            let left = deadline.saturating_duration_since(Instant::now());
            assert!(
                !left.is_zero(),
                "median worker did not reach expected state"
            );
            state = worker.shared.wake.wait_timeout(state, left).unwrap().0;
        }
    }

    #[test]
    fn async_configuration_constructs_before_pipeline_initialization() {
        let config: Value = toml::from_str("async = true\nwindow_size = 1").unwrap();
        let processor = MedianProcessor::new(Some(&config)).unwrap();
        assert!(processor.config.r#async);
    }

    #[test]
    fn gpu_median_requires_async_generation() {
        let config: Value = toml::from_str("gpu = true\nasync = false").unwrap();
        let error = MedianProcessor::new(Some(&config)).err().unwrap();
        assert!(error.to_string().contains("requires async = true"));
    }

    #[cfg(not(feature = "gpu-wgpu"))]
    #[test]
    fn gpu_median_requires_feature() {
        let config: Value = toml::from_str("gpu = true\nasync = true").unwrap();
        let error = MedianProcessor::new(Some(&config)).err().unwrap();
        assert!(error.to_string().contains("requires the gpu-wgpu feature"));
    }

    #[cfg(feature = "gpu-wgpu")]
    #[test]
    fn gpu_median_rejects_oversized_radius() {
        let config: Value = toml::from_str("gpu = true\nasync = true\nwindow_size = 65").unwrap();
        let error = MedianProcessor::new(Some(&config)).err().unwrap();
        assert!(error.to_string().contains("at most 64"));
    }

    #[test]
    fn pending_admission_is_bounded_and_keeps_newest_input() {
        let mut state = WorkerState::default();
        for value in [3, 5, 7] {
            assert!(state
                .admit(Frame::from_raw(1, 1, vec![value]).unwrap())
                .unwrap()
                .is_none());
        }
        assert_eq!(state.next_sequence, 3);
        assert_eq!(state.replaced, 2);
        let (sequence, frame) = state.pending.unwrap();
        assert_eq!(sequence, 2);
        assert_eq!(frame.as_cpu_ref().unwrap(), &[7]);
    }

    #[test]
    fn async_results_are_from_prior_frames_in_generation_order() {
        let mut processor = async_processor(0);
        let mut context = PipelineContext::new();
        processor.pipeline_ctx_init(&mut context).unwrap();
        let mut first = FrameContext::new(Frame::from_raw(1, 1, vec![3]).unwrap(), 0);
        processor.try_process(&mut first, &mut context).unwrap();
        assert!(first.get_typed::<MedianFrame>().is_none());
        wait_for(processor.worker.as_ref().unwrap(), |state| {
            state.completed == 1
        });
        let mut second = FrameContext::new(Frame::from_raw(1, 1, vec![8]).unwrap(), 1);
        processor.try_process(&mut second, &mut context).unwrap();
        assert_eq!(
            second
                .get_typed::<MedianFrame>()
                .unwrap()
                .0
                .as_cpu_ref()
                .unwrap(),
            &[3]
        );
        wait_for(processor.worker.as_ref().unwrap(), |state| {
            state.completed == 2
        });
        let mut third = FrameContext::new(Frame::from_raw(1, 1, vec![9]).unwrap(), 2);
        processor.try_process(&mut third, &mut context).unwrap();
        assert_eq!(
            third
                .get_typed::<MedianFrame>()
                .unwrap()
                .0
                .as_cpu_ref()
                .unwrap(),
            &[8]
        );
        processor.pipeline_finished(&mut context).unwrap();
        drop((first, second, third));
        assert!(context
            .frame_factory()
            .cpu_pool()
            .snapshot()
            .unwrap()
            .active
            .is_empty());
    }

    #[test]
    fn async_does_not_attach_a_median_from_different_dimensions() {
        let mut processor = async_processor(0);
        let mut context = PipelineContext::new();
        processor.pipeline_ctx_init(&mut context).unwrap();
        let mut first = FrameContext::new(Frame::from_raw(1, 1, vec![3]).unwrap(), 0);
        processor.try_process(&mut first, &mut context).unwrap();
        wait_for(processor.worker.as_ref().unwrap(), |state| {
            state.completed == 1
        });
        let mut second = FrameContext::new(Frame::from_raw(2, 1, vec![8, 8]).unwrap(), 1);
        processor.try_process(&mut second, &mut context).unwrap();
        assert!(second.get_typed::<MedianFrame>().is_none());
        processor.pipeline_finished(&mut context).unwrap();
    }

    #[test]
    fn worker_pool_failure_reaches_processing_and_finalization() {
        let pool = BufferPoolManager::new(PoolLimits::new(Some(1), Some(1)).unwrap());
        let mut context = PipelineContext::new();
        context.set_frame_factory(FrameFactory::new(pool.clone()));
        let mut processor = async_processor(0);
        processor.pipeline_ctx_init(&mut context).unwrap();
        let mut first = FrameContext::new(Frame::from_raw(1, 1, vec![3]).unwrap(), 0);
        processor.try_process(&mut first, &mut context).unwrap();
        wait_for(processor.worker.as_ref().unwrap(), |state| {
            state.completed == 1
        });
        let mut second = FrameContext::new(Frame::from_raw(1, 1, vec![8]).unwrap(), 1);
        processor.try_process(&mut second, &mut context).unwrap();
        drop((first, second));
        wait_for(processor.worker.as_ref().unwrap(), |state| {
            state.failure.is_some()
        });
        let mut third = FrameContext::new(Frame::from_raw(1, 1, vec![9]).unwrap(), 2);
        assert!(processor.try_process(&mut third, &mut context).is_err());
        assert!(third.get_typed::<MedianFrame>().is_none());
        assert!(processor.pipeline_finished(&mut context).is_err());
        assert!(pool.snapshot().unwrap().active.is_empty());
    }

    #[test]
    fn active_calculation_cancels_and_joins_without_publishing() {
        let mut processor = async_processor(30);
        let mut context = PipelineContext::new();
        processor.pipeline_ctx_init(&mut context).unwrap();
        let mut frame =
            FrameContext::new(Frame::from_raw(512, 512, vec![7; 512 * 512]).unwrap(), 0);
        processor.try_process(&mut frame, &mut context).unwrap();
        wait_for(processor.worker.as_ref().unwrap(), |state| state.active);
        let start = Instant::now();
        processor.pipeline_finished(&mut context).unwrap();
        assert!(start.elapsed() < Duration::from_secs(5));
        assert!(processor.worker.is_none());
        assert!(frame.get_typed::<MedianFrame>().is_none());
        assert!(context
            .frame_factory()
            .cpu_pool()
            .snapshot()
            .unwrap()
            .active
            .is_empty());
    }

    #[test]
    fn shutdown_does_not_signal_cancellation_before_closing() {
        let mut processor = async_processor(0);
        let mut context = PipelineContext::new();
        processor.pipeline_ctx_init(&mut context).unwrap();
        let mut worker = processor.worker.take().unwrap();
        let shared = worker.shared.clone();
        let cancelled = worker.cancelled.clone();

        // Hold the state lock while shutdown starts. Cancellation must remain
        // invisible until the shutdown thread can mark the state as closing.
        let guard = shared.state.lock().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let shutdown = thread::spawn(move || {
            started_tx.send(()).unwrap();
            worker.shutdown()
        });
        started_rx.recv().unwrap();
        thread::sleep(Duration::from_millis(20));
        assert!(!cancelled.load(Ordering::Acquire));
        drop(guard);

        shutdown.join().unwrap().unwrap();
        let state = shared.state.lock().unwrap();
        assert!(state.closing);
        assert!(state.failure.is_none());
    }

    #[test]
    fn external_pipeline_stop_cancels_active_calculation() {
        let stats = Arc::new(PipelineStatistics::new());
        let mut context = PipelineContext::new();
        context.services().publish(stats.clone()).unwrap();
        let mut processor = async_processor(30);
        processor.pipeline_ctx_init(&mut context).unwrap();
        let mut frame =
            FrameContext::new(Frame::from_raw(512, 512, vec![7; 512 * 512]).unwrap(), 0);
        processor.try_process(&mut frame, &mut context).unwrap();
        wait_for(processor.worker.as_ref().unwrap(), |state| state.active);
        stats.stop();
        let start = Instant::now();
        processor.pipeline_finished(&mut context).unwrap();
        assert!(start.elapsed() < Duration::from_secs(5));
        assert!(frame.get_typed::<MedianFrame>().is_none());
        assert!(context
            .frame_factory()
            .cpu_pool()
            .snapshot()
            .unwrap()
            .active
            .is_empty());
    }

    #[test]
    fn dropping_processor_also_joins_worker() {
        let mut processor = async_processor(30);
        let mut context = PipelineContext::new();
        processor.pipeline_ctx_init(&mut context).unwrap();
        let mut frame =
            FrameContext::new(Frame::from_raw(512, 512, vec![7; 512 * 512]).unwrap(), 0);
        processor.try_process(&mut frame, &mut context).unwrap();
        wait_for(processor.worker.as_ref().unwrap(), |state| state.active);
        drop(processor);
        assert!(context
            .frame_factory()
            .cpu_pool()
            .snapshot()
            .unwrap()
            .active
            .is_empty());
    }

    #[test]
    fn omitted_async_uses_synchronous_default() {
        let config: Value = toml::from_str("window_size = 1").unwrap();
        let mut processor = MedianProcessor::new(Some(&config)).unwrap();
        let mut frame = FrameContext::new(Frame::from_raw(1, 1, vec![7]).unwrap(), 0);
        let mut context = PipelineContext::new();
        processor.pipeline_ctx_init(&mut context).unwrap();
        assert!(matches!(
            processor.try_process(&mut frame, &mut context),
            Ok(FrameProcessorResult::Continue)
        ));
        assert_eq!(
            frame
                .get_typed::<MedianFrame>()
                .unwrap()
                .0
                .as_cpu_ref()
                .unwrap(),
            &[7]
        );
        processor.pipeline_finished(&mut context).unwrap();
    }

    #[test]
    fn synchronous_median_matches_existing_filter_into_pooled_output() {
        let input = Frame::from_raw(3, 3, vec![4, 9, 2, 8, 1, 7, 3, 6, 5]).unwrap();
        let expected = median_filter(input.as_cpu_image().unwrap(), 1, 1);
        let mut frame = FrameContext::new(input, 0);
        let mut context = PipelineContext::new();
        assert!(matches!(
            processor().try_process(&mut frame, &mut context),
            Ok(FrameProcessorResult::Continue)
        ));
        let median = frame
            .metadata
            .get("video/median")
            .unwrap()
            .downcast_ref::<Arc<Frame>>()
            .unwrap();
        assert!(Arc::ptr_eq(
            median,
            &frame.get_typed::<MedianFrame>().unwrap().0
        ));
        assert_eq!(median.as_cpu_ref().unwrap(), expected.as_raw());
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
        assert!(context
            .frame_factory()
            .cpu_pool()
            .snapshot()
            .unwrap()
            .active
            .is_empty());
    }

    #[test]
    fn pool_exhaustion_does_not_publish_median() {
        let pool = BufferPoolManager::new(PoolLimits::new(Some(1), Some(9)).unwrap());
        let factory = FrameFactory::new(pool.clone());
        let held = factory.create_luma8(3, 3, WaitPolicy::Immediate).unwrap();
        let mut context = PipelineContext::new();
        context.set_frame_factory(factory);
        let mut frame = FrameContext::new(
            Frame::from_raw(3, 3, vec![4, 9, 2, 8, 1, 7, 3, 6, 5]).unwrap(),
            0,
        );
        assert!(processor().try_process(&mut frame, &mut context).is_err());
        assert!(frame.metadata.get("video/median").is_none());
        assert!(frame.get_typed::<MedianFrame>().is_none());
        assert_eq!(
            frame.frame.as_cpu_ref().unwrap(),
            &[4, 9, 2, 8, 1, 7, 3, 6, 5]
        );
        drop(held);
        assert!(pool.snapshot().unwrap().active.is_empty());
    }
}
