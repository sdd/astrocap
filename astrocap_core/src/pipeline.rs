use crate::{Error, FrameProcessor, FrameSink, FrameSource, StageFactory};
use dashmap::DashMap;
use serde::Deserialize;
use std::any::Any;
use std::collections::HashMap;
use std::time::Instant;
use toml::Value;

pub type PipelineContextValue = Box<dyn Any + Send + Sync>;
pub type PipelineContext = DashMap<String, PipelineContextValue>;

#[derive(Deserialize, Debug)]
pub struct PipelineConfig {
    pub source: StageConfig,
    pub stages: Vec<StageConfig>,
    pub sink: StageConfig,
}

#[derive(Deserialize, Debug)]
pub struct StageConfig {
    pub name: Option<String>,
    pub stage_type: String,

    #[serde(flatten)]
    pub params: Option<toml::Value>,
}

// Wrapper types that add stage_type functionality
struct FrameSourceWrapper {
    inner: Box<dyn FrameSource>,
    stage_type: &'static str,
}

struct FrameProcessorWrapper {
    inner: Box<dyn FrameProcessor>,
    stage_type: &'static str,
}

struct FrameSinkWrapper {
    inner: Box<dyn FrameSink>,
    stage_type: &'static str,
}

pub struct Pipeline {
    source: FrameSourceWrapper,
    stages: Vec<FrameProcessorWrapper>,
    sink: FrameSinkWrapper,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("source", &self.source.stage_type)
            .field(
                "stages",
                &self.stages.iter().map(|s| s.stage_type).collect::<Vec<_>>(),
            )
            .field("sink", &self.sink.stage_type)
            .finish()
    }
}

/// Helper function to get or create timing data from FrameContext
fn get_or_create_timing_data(frame_ctx: &mut crate::FrameContext) -> &mut Vec<(u64, String)> {
    // Check if timing_data already exists
    if !frame_ctx.metadata.contains_key("timing_data") {
        frame_ctx.metadata.insert(
            "timing_data".to_string(),
            Box::new(Vec::<(u64, String)>::new()),
        );
    }

    // Get mutable reference to timing data
    frame_ctx
        .metadata
        .get_mut("timing_data")
        .unwrap()
        .downcast_mut::<Vec<(u64, String)>>()
        .expect("timing_data should be Vec<(u64, String)>")
}

/// Helper function to add astrocap stage timing to existing timing data (in microseconds)
fn add_stage_timing(
    frame_ctx: &mut crate::FrameContext,
    stage_name: &str,
    stage_type: &str,
    duration_us: u64,
) {
    let timing_data = get_or_create_timing_data(frame_ctx);

    // Get current timestamp (we'll use the time since we got the frame for relative timing)
    // This is approximate but gives us relative timing within the astrocap pipeline
    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64;

    // Add astrocap-level timing with a prefix to distinguish from GST timing
    let event_name = format!("astrocap_{}", stage_name);
    timing_data.push((timestamp_us, event_name));

    tracing::trace!(
        stage_name,
        stage_type,
        duration_us,
        timestamp_us,
        total_timing_events = timing_data.len(),
        "astrocap_pipeline.stage_latency_us"
    );
}

/// Helper function to log timing summary exactly like GST's print_timing_summary
fn log_frame_timing_summary(
    frame_ctx: &crate::FrameContext,
    frame_count: u64,
    _total_frame_duration_us: u64,
) {
    if let Some(timing_data) = frame_ctx
        .metadata
        .get("timing_data")
        .and_then(|data| data.downcast_ref::<Vec<(u64, String)>>())
    {
        if timing_data.is_empty() {
            tracing::info!(frame_count, "No timing data found on frame");
            return;
        }

        tracing::info!(frame_count, "=== Frame Processing Timeline ===");

        // Sort all events by timestamp (chronological order)
        let mut all_events = timing_data.clone();
        all_events.sort_by_key(|(timestamp, _)| *timestamp);

        // Get the start time from the first event
        let start_time = all_events[0].0;
        let mut prev_time = start_time;

        for (timestamp_us, event_name) in all_events {
            let elapsed_us = timestamp_us.saturating_sub(start_time);
            let delta_us = timestamp_us.saturating_sub(prev_time);
            tracing::info!(
                frame_count,
                "{}: Δ{} μs (+{} μs)",
                event_name,
                delta_us,
                elapsed_us
            );
            prev_time = timestamp_us;
        }

        tracing::info!(frame_count, "===================================");
    }
}

#[allow(unused)]
impl FrameSourceWrapper {
    pub fn new(inner: Box<dyn FrameSource>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn stage_type(&self) -> &'static str {
        self.stage_type
    }
}

#[allow(unused)]
impl FrameProcessorWrapper {
    pub fn new(inner: Box<dyn FrameProcessor>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn stage_type(&self) -> &'static str {
        self.stage_type
    }
}

#[allow(unused)]
impl FrameSinkWrapper {
    pub fn new(inner: Box<dyn FrameSink>, stage_type: &'static str) -> Self {
        Self { inner, stage_type }
    }

    pub fn name(&self) -> &str {
        self.inner.name()
    }

    pub fn stage_type(&self) -> &'static str {
        self.stage_type
    }
}

// Delegate trait methods to inner implementations with timing integration (in microseconds)
impl FrameSource for FrameSourceWrapper {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<crate::FrameContext> {
        let start_time = Instant::now();
        let mut result = self.inner.next_frame(ctx);
        let duration = start_time.elapsed();
        let duration_us = duration.as_micros() as u64;

        // Add timing to the frame context if we got a frame
        if let Some(ref mut frame_ctx) = result {
            add_stage_timing(frame_ctx, self.name(), self.stage_type, duration_us);

            // Log that we found existing GST timing data
            if let Some(timing_data) = frame_ctx
                .metadata
                .get("timing_data")
                .and_then(|data| data.downcast_ref::<Vec<(u64, String)>>())
            {
                let gst_event_count = timing_data
                    .iter()
                    .filter(|(_, name)| !name.starts_with("astrocap_"))
                    .count();
                if gst_event_count > 0 {
                    tracing::trace!(
                        gst_event_count,
                        "Frame source retrieved frame with GST timing data"
                    );
                }
            }
        }

        result
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

impl FrameProcessor for FrameProcessorWrapper {
    fn process(&mut self, frame_ctx: &mut crate::FrameContext, ctx: &mut PipelineContext) -> bool {
        let start_time = Instant::now();
        let result = self.inner.process(frame_ctx, ctx);
        let duration = start_time.elapsed();
        let duration_us = duration.as_micros() as u64;

        add_stage_timing(frame_ctx, self.name(), self.stage_type, duration_us);

        result
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

impl FrameSink for FrameSinkWrapper {
    fn consume(&mut self, frame_ctx: &mut crate::FrameContext, ctx: &mut PipelineContext) {
        let start_time = Instant::now();
        self.inner.consume(frame_ctx, ctx);
        let duration = start_time.elapsed();
        let duration_us = duration.as_micros() as u64;

        add_stage_timing(frame_ctx, self.name(), self.stage_type, duration_us);
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

// Collect references to StageFactory trait objects
inventory::collect!(&'static dyn StageFactory);

#[macro_export]
macro_rules! _register_astrocap_stage_impl {
    ($crate_name:expr, $stage_type:ty, $trait_name:ident) => {
        $crate::paste::paste! {
            struct [<$stage_type Factory>];

            impl $crate::StageFactory for [<$stage_type Factory>] {
                fn create(&self, params: Option<&toml::Value>) -> Result<Box<dyn std::any::Any>, $crate::Error> {
                    let instance = <$stage_type>::new(params)?;
                    Ok(Box::new(Box::new(instance) as Box<dyn $crate::$trait_name>) as Box<dyn std::any::Any>)
                }

                fn stage_type(&self) -> &'static str {
                    $crate::paste::paste! {
                        concat!($crate_name, "::", stringify!($stage_type))
                    }
                }
            }

            static [<$crate_name:snake:upper __ $stage_type:snake:upper _FACTORY>]: [<$stage_type Factory>] = [<$stage_type Factory>];

            $crate::inventory::submit! {
                &[<$crate_name:snake:upper __ $stage_type:snake:upper _FACTORY>] as &'static dyn $crate::StageFactory
            }
        }
    };
}

#[macro_export]
macro_rules! register_astrocap_frame_source {
    ($source_type:ty) => {
        $crate::_register_astrocap_stage_impl!(env!("CARGO_PKG_NAME"), $source_type, FrameSource);
    };
}

#[macro_export]
macro_rules! register_astrocap_frame_processor {
    ($processor_type:ty) => {
        $crate::_register_astrocap_stage_impl!(
            env!("CARGO_PKG_NAME"),
            $processor_type,
            FrameProcessor
        );
    };
}

#[macro_export]
macro_rules! register_astrocap_frame_sink {
    ($sink_type:ty) => {
        $crate::_register_astrocap_stage_impl!(env!("CARGO_PKG_NAME"), $sink_type, FrameSink);
    };
}

pub fn build_pipeline(config: &PipelineConfig) -> Result<Pipeline, Error> {
    tracing::trace!(?config, "Building pipeline");

    // Create specific factory functions for each trait type
    fn find_source_factory(
        type_name: &str,
        params: Option<&Value>,
    ) -> Result<FrameSourceWrapper, Error> {
        for factory in inventory::iter::<&dyn StageFactory>() {
            tracing::trace!("checking factory: {:?}", factory.stage_type());
            if factory.stage_type() == type_name {
                tracing::trace!(
                    "Found factory for source type: {}. Sending params {:?}",
                    type_name,
                    &params
                );
                let any_box = factory.create(params)?;

                // Try to downcast to Box<dyn FrameSource>
                let inner = any_box.downcast::<Box<dyn FrameSource>>()
                    .map(|boxed| *boxed)
                    .unwrap_or_else(|_any_box| {
                        panic!("Factory for '{}' returned incompatible type. Expected Box<dyn FrameSource> but got something else.", type_name);
                    });

                return Ok(FrameSourceWrapper::new(inner, factory.stage_type()));
            }
        }
        panic!("Unknown source type: {}", type_name)
    }

    fn find_processor_factory(
        type_name: &str,
        params: Option<&toml::Value>,
    ) -> Result<FrameProcessorWrapper, Error> {
        for factory in inventory::iter::<&dyn StageFactory>() {
            if factory.stage_type() == type_name {
                let any_box = factory.create(params)?;

                let inner = any_box.downcast::<Box<dyn FrameProcessor>>()
                    .map(|boxed| *boxed)
                    .unwrap_or_else(|_| {
                        panic!("Factory for '{}' returned incompatible type. Expected Box<dyn FrameProcessor> but got something else.", type_name);
                    });

                return Ok(FrameProcessorWrapper::new(inner, factory.stage_type()));
            }
        }
        panic!("Unknown processor type: {}", type_name)
    }

    fn find_sink_factory(
        type_name: &str,
        params: Option<&toml::Value>,
    ) -> Result<FrameSinkWrapper, Error> {
        for factory in inventory::iter::<&dyn StageFactory>() {
            if factory.stage_type() == type_name {
                let any_box = factory.create(params)?;

                let inner = any_box.downcast::<Box<dyn FrameSink>>()
                    .map(|boxed| *boxed)
                    .unwrap_or_else(|_| {
                        panic!("Factory for '{}' returned incompatible type. Expected Box<dyn FrameSink> but got something else.", type_name);
                    });

                return Ok(FrameSinkWrapper::new(inner, factory.stage_type()));
            }
        }
        panic!("Unknown sink type: {}", type_name)
    }

    tracing::trace!(params = ?config.source.params.as_ref(),
        stage_type = ?&config.source.stage_type,
        "trying to create source",
    );
    let source = find_source_factory(&config.source.stage_type, config.source.params.as_ref())?;
    let sink = find_sink_factory(&config.sink.stage_type, config.sink.params.as_ref())?;

    let mut stages: Vec<FrameProcessorWrapper> = Vec::new();
    for stage in &config.stages {
        stages.push(find_processor_factory(
            &stage.stage_type,
            stage.params.as_ref(),
        )?);
    }

    Ok(Pipeline {
        source,
        stages,
        sink,
    })
}

pub fn run_pipeline(mut pipeline_context: PipelineContext, pipeline: Pipeline) -> PipelineContext {
    let Pipeline {
        mut source,
        mut stages,
        mut sink,
    } = pipeline;

    let pipeline_start = Instant::now();
    let mut frame_count = 0u64;

    while let Some(mut ctx) = source.next_frame(&mut pipeline_context) {
        let frame_start = Instant::now();
        frame_count += 1;

        let mut continue_processing = true;
        for stage in stages.iter_mut() {
            if !stage.process(&mut ctx, &mut pipeline_context) {
                continue_processing = false;
                break;
            }
        }

        if continue_processing {
            sink.consume(&mut ctx, &mut pipeline_context);
        }

        let total_frame_duration = frame_start.elapsed();
        let total_frame_duration_us = total_frame_duration.as_micros() as u64;

        // Log comprehensive timing summary for this frame
        log_frame_timing_summary(&ctx, frame_count, total_frame_duration_us);

        // Log periodic throughput stats
        if frame_count % 100 == 0 {
            let elapsed = pipeline_start.elapsed();
            let fps = frame_count as f64 / elapsed.as_secs_f64();

            tracing::info!(
                frames_processed = frame_count,
                elapsed_seconds = elapsed.as_secs_f64(),
                fps,
                "Pipeline throughput"
            );

            // If you have a metrics library available
            // metrics::gauge!("astrocap_pipeline.fps").set(fps);
        }
    }

    let total_duration = pipeline_start.elapsed();
    let fps = frame_count as f64 / total_duration.as_secs_f64();

    tracing::info!(
        total_frames = frame_count,
        total_duration_seconds = total_duration.as_secs_f64(),
        average_fps = fps,
        "Pipeline completed"
    );

    pipeline_context
}
