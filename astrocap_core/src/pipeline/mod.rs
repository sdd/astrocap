use std::any::Any;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use serde::Deserialize;
use toml::Value;

pub(crate) mod timing;
pub(crate) mod wrappers;

use crate::pipeline::timing::log_frame_timing_summary;
use crate::pipeline::wrappers::{FrameProcessorWrapper, FrameSinkWrapper, FrameSourceWrapper};
use crate::statistics::PipelineStatistics;
use crate::traits::{FrameProcessor, FrameSink, FrameSource, StageFactory};
use crate::{AstrocapError, FrameProcessorResult};

pub type PipelineContextValue = Box<dyn Any + Send + Sync>;

#[derive(Debug)]
pub struct PipelineContext {
    metadata: HashMap<String, PipelineContextValue>,
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineContext {
    pub fn new() -> Self {
        Self {
            metadata: HashMap::new(),
        }
    }

    pub fn entry<'a, K: Into<String>>(
        &'a mut self,
        key: K,
    ) -> Entry<'a, String, PipelineContextValue> {
        self.metadata.entry(key.into())
    }

    pub fn get_as<'a, T: 'static>(&'a self, key: &str) -> Result<&'a T, AstrocapError> {
        let Some(value) = self.metadata.get(key) else {
            tracing::error!("key \"{}\" not present in frame context metadata", key);
            return Err(AstrocapError::FrameMetadataNotFoundError);
        };

        let Some(downcasted) = value.downcast_ref::<T>() else {
            tracing::error!("Could not downcast metadata value to requested type");
            return Err(AstrocapError::FrameMetadataTypeError);
        };

        Ok(downcasted)
    }

    pub fn put<T: Any + Send + Sync + 'static>(&mut self, key: &str, value: T) {
        self.metadata.insert(key.to_string(), Box::new(value));
    }
}

#[derive(Deserialize, Debug)]
pub struct PipelineConfig {
    pub source: StageConfig,
    #[serde(default)]
    pub stages: Vec<StageConfig>,
    pub sink: StageConfig,
}

#[derive(Deserialize, Debug)]
struct StageConfig {
    pub name: Option<String>,
    pub stage_type: String,

    #[serde(flatten)]
    pub params: Option<Value>,
}

pub struct Pipeline {
    pub source: FrameSourceWrapper,
    pub stages: Vec<FrameProcessorWrapper>,
    pub sink: FrameSinkWrapper,
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

inventory::collect!(&'static dyn StageFactory);

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

#[macro_export]
macro_rules! _register_astrocap_stage_impl {
    ($crate_name:expr, $stage_type:ty, $trait_name:ident) => {
        $crate::paste::paste! {
            struct [<$stage_type Factory>];

            impl $crate::StageFactory for [<$stage_type Factory>] {
                fn create(&self, params: Option<&toml::Value>) -> Result<Box<dyn std::any::Any>, $crate::AstrocapError> {
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

pub fn build_pipeline(config: &PipelineConfig) -> Result<Pipeline, AstrocapError> {
    tracing::trace!(?config, "Building pipeline");

    // Create specific factory functions for each trait type
    fn find_source_factory(
        type_name: &str,
        params: Option<&Value>,
    ) -> Result<FrameSourceWrapper, AstrocapError> {
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
                        panic!("Factory for '{}' returned an incompatible type. Expected Box<dyn FrameSource> but got something else.", type_name);
                    });

                return Ok(FrameSourceWrapper::new(inner, factory.stage_type()));
            }
        }
        panic!("Unknown source type: {}", type_name)
    }

    fn find_processor_factory(
        type_name: &str,
        params: Option<&toml::Value>,
    ) -> Result<FrameProcessorWrapper, AstrocapError> {
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
    ) -> Result<FrameSinkWrapper, AstrocapError> {
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

pub fn run_pipeline(
    mut pipeline_context: PipelineContext,
    pipeline: Pipeline,
    stats: Option<Arc<PipelineStatistics>>,
) -> PipelineContext {
    let Pipeline {
        mut source,
        mut stages,
        mut sink,
    } = pipeline;

    // Add statistics to pipeline context if provided
    if let Some(ref stats) = stats {
        pipeline_context.put("pipeline_statistics", stats.clone());
    }

    tracing::info!("Initializing pipeline context");
    source.pipeline_ctx_init(&mut pipeline_context).unwrap();
    for stage in stages.iter_mut() {
        stage.pipeline_ctx_init(&mut pipeline_context).unwrap();
    }
    sink.pipeline_ctx_init(&mut pipeline_context).unwrap();
    tracing::info!(?pipeline_context, "pipeline context initialized");

    let pipeline_start = Instant::now();
    let mut frame_count = 0u64;

    while let Some(mut ctx) = source.next_frame(&mut pipeline_context) {
        // Check if we should stop (for Ctrl+C handling)
        if let Some(ref stats) = stats {
            if !stats.is_running() {
                tracing::info!("Received shutdown signal, stopping pipeline gracefully");
                break;
            }
        }

        // Process any GStreamer timing data that came with this frame
        crate::pipeline::timing::process_gst_timing_data(&ctx, stats.as_deref());

        let frame_start = Instant::now();
        frame_count += 1;

        let mut continue_processing = true;
        for stage in stages.iter_mut() {
            if matches!(
                stage.process(&mut ctx, &mut pipeline_context),
                FrameProcessorResult::Skip
            ) {
                continue_processing = false;
                break;
            }
        }

        if continue_processing {
            sink.consume(&mut ctx, &mut pipeline_context);
        }

        let total_frame_duration = frame_start.elapsed();
        let total_frame_duration_us = total_frame_duration.as_micros() as u64;

        // Record statistics if available
        if let Some(ref stats) = stats {
            stats.record_frame(total_frame_duration_us);
        }

        // Log comprehensive timing summary for this frame
        log_frame_timing_summary(&ctx, frame_count, total_frame_duration_us);

        // Log periodic throughput stats
        if frame_count.is_multiple_of(100) {
            let elapsed = pipeline_start.elapsed();
            let fps = frame_count as f64 / elapsed.as_secs_f64();

            tracing::info!(
                frames_processed = frame_count,
                elapsed_seconds = elapsed.as_secs_f64(),
                fps,
                "Pipeline throughput"
            );
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

pub fn run_pipeline_with_config_file_path(
    path: &str,
    stats: Arc<PipelineStatistics>,
) -> PipelineContext {
    let config_raw = std::fs::read_to_string(path).expect("Failed to read config file");
    let config: PipelineConfig = toml::from_str(&config_raw).expect("Failed to parse config file");
    let pipeline = build_pipeline(&config).expect("Failed to build pipeline from config");
    let pipeline_context = PipelineContext::new();

    run_pipeline(pipeline_context, pipeline, Some(stats))
}
