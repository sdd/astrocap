use crate::{Error, FrameProcessor, FrameSink, FrameSource, StageFactory};
use dashmap::DashMap;
use serde::Deserialize;
use std::any::Any;
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

// Delegate trait methods to inner implementations
impl FrameSource for FrameSourceWrapper {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<crate::FrameContext> {
        self.inner.next_frame(ctx)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

impl FrameProcessor for FrameProcessorWrapper {
    fn process(&mut self, frame_ctx: &mut crate::FrameContext, ctx: &mut PipelineContext) -> bool {
        self.inner.process(frame_ctx, ctx)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

impl FrameSink for FrameSinkWrapper {
    fn consume(&mut self, frame_ctx: &crate::FrameContext, ctx: &mut PipelineContext) {
        self.inner.consume(frame_ctx, ctx)
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

    while let Some(mut ctx) = source.next_frame(&mut pipeline_context) {
        let mut continue_processing = true;
        for stage in stages.iter_mut() {
            if !stage.process(&mut ctx, &mut pipeline_context) {
                continue_processing = false;
                break;
            }
        }

        if continue_processing {
            sink.consume(&ctx, &mut pipeline_context);
        }
    }

    pipeline_context
}
