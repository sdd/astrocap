use crate::median_config::MedianConfig;
use crate::median_filter::median_filter;
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::traits::FrameProcessor;
use astrocap_core::{AstrocapError, Frame, FrameContext, FrameProcessorResult};
use image::GrayImage;
use std::sync::{Arc, Mutex};
use toml::Value;

pub struct MedianProcessor {
    config: MedianConfig,
    async_state: Option<AsyncState>,
}

struct AsyncState {
    frame: Arc<Mutex<Arc<Frame>>>,
    median: Arc<Mutex<Arc<Frame>>>,
}

impl MedianProcessor {
    pub fn new(config: Option<&Value>) -> Result<Self, AstrocapError> {
        let Some(config) = config else {
            return Err(AstrocapError::PluginMissingConfigError);
        };

        let config: MedianConfig = config.clone().try_into().map_err(|e| {
            AstrocapError::PluginInvalidConfigError(format!("MedianProcessor:: {}", e.to_string()))
        })?;

        Ok(Self {
            async_state: None,
            config,
        })
    }

    fn process_sync(
        frame_ctx: &mut FrameContext,
        _ctx: &mut PipelineContext,
        config: &MedianConfig,
    ) -> FrameProcessorResult {
        if let Ok(img) = frame_ctx.frame.get_image(None) {
            let img_median: GrayImage = median_filter(img, config.window_size, config.window_size);

            let frame = Frame::from(img_median);

            frame_ctx.put("video/median", Arc::new(frame));
        }

        FrameProcessorResult::Continue
    }

    fn process_async(
        frame_ctx: &mut FrameContext,
        _ctx: &mut PipelineContext,
        async_state: &AsyncState,
    ) -> FrameProcessorResult {
        let frame = frame_ctx.take_frame();
        let Some(cpu_frame) = frame.to_cpu_frame() else {
            tracing::error!("No frame");
            return FrameProcessorResult::Continue;
        };

        let frame = Frame::Cpu(cpu_frame.to_shared());
        let cloned_frame = frame.clone();
        {
            *async_state.frame.lock().unwrap() = Arc::new(cloned_frame);
        }

        let median = { async_state.median.lock().unwrap().clone() };

        frame_ctx.put("video/median", median);

        frame_ctx.frame = frame;

        FrameProcessorResult::Continue
    }
}

impl FrameProcessor for MedianProcessor {
    fn pipeline_ctx_init(&mut self, _ctx: &mut PipelineContext) -> Result<(), AstrocapError> {
        let window_size = self.config.window_size;

        let async_state = if self.config.r#async {
            let frame = Arc::new(Mutex::new(Arc::new(Frame::None)));
            let median = Arc::new(Mutex::new(Arc::new(Frame::None)));

            let frame_clone = frame.clone();
            let median_clone = median.clone();

            std::thread::spawn(move || loop {
                let frame = { frame_clone.lock().unwrap().clone() };

                if let Some(img) = frame.as_cpu_image() {
                    let img_median: GrayImage = median_filter(img, window_size, window_size);

                    let median = Arc::new(Frame::shared_from_img(img_median));

                    {
                        *median_clone.lock().unwrap() = median;
                    }
                } else {
                    tracing::error!("No frame");
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
            });

            Some(AsyncState { frame, median })
        } else {
            None
        };

        self.async_state = async_state;

        Ok(())
    }

    fn process(
        &mut self,
        frame_ctx: &mut FrameContext,
        ctx: &mut PipelineContext,
    ) -> FrameProcessorResult {
        if self.config.r#async {
            Self::process_async(frame_ctx, ctx, self.async_state.as_ref().unwrap())
        } else {
            Self::process_sync(frame_ctx, ctx, &self.config)
        }
    }

    fn name(&self) -> &str {
        "median"
    }
}
