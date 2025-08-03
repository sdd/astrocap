mod config;
mod frame_buffer;
mod gst_pipeline;
mod ring_buffer_sink;

use std::sync::{
    atomic::{AtomicU8, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

use crate::frame_buffer::{BufferStateCallback, FrameBuffer, FrameInfo};
use astrocap_core::pipeline::PipelineContext;
use astrocap_core::{register_astrocap_frame_source, Frame, FrameContext, FrameSource};
use config::{Config, InputConfig};
use gst::prelude::*;
use gst_pipeline::{build_file_pipeline, build_rtsp_client_pipeline};

// const TARGET_FORMAT: VideoFormat = VideoFormat::Gray8;

#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
enum PlayState {
    Initializing = 0,
    Playing = 1,
    Stopped = 2,
}

impl From<u8> for PlayState {
    fn from(value: u8) -> Self {
        match value {
            0 => PlayState::Initializing,
            1 => PlayState::Playing,
            2 => PlayState::Stopped,
            _ => PlayState::Initializing, // Default fallback
        }
    }
}

pub struct GstSource {
    config: Config,
    ring_buffer: Arc<FrameBuffer>,
    pipeline: Arc<Mutex<Option<gst::Pipeline>>>,
    producer_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    play_state: Arc<AtomicU8>, // Store as u8 for atomic operations
}

impl GstSource {
    pub fn new(params: Option<&toml::Value>) -> Self {
        let Some(params) = params else {
            panic!("No parameters provided");
        };

        let config: Config = params.try_into().unwrap();

        // Create the ring buffer with appropriate mode
        let frame_info = FrameInfo {
            width: 1920,
            height: 1080,
            stride: 1920,
            timestamp: 0,
            frame_size: 1920 * 1080,
        };

        // Use drop_frames=false for non-live mode to enable back-pressure
        let drop_frames = config.input.is_live_mode();
        let ring_buffer = Arc::new(FrameBuffer::with_drop_mode(frame_info, 10, drop_frames));

        gst::init().unwrap();

        // Build the pipeline but don't start it yet
        let pipeline = match &config.input {
            InputConfig::File { path, .. } => build_file_pipeline(path, ring_buffer.clone()),
            InputConfig::Rtsp { uri } => build_rtsp_client_pipeline(uri, ring_buffer.clone()),
        }
        .expect("Failed to build pipeline");

        // Store pipeline in Arc so both the main struct and callback can access it
        let pipeline_for_callback = Arc::new(pipeline.clone());
        let pipeline_arc = Arc::new(Mutex::new(Some(pipeline)));

        // Set up pipeline state callback for non-live mode AFTER we have the pipeline
        if !drop_frames {
            let callback: BufferStateCallback = Arc::new(move |is_full| {
                tracing::info!("Pipeline callback triggered: is_full={}", is_full);

                let pipeline_clone = pipeline_for_callback.clone();

                // Spawn state change in a separate thread to avoid blocking
                thread::spawn(move || {
                    if is_full {
                        tracing::info!("About to pause pipeline due to full buffer");
                        match pipeline_clone.set_state(gst::State::Paused) {
                            Ok(gst::StateChangeSuccess::Success) => {
                                tracing::info!("Pipeline paused successfully (synchronous)");
                            }
                            Ok(gst::StateChangeSuccess::Async) => {
                                tracing::info!("Pipeline pause in progress (asynchronous)");
                                // For async state changes, wait for completion
                                let (result, current, pending) =
                                    pipeline_clone.state(Some(gst::ClockTime::from_seconds(1)));
                                match result {
                                    Ok(_) => {
                                        if current == gst::State::Paused {
                                            tracing::info!("Pipeline pause completed");
                                        } else {
                                            tracing::warn!("Pipeline pause incomplete, current state: {:?}, pending: {:?}", current, pending);
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "Error waiting for pipeline pause: {:?}",
                                            e
                                        );
                                    }
                                }
                            }
                            Ok(gst::StateChangeSuccess::NoPreroll) => {
                                tracing::info!("Pipeline paused (no preroll)");
                            }
                            Err(e) => {
                                tracing::error!("Failed to pause pipeline: {:?}", e);
                            }
                        }
                    } else {
                        tracing::info!("About to resume pipeline, buffer has space");
                        match pipeline_clone.set_state(gst::State::Playing) {
                            Ok(gst::StateChangeSuccess::Success) => {
                                tracing::info!("Pipeline resumed successfully (synchronous)");
                            }
                            Ok(gst::StateChangeSuccess::Async) => {
                                tracing::info!("Pipeline resume in progress (asynchronous)");
                                // For async state changes, wait for completion
                                let (result, current, pending) =
                                    pipeline_clone.state(Some(gst::ClockTime::from_seconds(1)));
                                match result {
                                    Ok(_) => {
                                        if current == gst::State::Playing {
                                            tracing::info!("Pipeline resume completed");
                                        } else {
                                            tracing::warn!("Pipeline resume incomplete, current state: {:?}, pending: {:?}", current, pending);
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            "Error waiting for pipeline resume: {:?}",
                                            e
                                        );
                                    }
                                }
                            }
                            Ok(gst::StateChangeSuccess::NoPreroll) => {
                                tracing::info!("Pipeline resumed (no preroll)");
                            }
                            Err(e) => {
                                tracing::error!("Failed to resume pipeline: {:?}", e);
                            }
                        }
                    }
                });
            });

            tracing::info!("Setting state callback on ring buffer");
            ring_buffer.set_state_callback(callback);
        }

        Self {
            config,
            ring_buffer,
            pipeline: pipeline_arc,
            producer_handle: Arc::new(Mutex::new(None)),
            play_state: Arc::new(AtomicU8::new(PlayState::Initializing as u8)),
        }
    }

    fn get_play_state(&self) -> PlayState {
        PlayState::from(self.play_state.load(Ordering::SeqCst))
    }

    fn set_play_state(&self, state: PlayState) {
        self.play_state.store(state as u8, Ordering::SeqCst);
    }

    fn start_producer_thread(&self) {
        if self
            .play_state
            .compare_exchange(
                PlayState::Initializing as u8,
                PlayState::Playing as u8,
                Ordering::SeqCst,
                Ordering::SeqCst,
            )
            .is_ok()
        {
            let pipeline = self
                .pipeline
                .lock()
                .unwrap()
                .take()
                .expect("Pipeline should be available");

            let play_state = self.play_state.clone();

            let handle = thread::spawn(move || {
                Self::run_pipeline(pipeline, play_state);
            });

            *self.producer_handle.lock().unwrap() = Some(handle);
        }
    }

    fn run_pipeline(pipeline: gst::Pipeline, play_state: Arc<AtomicU8>) {
        pipeline.set_state(gst::State::Playing).unwrap();

        let pipeline = Self::start_gst_playback_loop(pipeline);

        // Mark pipeline as stopped before cleanup
        play_state.store(PlayState::Stopped as u8, Ordering::SeqCst);

        // Clean shutdown
        let _ = pipeline.set_state(gst::State::Null);
    }

    fn start_gst_playback_loop(pipeline: gst::Pipeline) -> gst::Pipeline {
        tracing::trace!("Starting playback loop");

        let bus = pipeline.bus().unwrap();

        loop {
            let msg = bus.timed_pop(gst::ClockTime::from_mseconds(50));

            match msg {
                Some(msg) => {
                    use gst::MessageView;
                    match msg.view() {
                        MessageView::Eos(_) => {
                            tracing::info!("gst Eos msg received (end of stream)");
                            break;
                        }
                        MessageView::Error(err) => {
                            tracing::error!(
                                "Pipeline error: {} - {}",
                                err.error(),
                                err.debug().unwrap_or_default()
                            );
                            break;
                        }
                        MessageView::StateChanged(state_changed) => {
                            if state_changed.src().map(|s| s == &pipeline).unwrap_or(false) {
                                tracing::debug!(
                                    "Pipeline state changed from {:?} to {:?}",
                                    state_changed.old(),
                                    state_changed.current()
                                );
                            }
                        }
                        MessageView::StreamStart(_) => {
                            tracing::debug!("Stream started");
                        }

                        MessageView::AsyncDone(_) => {
                            tracing::trace!("gst async operation completed");
                        }

                        MessageView::Tag(tag) => {
                            tracing::trace!("Received tag: {:?}", tag.tags());
                        }

                        MessageView::StreamStatus(status) => {
                            tracing::trace!("Received stream status: {:?}", status);
                        }

                        MessageView::Latency(latency) => {
                            tracing::trace!("Received Latency msg: {:?}", latency);
                        }

                        MessageView::DurationChanged(duration) => {
                            tracing::trace!("Received DurationChanged msg: {:?}", duration);
                        }

                        MessageView::NewClock(new_clock) => {
                            tracing::trace!("Received NewClock msg: {:?}", new_clock);
                        }
                        _ => {
                            // Handle other messages if needed
                            tracing::trace!("Received other message: {:?}", msg.type_());
                        }
                    }
                }
                None => {
                    // Timeout polling for GST message from bus.
                    // check if the pipeline is still running
                    let state = pipeline.current_state();

                    // If pipeline stopped unexpectedly, break out
                    if state == gst::State::Null {
                        tracing::warn!("Pipeline went to NULL state unexpectedly");
                        break;
                    }
                }
            }
        }

        pipeline
    }
}

impl FrameSource for GstSource {
    fn next_frame(&mut self, ctx: &mut PipelineContext) -> Option<FrameContext> {
        // Start the producer thread if not already started
        self.start_producer_thread();

        // Handle different play states
        match self.get_play_state() {
            PlayState::Initializing | PlayState::Playing => {
                // Block with timeout to periodically check if pipeline has stopped
                loop {
                    if let Some(frame_data) = self
                        .ring_buffer
                        .read_frame_blocking(Duration::from_millis(250))
                    {
                        return self.convert_frame_data_with_counter(frame_data, ctx);
                    }
                    tracing::debug!("timeout or None when waiting for a frame");

                    // Check if pipeline stopped while we were waiting
                    if self.get_play_state() == PlayState::Stopped {
                        tracing::debug!("gst pipeline stopped whilst waiting");
                        // Try to get any remaining frames without blocking
                        if let Some(frame_data) = self.ring_buffer.try_read_frame() {
                            return self.convert_frame_data_with_counter(frame_data, ctx);
                        } else {
                            return None;
                        }
                    }
                }
            }
            PlayState::Stopped => {
                tracing::debug!("gst pipeline stopped");
                // Pipeline is done, try to get any remaining frames
                if let Some(frame_data) = self.ring_buffer.try_read_frame() {
                    self.convert_frame_data_with_counter(frame_data, ctx)
                } else {
                    None
                }
            }
        }
    }

    fn name(&self) -> &str {
        "gst_source"
    }
}

impl GstSource {
    fn convert_frame_data_with_counter(
        &self,
        frame_data: bytes::Bytes,
        ctx: &mut PipelineContext,
    ) -> Option<FrameContext> {
        // Track frames generated in pipeline context
        let counter = ctx
            .entry("frames_generated".to_string())
            .or_insert_with(|| Box::new(AtomicUsize::new(0)));

        // Convert bytes to ImageBuffer
        let width = 1920u32;
        let height = 1080u32;

        if let Some(img_buf) = image::ImageBuffer::from_raw(width, height, frame_data.to_vec()) {
            if let Some(atomic_counter) = counter.downcast_ref::<AtomicUsize>() {
                atomic_counter.fetch_add(1, Ordering::SeqCst);
            }

            tracing::trace!("Frame emitted");

            Some(FrameContext::new(Frame::ImgBuf(img_buf)))
        } else {
            tracing::warn!("Null frame emitted");
            None
        }
    }
}

register_astrocap_frame_source!(GstSource);
