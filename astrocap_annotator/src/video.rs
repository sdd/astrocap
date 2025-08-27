use anyhow::Result;
use gst::prelude::*;
use gst::{Element, MessageView, Pipeline, State};
use gst_app::AppSink;
use gst_video::VideoInfo;
use std::path::Path;
use std::sync::mpsc;

pub struct VideoFrameCache {
    frames: Vec<egui::ColorImage>,
    width: u32,
    height: u32,
    fps: f64,
}

impl VideoFrameCache {
    pub fn new<P: AsRef<Path>>(video_path: P) -> Result<Self> {
        Self::load_from_file(video_path)
    }

    pub fn load_from_file<P: AsRef<Path>>(video_path: P) -> Result<Self> {
        gst::init()?;

        let pipeline_str = format!(
            "filesrc location={} ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink",
            video_path.as_ref().to_string_lossy()
        );

        let pipeline = gst::parse_launch(&pipeline_str)?;
        let pipeline = pipeline.dynamic_cast::<Pipeline>().unwrap();

        let sink = pipeline
            .by_name("sink")
            .unwrap()
            .dynamic_cast::<AppSink>()
            .unwrap();

        // Disable sync to process frames as fast as possible
        sink.set_property("sync", false);

        let (tx, rx) = mpsc::channel();
        let mut width = 0u32;
        let mut height = 0u32;

        sink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = sink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                    let buffer = sample.buffer().ok_or(gst::FlowError::Error)?;
                    let caps = sample.caps().ok_or(gst::FlowError::Error)?;

                    let video_info =
                        VideoInfo::from_caps(caps).map_err(|_| gst::FlowError::Error)?;
                    let width = video_info.width() as usize;
                    let height = video_info.height() as usize;

                    let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
                    let data = map.as_slice();

                    let color_image = egui::ColorImage::from_rgb([width, height], data);
                    let _ = tx.send((color_image, width as u32, height as u32));

                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );

        // Start pipeline
        pipeline.set_state(State::Playing)?;

        let mut frames = Vec::new();

        // Collect frames and wait for completion
        let bus = pipeline.bus().unwrap();
        loop {
            // Check for new frames
            while let Ok((frame, w, h)) = rx.try_recv() {
                if width == 0 {
                    width = w;
                    height = h;
                }
                frames.push(frame);

                // Log progress every 100 frames
                if frames.len() % 100 == 0 {
                    tracing::info!("Loaded {} frames so far...", frames.len());
                }
            }

            // Check for pipeline messages
            if let Some(msg) = bus.timed_pop(gst::ClockTime::from_mseconds(100)) {
                match msg.view() {
                    MessageView::Eos(..) => break,
                    MessageView::Error(err) => {
                        anyhow::bail!("GStreamer error: {}", err.error());
                    }
                    _ => {}
                }
            }
        }

        // Collect any remaining frames
        while let Ok((frame, w, h)) = rx.try_recv() {
            if width == 0 {
                width = w;
                height = h;
            }
            frames.push(frame);
        }

        // Final count
        if frames.len() % 100 != 0 {
            tracing::info!("Loaded {} frames total", frames.len());
        }

        pipeline.set_state(State::Null)?;

        // Estimate FPS (30 fps default if we can't determine)
        let fps = 30.0; // TODO: Extract this from metadata if needed

        tracing::info!(
            "Loaded {} frames ({}x{}) @ {:.2} fps",
            frames.len(),
            width,
            height,
            fps
        );

        Ok(Self {
            frames,
            width,
            height,
            fps,
        })
    }

    pub fn get_frame(&self, frame_number: usize) -> Option<&egui::ColorImage> {
        self.frames.get(frame_number)
    }

    pub fn get_frame_texture(
        &self,
        ctx: &egui::Context,
        frame_number: usize,
    ) -> Result<egui::TextureHandle, String> {
        let frame = self.get_frame(frame_number).ok_or("Frame not found")?;

        // Create texture options for pixelated rendering
        let texture_options = egui::TextureOptions {
            magnification: egui::TextureFilter::Nearest,
            minification: egui::TextureFilter::Nearest,
            ..Default::default()
        };

        Ok(ctx.load_texture(
            format!("frame_{}", frame_number),
            frame.clone(),
            texture_options,
        ))
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn fps(&self) -> f64 {
        self.fps
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    pub fn duration_seconds(&self) -> f64 {
        self.frames.len() as f64 / self.fps
    }
}
