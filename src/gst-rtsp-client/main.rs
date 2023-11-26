use std::mem::MaybeUninit;
use std::ops::Deref;
use std::path::Path;
use std::sync::Arc;
use anyhow::{anyhow, Result};
use clap::Parser;
use derive_more::{Display, Error};
use gst::{BufferRef, element_error, glib, Pipeline};
use gst::prelude::*;
use gst_app::AppSink;
use gst_video::{VideoFormat, VideoFrame, VideoFrameRef};
use http::uri::Uri;
use image::{EncodableLayout, GenericImageView, ImageBuffer, Luma, Pixel, PixelWithColorType, Rgb};
use image::flat::View;
use ringbuf::{Consumer, Producer, Rb, SharedRb};
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use gst_video::video_frame::Readable;
use tracing::{debug, info, instrument, Level, span};

mod run;

#[derive(Debug, Display, Error)]
#[display(fmt = "Received error from {src}: {error} (debug: {debug:?})")]
struct ErrorMessage {
    src: glib::GString,
    error: glib::Error,
    debug: Option<glib::GString>,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// rtsp URI to the host
    #[arg(short, long)]
    uri: Option<Uri>,

    /// video file to read from
    #[arg(short, long)]
    file: Option<String>,

    #[clap(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

fn app_main() -> Result<()> {
    // Set up logging
    let subscriber = tracing_subscriber::FmtSubscriber::new();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();
    info!(?args, "arguments");

    if args.file.is_none() == args.uri.is_none() {
        eprintln!("Either file or uri must be specified (but not both)");
        return Err(anyhow!("Invalid Arguments"));
    }

    // Initialize GStreamer
    gst::init()?;

    // Set up the ring buffer
    let rb = SharedRb::<VideoFrame<Readable>, Vec<_>>::new(5);
    let (prod, cons) = rb.split();

    let cons_thread = thread::spawn(move || consume_frames_to_snapshot_files(cons));

    let prod_thread = thread::spawn(move || produce_frames(&args, prod));

    cons_thread.join().expect("problem with consumer thread");
    prod_thread.join().expect("Problem with producer thread");

    Ok(())
}

#[instrument(skip_all)]
fn consume_frames_to_snapshot_files(mut cons: Consumer<VideoFrame<Readable>, Arc<SharedRb<VideoFrame<Readable>, Vec<MaybeUninit<VideoFrame<Readable>>>>>>) {
    let mut idx = 0;

    loop {
        sleep(Duration::from_millis(50));
        if let Some(frame) = cons.pop() {
            info!("Consuming a frame");
            save_frame_to_file::<Luma<u8>>(frame, &format!("screenshot-luma-only-{}.png", idx));
            info!("Saved a frame");
            idx += 1;
        }
    }
}

#[instrument(skip_all)]
fn produce_frames(args: &Args, mut prod: Producer<VideoFrame<Readable>, Arc<SharedRb<VideoFrame<Readable>, Vec<MaybeUninit<VideoFrame<Readable>>>>>>) -> Result<()> {
    let (pipeline, appsink) = if let Some(uri) = &args.uri {
        build_rtsp_client_pipeline(
            uri,
            VideoFormat::Gray8,
        )
    } else if let Some(path) = &args.file {
        build_file_pipeline(
            path,
            VideoFormat::Gray8,
        )
    } else {
        return Err(anyhow!("Invalid Arguments"));
    }.unwrap();

    info!("Trying to Play");
    pipeline.set_state(gst::State::Playing).unwrap();
    info!("Playing");

    loop {
        let sample = appsink.pull_sample().unwrap();
        info!("Got a Sample");
        let buffer = sample.buffer_owned().unwrap();
        debug!("got a Buffer");

        let caps = sample.caps().expect("Sample without caps");
        debug!("got Caps");
        let info = gst_video::VideoInfo::from_caps(caps).expect("Failed to parse caps");
        debug!("got VideoInfo");

        let frame = gst_video::VideoFrame::from_buffer_readable(
            buffer,
            &info,
        ).expect("Could not create VideoFrame from Buffer and VideoInfo");
        info!("got a VideoFrame");

        if !prod.is_full() {
            prod.push(frame).expect("Could not push frame to ring buffer");
            info!("pushed to the RingBuffer");
        } else {
            debug!("skipped the RingBuffer");
        }
    }
}

fn build_rtsp_client_pipeline(uri: &Uri, target_format: VideoFormat) -> Result<(Pipeline, AppSink)> {
    let pipeline_str = format!(
        "rtspsrc location={} latency=0 ! queue ! rtph264depay ! h264parse ! avdec_h264",
        uri
    );

    build_generic_pipeline(&pipeline_str, target_format)
}

fn build_file_pipeline(path: &str, target_format: VideoFormat) -> Result<(Pipeline, AppSink)> {
    let pipeline_str = format!(
        "filesrc location={} ! decodebin",
        path
    );

    build_generic_pipeline(&pipeline_str, target_format)
}

fn build_generic_pipeline(pipeline_str: &str, target_format: VideoFormat) -> Result<(Pipeline, AppSink)> {
    let pipeline_str = format!(
        "{} ! videoconvert ! appsink name=sink",
        pipeline_str
    );
    let pipeline = gst::parse_launch(&pipeline_str)?
        .downcast::<gst::Pipeline>()
        .expect("Expected a gst::Pipeline");

    // Get access to the appsink element.
    let appsink = pipeline
        .by_name("sink")
        .expect("Sink element not found")
        .downcast::<gst_app::AppSink>()
        .expect("Sink element is expected to be an appsink!");

    // indicate to the pipeline that we want the resultant video to have 8 bit greyscale pixel format
    appsink.set_caps(Some(
        &gst_video::VideoCapsBuilder::new()
            .format(target_format)
            .build(),
    ));

    Ok((pipeline, appsink))
}

fn save_frame_to_file<P: Pixel>(frame: VideoFrame<Readable>, path: &str)
    where for<'a> &'a[u8]: Deref<Target = [P::Subpixel]>,
          [P::Subpixel]: EncodableLayout,
          P: PixelWithColorType,
{
    let img = ImageBuffer::<P, &[u8]>::from_raw(
        frame.width(), frame.height(), frame.plane_data(0).unwrap()
    ).unwrap();
    img.save(path).unwrap();
}

fn main() {
    match run::run(app_main) {
        Ok(r) => r,
        Err(e) => eprintln!("Error! {e}"),
    }
}
