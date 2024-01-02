use http::Uri;
use gst_video::{VideoFormat, VideoFrame};
use gst::Pipeline;
use gst_app::AppSink;
use gst::glib::Cast;
use gst::prelude::{ElementExt, GstBinExt};
use tracing::{debug, info, instrument};
use rtrb::Producer;
use std::sync::Arc;
use gst_video::video_frame::Readable;
use anyhow::anyhow;
use crate::cli::Args;

pub fn build_rtsp_client_pipeline(uri: &Uri, target_format: VideoFormat) -> anyhow::Result<(Pipeline, AppSink)> {
    let pipeline_str = format!(
        "rtspsrc location={} latency=0 ! queue ! rtph264depay ! h264parse ! avdec_h264",
        uri
    );

    build_generic_pipeline(&pipeline_str, target_format)
}

pub fn build_file_pipeline(path: &str, target_format: VideoFormat) -> anyhow::Result<(Pipeline, AppSink)> {
    let pipeline_str = format!(
        "filesrc location={} ! decodebin",
        path
    );

    build_generic_pipeline(&pipeline_str, target_format)
}

fn build_generic_pipeline(pipeline_str: &str, target_format: VideoFormat) -> anyhow::Result<(Pipeline, AppSink)> {
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

#[instrument(skip_all)]
pub fn produce_frames(args: Arc<Args>, mut prod: Producer<Arc<(isize, VideoFrame<Readable>)>>) -> anyhow::Result<()> {
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

    debug!("Trying to Play");
    pipeline.set_state(gst::State::Playing).unwrap();
    info!("Playing");

    let mut frame_index = 0;

    loop {
        let sample = appsink.pull_sample().unwrap();
        debug!("Got a Sample");
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
        debug!("got a VideoFrame");

        if !prod.is_full() {
            prod.push(Arc::new((frame_index, frame))).expect("Could not push frame to ring buffer");
            debug!("pushed to the RingBuffer");
        } else {
            debug!("skipped the RingBuffer");
        }

        frame_index += 1;
    }
}
