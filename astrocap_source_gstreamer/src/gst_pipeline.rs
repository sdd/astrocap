use std::sync::Arc;

use gst::glib::Cast;
use gst::prelude::GstBinExt;
use gst::prelude::*;
use gst::Pipeline;
use http::Uri;

use crate::frame_buffer::FrameBuffer;
use crate::ring_buffer_sink::RingBufferSink;

pub fn build_rtsp_client_pipeline(
    uri: &Uri,
    ring_buffer: Arc<FrameBuffer>,
) -> anyhow::Result<Pipeline> {
    let pipeline_str = format!(
        "rtspsrc location={} latency=0 ! queue ! rtph265depay ! h265parse ! avdec_h265",
        uri
    );

    build_generic_pipeline(&pipeline_str, ring_buffer)
}

pub fn build_file_pipeline(path: &str, ring_buffer: Arc<FrameBuffer>) -> anyhow::Result<Pipeline> {
    let pipeline_str = format!("filesrc location={} ! decodebin", path);

    build_generic_pipeline(&pipeline_str, ring_buffer)
}

fn build_generic_pipeline(
    input_description: &str,
    ring_buffer: Arc<FrameBuffer>,
) -> anyhow::Result<Pipeline> {
    gst::Element::register(
        None,
        "ringbuffersink",
        gst::Rank::None,
        RingBufferSink::static_type(),
    )?;

    let pipeline_str = format!(
        "{} ! videoconvert ! video/x-raw,format=GRAY8 ! ringbuffersink name=sink",
        input_description
    );

    let pipeline = gst::parse_launch(&pipeline_str)?
        .downcast::<Pipeline>()
        .expect("Expected a gst::Pipeline");

    let sink = pipeline
        .by_name("sink")
        .unwrap()
        .downcast::<RingBufferSink>()
        .unwrap();

    sink.set_buffer(ring_buffer);

    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_stage_name() {
        assert_eq!(normalize_stage_name("decodebin0"), "decodebin");
        assert_eq!(normalize_stage_name("decodebin1"), "decodebin");
        assert_eq!(normalize_stage_name("decodebin123"), "decodebin");
        assert_eq!(normalize_stage_name("queue"), "queue");
        assert_eq!(normalize_stage_name("videoconvert2"), "videoconvert");
        assert_eq!(normalize_stage_name("capsfilter3"), "capsfilter");
    }
}
