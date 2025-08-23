use anyhow::Result;
use gst::prelude::*;
use gst::Pipeline;
use std::sync::Arc;

use crate::astrocap_frame_queue::AstrocapFrameQueue;
use crate::astrocap_gst_sink::AstrocapGstSink;
use crate::frame_buffer_pool::{create_shared_pool, SharedFrameBufferPool};

pub fn build_file_pipeline(
    path: &str,
    ring_buffer: Arc<AstrocapFrameQueue>,
    pool: Option<SharedFrameBufferPool>,
) -> Result<Pipeline> {
    let pipeline_str = format!("filesrc location={} ! decodebin", path);
    build_generic_pipeline(&pipeline_str, ring_buffer, pool)
}

pub fn build_rtsp_client_pipeline(
    uri: &str, // Changed from &Uri to &str to match caller
    ring_buffer: Arc<AstrocapFrameQueue>,
    pool: Option<SharedFrameBufferPool>,
) -> Result<Pipeline> {
    let pipeline_str = format!("rtspsrc location={} ! decodebin", uri);
    build_generic_pipeline(&pipeline_str, ring_buffer, pool)
}

// if pool is None, the astrocapsink will run in non-ZC mode
fn build_generic_pipeline(
    source_pipeline: &str,
    ring_buffer: Arc<AstrocapFrameQueue>,
    pool: Option<SharedFrameBufferPool>,
) -> Result<Pipeline> {
    let pipeline = gst::parse_launch(source_pipeline)?;
    let pipeline = pipeline
        .downcast::<Pipeline>()
        .expect("Expected a gst::Pipeline");

    // Create videoconvert for format conversion
    let videoconvert = gst::ElementFactory::make("videoconvert")
        .name("convert")
        .build()?;

    let astrocapsink = AstrocapGstSink::new();
    if let Some(pool) = pool {
        astrocapsink.set_frame_buffer_pool(pool);
    }
    astrocapsink.set_astrocap_frame_queue(ring_buffer.clone());

    // Add both elements to pipeline
    pipeline.add_many([&videoconvert, astrocapsink.upcast_ref()])?;

    // Link videoconvert to astrocapsink with explicit caps
    let caps = gst::Caps::builder("video/x-raw")
        .field("format", "GRAY8")
        .field("width", ring_buffer.frame_info().width as i32)
        .field("height", ring_buffer.frame_info().height as i32)
        .build();

    videoconvert.link_filtered(&astrocapsink, &caps)?;

    // Find decodebin and connect its pad-added signal
    let mut decodebin = None;
    for element in pipeline.iterate_elements() {
        if let Ok(element) = element {
            if let Some(factory) = element.factory() {
                if factory.name().as_str().contains("decodebin") {
                    decodebin = Some(element);
                    break;
                }
            }
        }
    }

    let decodebin = decodebin.ok_or_else(|| anyhow::anyhow!("Could not find decodebin element"))?;

    // Connect decodebin's dynamic pad to videoconvert
    let videoconvert_weak = videoconvert.downgrade();
    decodebin.connect_pad_added(move |_element, src_pad| {
        let Some(videoconvert) = videoconvert_weak.upgrade() else {
            tracing::warn!("VideoConvert was dropped, cannot link pad");
            return;
        };

        let sink_pad = videoconvert
            .static_pad("sink")
            .expect("videoconvert should have sink pad");

        if sink_pad.is_linked() {
            tracing::debug!("VideoConvert sink pad already linked, ignoring");
            return;
        }

        let caps = src_pad
            .current_caps()
            .or_else(|| Some(src_pad.query_caps(None)));
        if let Some(caps) = caps {
            let structure = caps.structure(0).unwrap();
            let media_type = structure.name();

            tracing::info!(
                "Connecting decodebin pad with caps: {} to videoconvert",
                media_type
            );

            // Only link video pads
            if media_type.starts_with("video/") {
                if let Err(e) = src_pad.link(&sink_pad) {
                    tracing::error!("Failed to link decodebin to videoconvert: {}", e);
                } else {
                    tracing::info!(
                        "✅ Successfully linked decodebin → videoconvert → astrocapsink"
                    );
                }
            }
        } else {
            tracing::warn!("No caps available on decodebin src pad");
        }
    });

    Ok(pipeline)
}
