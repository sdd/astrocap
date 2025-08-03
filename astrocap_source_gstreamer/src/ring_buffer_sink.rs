use std::sync::{Arc, Mutex};

use gst::glib;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst_base::subclass::prelude::*;
use once_cell::sync::Lazy;

use crate::frame_buffer::FrameBuffer;

// Custom GObject wrapper
glib::wrapper! {
    pub struct RingBufferSink(ObjectSubclass<imp::RingBufferSink>) @extends gst_base::BaseSink, gst::Element, gst::Object;
}

impl RingBufferSink {
    pub fn new() -> Self {
        glib::Object::builder().build()
    }

    pub fn set_buffer(&self, frame_buffer: Arc<FrameBuffer>) {
        let imp = self.imp();
        *imp.buffer.lock().unwrap() = Some(frame_buffer);
    }
}

mod imp {
    use super::*;

    #[derive(Default)]
    pub struct RingBufferSink {
        pub buffer: Mutex<Option<Arc<FrameBuffer>>>,
    }

    impl RingBufferSink {
        fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
            let ring_buffer_guard = self.buffer.lock().unwrap();
            let Some(ref ring_buffer) = *ring_buffer_guard else {
                return Err(gst::FlowError::Error);
            };

            // Map the buffer for reading
            let map = buffer.map_readable().map_err(|_| gst::FlowError::Error)?;
            let data = map.as_slice();

            tracing::debug!("FrameBuffer: write_frame called");

            // Try to write frame to ring buffer
            match ring_buffer.write_frame(data) {
                Ok(()) => Ok(gst::FlowSuccess::Ok),
                Err("Buffer full") => {
                    // Pipeline should already be paused by threshold, but just in case
                    tracing::warn!("RingBufferSink: Frame rejected due to full buffer");
                    Err(gst::FlowError::Flushing) // Tell GStreamer to retry later
                }
                Err(_) => Err(gst::FlowError::Error),
            }
        }
    }

    #[glib::object_subclass]
    impl ObjectSubclass for RingBufferSink {
        const NAME: &'static str = "RingBufferSink";
        type Type = super::RingBufferSink;
        type ParentType = gst_base::BaseSink;
    }

    impl ObjectImpl for RingBufferSink {}

    impl GstObjectImpl for RingBufferSink {}

    impl ElementImpl for RingBufferSink {
        fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
            static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
                gst::subclass::ElementMetadata::new(
                    "Ring Buffer Sink",
                    "Sink/Video",
                    "Writes video frames directly to ring buffer",
                    "Scott Donnelly <scott@donnel.ly>",
                )
            });

            Some(&*ELEMENT_METADATA)
        }

        fn pad_templates() -> &'static [gst::PadTemplate] {
            static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
                let caps = gst_video::VideoCapsBuilder::new()
                    .format(gst_video::VideoFormat::Gray8)
                    .build();

                vec![gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap()]
            });

            PAD_TEMPLATES.as_ref()
        }
    }

    impl BaseSinkImpl for RingBufferSink {
        fn render(&self, buffer: &gst::Buffer) -> Result<gst::FlowSuccess, gst::FlowError> {
            self.render(buffer)
        }
    }
}

pub fn register() -> Result<(), glib::BoolError> {
    gst::Element::register(
        None,
        "ringbuffersink",
        gst::Rank::None,
        RingBufferSink::static_type(),
    )
}
