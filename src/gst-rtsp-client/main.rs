use std::ops::Deref;
use anyhow::Result;
use clap::Parser;
use derive_more::{Display, Error};
use gst::{BufferRef, Bus, element_error, glib, Pipeline};
use gst::prelude::*;
use gst_video::{VideoFormat, VideoFrameRef};
use http::uri::Uri;
use image::{EncodableLayout, GenericImageView, ImageBuffer, Luma, Pixel, PixelWithColorType, Rgb};
use image::flat::View;

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
    uri: Uri,

    #[clap(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

fn app_main() -> Result<()> {
    let args = Args::parse();
    assert_eq!(args.uri.scheme_str(), Some("rtsp"));

    // Initialize GStreamer
    gst::init()?;

    let pipeline = build_rtsp_client_pipeline(
        &args.uri,
        VideoFormat::Gray8,
        move |frame| {
            save_frame_to_file::<Luma<u8>>(frame, "screenshot-luma-only.png")
        }
    )?;

    run_pipeline(pipeline)
}

fn run_pipeline(pipeline: Pipeline) -> Result<()> {
    let bus = pipeline.bus().expect("Could not get the pipeline bus");
    println!("Got the bus");

    pipeline.set_state(gst::State::Playing)?;
    println!("Started the pipeline");

    for msg in bus.iter_timed(gst::ClockTime::NONE) {
        use gst::MessageView;
        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                pipeline.set_state(gst::State::Null)?;
                return Err(ErrorMessage {
                    src: msg
                        .src()
                        .map(|s| s.path_string())
                        .unwrap_or_else(|| glib::GString::from("UNKNOWN")),
                    error: err.error(),
                    debug: err.debug(),
                }
                    .into());
            },
            _ => (),
        }
    }

    println!("Stopping the pipeline");
    pipeline.set_state(gst::State::Null)?;

    Ok(())
}

fn build_rtsp_client_pipeline<
    F: FnMut(VideoFrameRef<&BufferRef>) -> () + Send + 'static,
>(uri: &Uri, target_format: VideoFormat, mut frame_handler: F) -> Result<Pipeline> {
    let pipeline_str = format!(
        "rtspsrc location={} latency=0 ! queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink name=sink",
        uri
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

    let mut got_snapshot = false;

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            // Add a handler to the "new-sample" signal.
            .new_sample(move |appsink| {
                println!("Got a sample");

                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let buffer = sample.buffer().ok_or_else(|| {
                    element_error!(
                        appsink,
                        gst::ResourceError::Failed,
                        ("Failed to get buffer from appsink")
                    );

                    gst::FlowError::Error
                })?;

                // Make sure that we only get a single buffer
                if got_snapshot {
                    return Err(gst::FlowError::Eos);
                }
                got_snapshot = true;

                let caps = sample.caps().expect("Sample without caps");
                let info = gst_video::VideoInfo::from_caps(caps).expect("Failed to parse caps");

                let frame = gst_video::VideoFrameRef::from_buffer_ref_readable(
                    buffer,
                    &info,
                ).map_err(|_| {
                    element_error!(
                            appsink,
                            gst::ResourceError::Failed,
                            ("Failed to map buffer readable")
                        );

                    gst::FlowError::Error
                })?;
                println!("frame info: {:?}", frame.info());

                // Process the video frame here
                frame_handler(frame);

                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    println!("Connected the new-sample event handler");

    Ok(pipeline)
}

fn save_frame_to_file<P: Pixel>(frame: VideoFrameRef<&BufferRef>, path: &str)
    where for<'a> &'a[u8]: Deref<Target = [P::Subpixel]>,
          [P::Subpixel]: EncodableLayout,
          P: PixelWithColorType,
{
    let img = ImageBuffer::<P, &[u8]>::from_raw(
        frame.width(), frame.height(), frame.plane_data(0).unwrap()
    ).unwrap();
    img.save(path).unwrap();
    println!("Saved a screenshot");
}

#[allow(dead_code)]
fn save_video_frame_with_non_packed_pixels_to_file(frame: VideoFrameRef<&BufferRef>, path: &str) {
    // the RGBx pixel layout in gstreamer includes an unused byte
    // after each RGB triplet. This differs from what the image crate expects
    // in order to be able to directly create an ImageBuffer,
    // which is packed RGB with no unused byte padding the end of each pixel.

    // First we need to tell image what the source layout is
    let layout = image::flat::SampleLayout {
        channels: 3,       // RGB
        channel_stride: 1, // 1 byte from component to component
        width: frame.width(),
        width_stride: 4, // 4 byte from pixel to pixel (skipping the unused fourth byte per pixel)
        height: frame.height(),
        height_stride: frame.plane_stride()[0] as usize, // stride from line to line
    };
    println!("layout: {:?}", &layout);

    // Then create a FlatSamples around the borrowed video frame data from GStreamer with
    // the correct stride as provided by GStreamer.
    let flat_samples = image::FlatSamples::<&[u8]> {
        samples: frame.plane_data(0).unwrap(),
        layout,
        color_hint: Some(image::ColorType::Rgb8),
    };

    // we now create a View onto the flat_samples that lets us iterate
    // over each pixel, removing the padded extra byte from each pixel
    let view: View<&[u8], Rgb<u8>> = flat_samples.as_view().unwrap();

    // we collect the pixels into a new buffer. This packs the pixels to remove
    // the padding
    let buffer: Vec<_> = view.pixels().flat_map(|p| p.2.0).collect();

    // now that we have a packed pixel format, we can wrap the buffer into an ImageBuffer,
    // giving us more capabilities
    let ib = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
        view.width(), view.height(), buffer
    ).unwrap();

    // we can now use the ImageBuffer::save convenience method to save the frame as a PNG
    ib.save(path).unwrap();
}

fn main() {
    match run::run(app_main) {
        Ok(r) => r,
        Err(e) => eprintln!("Error! {e}"),
    }
}
