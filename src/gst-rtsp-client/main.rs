use anyhow::Result;
use clap::Parser;
use derive_more::{Display, Error};
use gst::{element_error, glib};
use gst::prelude::*;

mod run;

/// Connect to a RTSP video source and stream frames from it
///
///
///

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
    uri: http::uri::Uri,

    #[clap(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

fn app_main() -> Result<()> {
    let args = Args::parse();
    assert_eq!(args.uri.scheme_str(), Some("rtsp"));

    // Initialize GStreamer
    gst::init()?;

    // Create a pipeline
    let pipeline = gst::Pipeline::default();

    // for now we'll use a videotestsrc to prove out the rest before
    // switching to an rtspsrc
    let src = gst::ElementFactory::make("videotestsrc").build()?;

    let appsink = gst_app::AppSink::builder().build();

    pipeline.add_many([&src, appsink.upcast_ref()])?;
    src.link(&appsink)?;

    println!("Built and linked the pipeline");

    let bus = pipeline.bus().expect("Could not get the pipeline bus");
    println!("Got the bus");

    appsink.set_callbacks(
        gst_app::AppSinkCallbacks::builder()
            // Add a handler to the "new-sample" signal.
            .new_sample(|appsink| {
                println!("Got a frame");

                // Pull the sample in question out of the appsink's buffer.
                let sample = appsink.pull_sample().map_err(|_| gst::FlowError::Eos)?;
                let buffer = sample.buffer().ok_or_else(|| {
                    element_error!(
                        appsink,
                        gst::ResourceError::Failed,
                        ("Failed to get buffer from appsink")
                    );

                    gst::FlowError::Error
                })?;

                // At this point, buffer is only a reference to an existing memory region somewhere.
                // When we want to access its content, we have to map it while requesting the required
                // mode of access (read, read/write).
                // This type of abstraction is necessary, because the buffer in question might not be
                // on the machine's main memory itself, but rather in the GPU's memory.
                // So mapping the buffer makes the underlying memory region accessible to us.
                // See: https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/allocation.html
                let map = buffer.map_readable().map_err(|_| {
                    element_error!(
                        appsink,
                        gst::ResourceError::Failed,
                        ("Failed to map buffer readable")
                    );

                    gst::FlowError::Error
                })?;

                // Process the video frame here

                Ok(gst::FlowSuccess::Ok)
            })
            .build(),
    );
    println!("Connected the new-sample event handler");

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

fn main() {
    match run::run(app_main) {
        Ok(r) => r,
        Err(e) => eprintln!("Error! {e}"),
    }
}