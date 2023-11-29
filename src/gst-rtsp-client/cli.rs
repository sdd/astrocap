use clap::Parser;
use http::Uri;
use derive_more::{Display, Error};
use gst::glib;

#[derive(Debug, Display, Error)]
#[display(fmt = "Received error from {src}: {error} (debug: {debug:?})")]
struct ErrorMessage {
    src: glib::GString,
    error: glib::Error,
    debug: Option<glib::GString>,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// rtsp URI to the host
    #[arg(short, long)]
    pub(crate) uri: Option<Uri>,

    /// video file to read from
    #[arg(short, long)]
    pub(crate) file: Option<String>,

    #[clap(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}
