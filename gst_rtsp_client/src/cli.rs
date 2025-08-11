use clap::Parser;
use derive_more::{Display, Error};
use gst::glib;
use http::Uri;
use std::path::PathBuf;

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

    /// mask image
    #[arg(short, long)]
    pub(crate) mask: Option<String>,

    /// star index path to enable solver
    #[arg(short, long)]
    pub(crate) star_index_path: Option<PathBuf>,

    #[clap(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}
