mod utils;

use anyhow::Result;
use clap::Parser;

use opencv::{
    prelude::*,
    videoio::{self, VideoCapture},
};

/// Parse and display metadata from an MKV file
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// path to the file
    path: std::path::PathBuf,

    #[clap(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let str_path = args.path.to_str().expect("Could not parse filename from path argument");

    println!("Attempting to read file '{:?}'", &str_path);

    let mut vidcap = VideoCapture::from_file(
        str_path,
        videoio::CAP_ANY
    ).expect(&format!("OpenCV Could not open the file '{:?}'", &args.path));

    if !VideoCapture::is_opened(&vidcap)? {
        panic!("OpenCV Could not open the file '{:?}'", &args.path);
    }

    let mut _frame = Mat::default();
    let mut frame_number = 0;

    loop {

        // reads frames one by onex
        vidcap.read(&mut _frame)?;

        if utils::empty(&_frame)? {
            println!("No more video frames to read");
            break;
        }

        println!("Frame: #{}", frame_number);
        frame_number = frame_number + 1;
    }

    Ok(())
}
