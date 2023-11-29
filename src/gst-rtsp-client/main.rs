use std::sync::Arc;
use anyhow::{anyhow, Result};
use clap::Parser;
use gst_video::VideoFrame;
use rtrb::RingBuffer;
use std::thread;
use gst_video::video_frame::Readable;
use tracing::info;
use cli::Args;

mod run;
mod cli;
mod gst_pipeline;
mod snapshot_consumer;
// mod buffer;

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

    let (prod, cons) = RingBuffer::<Arc<VideoFrame<Readable>>>::new(5);

    let cons_thread = thread::spawn(move || snapshot_consumer::consume_frames_to_snapshot_files(cons));
    let prod_thread = thread::spawn(move || gst_pipeline::produce_frames(&args, prod));

    cons_thread.join().expect("problem with consumer thread");
    let _ = prod_thread.join().expect("Problem with producer thread");

    Ok(())
}

fn main() {
    match run::run(app_main) {
        Ok(r) => r,
        Err(e) => eprintln!("Error! {e}"),
    }
}
