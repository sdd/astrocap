extern crate core;

use anyhow::{anyhow, Result};
use clap::Parser;
use cli::Args;
use gst_video::video_frame::Readable;
use gst_video::VideoFrame;
use rtrb::RingBuffer;
use std::sync::Arc;
use std::thread;
use tracing::info;

use crate::point_extractor_consumer::PointExtractorConsumer;

mod cli;
mod gst_pipeline;
mod image_luma_extractor;
mod map_colors_2;
mod median_filter;
mod point_extractor_consumer;
mod run;
mod snapshot_consumer;
mod video_frame_to_image_buffer;
// pub(crate) mod window_renderer;

fn app_main() -> Result<()> {
    // Set up logging
    let subscriber = tracing_subscriber::fmt()
        .event_format(
            tracing_subscriber::fmt::format()
                .with_target(false) // don't include targets
                .with_thread_ids(false) // include the thread ID of the current thread
                .with_thread_names(false) // include the name of the current thread
                .compact(),
        )
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    let args = Args::parse();
    info!(?args, "arguments");

    if args.file.is_none() == args.uri.is_none() {
        eprintln!("Either file or uri must be specified (but not both)");
        return Err(anyhow!("Invalid Arguments"));
    }

    let shared_args: Arc<Args> = Arc::new(args);

    // Initialize GStreamer
    gst::init()?;

    let rec = rerun::RecordingStreamBuilder::new("astrocap").connect_grpc()?;

    let (prod, cons) = RingBuffer::<Arc<(isize, VideoFrame<Readable>)>>::new(5);

    let prod_thread_args = shared_args.clone();
    let prod_thread = thread::spawn(move || gst_pipeline::produce_frames(prod_thread_args, prod));

    let cons_thread_args = shared_args.clone();
    let cons_thread = thread::spawn(move || {
        let mut consumer = PointExtractorConsumer::new(cons_thread_args, rec);
        consumer.consume_frames_to_extracted_point_stream(cons)
    });

    cons_thread.join().expect("problem with consumer thread");
    let _ = prod_thread.join().expect("Problem with producer thread");

    Ok(())
}

// #[show_image::main]
fn main() {
    match run::run(app_main) {
        Ok(r) => r,
        Err(e) => eprintln!("Error! {e}"),
    }
}
