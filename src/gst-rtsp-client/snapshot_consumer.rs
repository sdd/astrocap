use image::{EncodableLayout, ImageBuffer, Luma, Pixel, PixelWithColorType};
use gst_video::VideoFrame;
use gst_video::video_frame::Readable;
use std::ops::Deref;
use tracing::{debug, instrument};
use rtrb::Consumer;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

#[instrument(skip_all)]
pub fn consume_frames_to_snapshot_files(cons: Consumer<Arc<VideoFrame<Readable>>>) {
    let mut idx = 0;

    loop {
        sleep(Duration::from_millis(50));
        if let Ok(arc_frame_ref) = cons.peek() {
            let frame = arc_frame_ref.clone();
            debug!("Consuming a frame");
            save_frame_to_file::<Luma<u8>>(&frame, &format!("screenshot-luma-only-{}.png", idx));
            debug!("Saved a frame");
            idx += 1;
        }
    }
}

#[allow(dead_code)]
fn save_frame_to_file<P: Pixel>(frame: &VideoFrame<Readable>, path: &str)
    where for<'a> &'a[u8]: Deref<Target = [P::Subpixel]>,
          [P::Subpixel]: EncodableLayout,
          P: PixelWithColorType,
{
    let img = ImageBuffer::<P, &[u8]>::from_raw(
        frame.width(), frame.height(), frame.plane_data(0).unwrap()
    ).unwrap();
    img.save(path).unwrap();
}
