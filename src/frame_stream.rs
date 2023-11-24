use futures_core::FusedStream;
use crate::frame::Frame;

pub type FrameStream = dyn FusedStream<Item=Frame>;



// we want three types of stream here:
//
// 1. Frames from a video file. Process them as fast as possible. Use flow control so that no frames get dropped.
//
// 2. Live video. We want to buffer a certain number of frames but drop any in excess of that once the buffer overflows
//
// 3. Pseudo-live. Treat a video file as if it was live for the purposes of performance testing.
//      Generate a frame every n milliseconds and discard any unprocessed frames if the buffer starts to overflow.macro_rules!
