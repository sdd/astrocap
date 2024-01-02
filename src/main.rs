use std::io;
// use eye::prelude::*;

// This bin is disabled, due to eye having a transient dependency on clang-sys
// which runs in `runtime` mode, but opencv needs to use clang-sys in non-runtime mode

fn main() -> io::Result<()> {
    // Create a context
    // let ctx = Context::new();

    // Query for available devices.
    // let devices = ctx.query_devices()?;
    // if devices.is_empty() {
    //     return Err(io::Error::new(io::ErrorKind::Other, "No devices available"));
    // }

    // First, we need a capture device to read images from. For this example, let's just choose
    // whatever device is first in the list.
    // let dev = Device::with_uri(&devices[0])?;

    // Query for available streams and just choose the first one.
    // let streams = dev.query_streams()?;
    // let stream_desc = streams[0].clone();
    // println!("Stream: {:?}", stream_desc);

    // Since we want to capture images, we need to access the native image stream of the device.
    // The backend will internally select a suitable implementation for the platform stream. On
    // Linux for example, most devices support memory-mapped buffers.
    // let mut stream = dev.start_stream(&stream_desc)?;

    // Here we create a loop and just capture images as long as the device produces them. Normally,
    // this loop will run forever unless we unplug the camera or exit the program.
    // loop {
    //     let _frame = stream
    //         .next()
    //         .expect("Stream is dead")
    //         .expect("Failed to capture frame");
    // }
    Ok(())
}