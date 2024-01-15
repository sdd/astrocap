use gst_video::video_frame::Readable;
use gst_video::VideoFrame;
use image::{EncodableLayout, ImageBuffer, Luma, Pixel, PixelWithColorType};
use std::ops::Deref;
use std::sync::Arc;

pub(crate) struct VideoFrameExt<'a>(pub &'a VideoFrame<Readable>);

impl VideoFrameExt<'_> {
    #[allow(dead_code)]
    pub fn as_img_buf_ref(&self) -> ImageBuffer<Luma<u8>, &[u8]> {
        ImageBuffer::<Luma<u8>, &[u8]>::from_raw(
            self.0.width(),
            self.0.height(),
            self.0.plane_data(0).unwrap(),
        )
        .expect("Could not create ImageBuffer from VideoFrame")
    }

    #[allow(dead_code)]
    pub fn as_img_buf_arc(&self) -> ImageBuffer<Luma<u8>, Arc<[u8]>> {
        ImageBuffer::<Luma<u8>, Arc<[u8]>>::from_raw(
            self.0.width(),
            self.0.height(),
            Arc::from(self.0.plane_data(0).unwrap()),
        )
        .expect("Could not create ImageBuffer<_, Arc<_>> from VideoFrame")
    }

    #[allow(dead_code)]
    pub fn into_img_buf(self: Self) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let buf_cloned: Vec<u8> = self.0.plane_data(0).unwrap().into();
        ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(self.0.width(), self.0.height(), buf_cloned)
            .expect("Could not create ImageBuffer from VideoFrame")
    }
}
