use astrocap_model::traits::ImageLumaExtractor;
use image::{ImageBuffer, Luma, Pixel};

pub(crate) struct Img(pub ImageBuffer<Luma<u8>, Vec<u8>>);

impl ImageLumaExtractor for Img {
    fn get_luma8_for_pixel(&self, x: u32, y: u32) -> u8 {
        self.0.get_pixel(x, y).channels()[0]
    }
    fn width(&self) -> u32 {
        self.0.width()
    }
    fn height(&self) -> u32 {
        self.0.height()
    }
}
