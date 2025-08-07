use image::{GenericImage, GenericImageView, ImageBuffer, Pixel};
use imageproc::definitions::Image;
use std::ops::Deref;

pub fn map_colors<P, Q, R, F, C1, C2>(
    image1: &ImageBuffer<P, C1>,
    image2: &ImageBuffer<Q, C2>,
    f: F,
) -> Image<R>
where
    P: Pixel,
    Q: Pixel,
    R: Pixel,
    F: Fn(P, Q) -> R,
    C1: Deref<Target = [P::Subpixel]>,
    C2: Deref<Target = [Q::Subpixel]>,
{
    assert_eq!(image1.dimensions(), image2.dimensions());

    let (width, height) = image1.dimensions();
    let mut out: ImageBuffer<R, Vec<R::Subpixel>> = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            unsafe {
                let p = image1.unsafe_get_pixel(x, y);
                let q = image2.unsafe_get_pixel(x, y);
                out.unsafe_put_pixel(x, y, f(p, q));
            }
        }
    }

    out
}
