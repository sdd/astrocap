use image::{GenericImageView, ImageBuffer, Pixel};
use std::cmp::{max, min};
use std::ops::Deref;

#[must_use = "the function does not modify the original image"]
pub fn median_filter<C, P>(
    image: &ImageBuffer<P, C>,
    x_radius: u32,
    y_radius: u32,
) -> ImageBuffer<P, Vec<u8>>
where
    P: Pixel<Subpixel = u8>,
    C: Deref<Target = [P::Subpixel]>,
{
    let (width, height) = image.dimensions();

    if width == 0 || height == 0 {
        return ImageBuffer::from_raw(image.width(), image.height(), image.to_vec()).unwrap();
    }

    let mut out = ImageBuffer::<P, Vec<u8>>::new(width, height);
    let rx = x_radius as i32;
    let ry = y_radius as i32;

    let mut hist = initialise_histogram_for_top_left_pixel(image, x_radius, y_radius);
    slide_down_column(&mut hist, image, &mut out, 0, rx, ry);

    for x in 1..width {
        if x % 2 == 0 {
            slide_right(&mut hist, image, x, 0, rx, ry);
            slide_down_column(&mut hist, image, &mut out, x, rx, ry);
        } else {
            slide_right(&mut hist, image, x, height - 1, rx, ry);
            slide_up_column(&mut hist, image, &mut out, x, rx, ry);
        }
    }
    out
}

fn initialise_histogram_for_top_left_pixel<P, C>(
    image: &ImageBuffer<P, C>,
    x_radius: u32,
    y_radius: u32,
) -> HistSet
where
    P: Pixel<Subpixel = u8>,
    C: Deref<Target = [P::Subpixel]>,
{
    let (width, height) = image.dimensions();
    let kernel_size = (2 * x_radius + 1) * (2 * y_radius + 1);
    let num_channels = P::CHANNEL_COUNT;

    let mut hist = HistSet::new(num_channels, kernel_size);
    let rx = x_radius as i32;
    let ry = y_radius as i32;

    for dy in -ry..(ry + 1) {
        let py = min(max(0, dy), height as i32 - 1) as u32;

        for dx in -rx..(rx + 1) {
            let px = min(max(0, dx), width as i32 - 1) as u32;

            hist.incr(image, px, py);
        }
    }

    hist
}

fn slide_right<P, C>(
    hist: &mut HistSet,
    image: &ImageBuffer<P, C>,
    x: u32,
    y: u32,
    rx: i32,
    ry: i32,
) where
    P: Pixel<Subpixel = u8>,
    C: Deref<Target = [P::Subpixel]>,
{
    let (width, height) = image.dimensions();

    let prev_x = max(0, x as i32 - rx - 1) as u32;
    let next_x = min(x as i32 + rx, width as i32 - 1) as u32;

    for dy in -ry..(ry + 1) {
        let py = min(max(0, y as i32 + dy), (height - 1) as i32) as u32;

        hist.decr(image, prev_x, py);
        hist.incr(image, next_x, py);
    }
}

fn slide_down_column<P, C>(
    hist: &mut HistSet,
    image: &ImageBuffer<P, C>,
    out: &mut ImageBuffer<P, Vec<u8>>,
    x: u32,
    rx: i32,
    ry: i32,
) where
    P: Pixel<Subpixel = u8>,
    C: Deref<Target = [P::Subpixel]>,
{
    let (width, height) = image.dimensions();
    hist.set_to_median(out, x, 0);

    for y in 1..height {
        let prev_y = max(0, y as i32 - ry - 1) as u32;
        let next_y = min(y as i32 + ry, height as i32 - 1) as u32;

        for dx in -rx..(rx + 1) {
            let px = min(max(0, x as i32 + dx), (width - 1) as i32) as u32;

            hist.decr(image, px, prev_y);
            hist.incr(image, px, next_y);
        }

        hist.set_to_median(out, x, y);
    }
}

fn slide_up_column<P, C>(
    hist: &mut HistSet,
    image: &ImageBuffer<P, C>,
    out: &mut ImageBuffer<P, Vec<u8>>,
    x: u32,
    rx: i32,
    ry: i32,
) where
    P: Pixel<Subpixel = u8>,
    C: Deref<Target = [P::Subpixel]>,
{
    let (width, height) = image.dimensions();
    hist.set_to_median(out, x, height - 1);

    for y in (0..(height - 1)).rev() {
        let prev_y = min(y as i32 + ry + 1, height as i32 - 1) as u32;
        let next_y = max(0, y as i32 - ry) as u32;

        for dx in -rx..(rx + 1) {
            let px = min(max(0, x as i32 + dx), (width - 1) as i32) as u32;

            hist.decr(image, px, prev_y);
            hist.incr(image, px, next_y);
        }

        hist.set_to_median(out, x, y);
    }
}

// A collection of 256-slot histograms, one per image channel.
// Used to implement median_filter.
struct HistSet {
    // One histogram per image channel.
    data: Vec<[u32; 256]>,
    // Calls to `median` will only return the correct answer
    // if there are `expected_count` entries in the relevant
    // histogram in `data`.
    expected_count: u32,
}

impl HistSet {
    fn new(num_channels: u8, expected_count: u32) -> HistSet {
        // Can't use vec![[0u32; 256], num_channels as usize]
        // because arrays of length > 32 aren't cloneable.
        let mut data = vec![];
        for _ in 0..num_channels {
            data.push([0u32; 256]);
        }

        HistSet {
            data,
            expected_count,
        }
    }

    fn incr<P, C>(&mut self, image: &ImageBuffer<P, C>, x: u32, y: u32)
    where
        P: Pixel<Subpixel = u8>,
        C: Deref<Target = [P::Subpixel]>,
    {
        unsafe {
            let pixel = image.unsafe_get_pixel(x, y);
            let channels = pixel.channels();
            for c in 0..channels.len() {
                let p = *channels.get_unchecked(c) as usize;
                let hist = self.data.get_unchecked_mut(c);
                *hist.get_unchecked_mut(p) += 1;
            }
        }
    }

    fn decr<P, C>(&mut self, image: &ImageBuffer<P, C>, x: u32, y: u32)
    where
        P: Pixel<Subpixel = u8>,
        C: Deref<Target = [P::Subpixel]>,
    {
        unsafe {
            let pixel = image.unsafe_get_pixel(x, y);
            let channels = pixel.channels();
            for c in 0..channels.len() {
                let p = *channels.get_unchecked(c) as usize;
                let hist = self.data.get_unchecked_mut(c);
                *hist.get_unchecked_mut(p) -= 1;
            }
        }
    }

    fn set_to_median<P>(&self, image: &mut ImageBuffer<P, Vec<u8>>, x: u32, y: u32)
    where
        P: Pixel<Subpixel = u8>,
    {
        unsafe {
            let target = image.get_pixel_mut(x, y);
            let channels = target.channels_mut();
            for c in 0..channels.len() {
                *channels.get_unchecked_mut(c) = self.channel_median(c as u8);
            }
        }
    }

    fn channel_median(&self, c: u8) -> u8 {
        let hist = unsafe { self.data.get_unchecked(c as usize) };

        let mut count = 0;

        for i in 0..256 {
            unsafe {
                count += *hist.get_unchecked(i);
            }

            if 2 * count >= self.expected_count {
                return i as u8;
            }
        }

        255
    }
}
