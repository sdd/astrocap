use image::GrayImage;
use crate::point_extractor_consumer::ImagePointCandidate;

const POINT_SKIP_STEP: u32 = 3;
const POINT_EXCLUSION_RADIUS: f64 = 20.0;
// const POINT_EXCLUSION_RADIUS_2: f64 = 400.0;
const POINT_THRESHOLD: u8 = 45;
const STEP_X: u32 = 1;
const STEP_Y: u32 = 1;
pub const PATCH_SIZE: u32 = 20;

const INITIAL_RADIUS: f64 = 3.0;

pub trait PointDetector {
    fn extract_from_img(&self, img: &GrayImage) -> Vec<ImagePointCandidate>;
}

pub struct PointDetectPeak {}

impl PointDetector for PointDetectPeak {
    fn extract_from_img(&self, img: &GrayImage) -> Vec<ImagePointCandidate> {
        let mut points: Vec<ImagePointCandidate> = vec![];
        let mut open_points: Vec<ImagePointCandidate> = vec![];

        let (img_w, img_h) = img.dimensions();
        let mut x: u32 = 0;
        let mut y: u32 = 0;

        while y < img_h {
            open_points = open_points.into_iter().filter(|p| y as f64 - p.y < POINT_EXCLUSION_RADIUS).collect();

            while x < img_w {
                if inside_existing_point(x, y, &open_points) {
                    x += POINT_SKIP_STEP;
                    continue;
                }

                let val: u8 = img.get_pixel(x, y)[0];

                if val > POINT_THRESHOLD {
                    let mut curr_val: u8 = val;
                    let mut point_x = x;
                    while point_x + 1 < img_w && img.get_pixel(point_x + 1, y)[0] > curr_val {
                        point_x += 1;
                        curr_val = img.get_pixel(point_x, y)[0];
                    }

                    let mut point_y = y;
                    while point_y + 1 < img_h && img.get_pixel(point_x, point_y + 1)[0] > curr_val {
                        point_y += 1;
                        curr_val = img.get_pixel(point_x, point_y)[0];
                    }

                    points.push(ImagePointCandidate {
                        x: point_x as f64,
                        y: point_y as f64,
                        amplitude: curr_val as f64,
                        radius: INITIAL_RADIUS,
                        log_likelihood: 0,
                        age: 0,
                    });

                    open_points.push(ImagePointCandidate {
                        x: point_x as f64,
                        y: point_y as f64,
                        amplitude: curr_val as f64,
                        radius: INITIAL_RADIUS,
                        log_likelihood: 0,
                        age: 0,
                    });
                }

                x += STEP_X;
            }
            x = 0;
            y += STEP_Y;
        }

        points
    }
}

pub fn inside_existing_point(x: u32, _y: u32, points: &[ImagePointCandidate]) -> bool {
    points.iter().any(|point| {
        let xd =point.x - (x as f64);//.abs();
        //let yd = (point.y - (y as f64)).abs();
        xd < POINT_EXCLUSION_RADIUS// && ((xd * xd) + (yd * yd)) < POINT_EXCLUSION_RADIUS_2
    })
}
