use std::error::Error;
use image::{DynamicImage, EncodableLayout, GrayImage, ImageBuffer, Luma, Pixel, PixelWithColorType, Rgba};
use gst_video::VideoFrame;
use gst_video::video_frame::Readable;
use format_num::NumberFormat;
use ordered_float::OrderedFloat;
use std::ops::{Deref, Neg};
use tracing::{debug, info, instrument};
use rtrb::Consumer;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;
use rusttype::Font;
use imageproc::drawing::{draw_hollow_circle_mut, draw_text_mut};
use imageproc::filter::median_filter;
use imageproc::map::map_colors2;

use kiddo::float::kdtree::KdTree;
use kiddo::SquaredEuclidean;
use show_image::{WindowOptions, WindowProxy};
use serde::{Deserialize, Serialize};

use crate::fit_nelder_mead::PointFitterGaussianNelderMead;
use crate::fit_nelder_mead::PointFitter;
use crate::point_detect_peak::PointDetectPeak;
use crate::point_detect_peak::PointDetector;

type Tree = KdTree<f64, usize, 2, 32, u32>;
// const ANNOTATED_IMG_MARKER_SRC_RADIUS: i32 = 10;
const MAX_POINT_MATCH_DIST: f64 = 4.0;
const MIN_EXISTING_MATCH_SCORE: f64 = 0.0;
const CANDIDATE_MAX_RADIUS: f64 = 2.5;//2.5;
const CANDIDATE_MIN_RADIUS: f64 = 1.6;//2.0;

const UNMATCHED_POINT_PENALTY: f64 = 0.5;
// const MATCHED_POINT_BENEFIT: f64 = 5.0;
const POINT_DISCARD_THRESHOLD: f64 = -5.0;
// const STARTING_LOG_LIKELIHOOD: i64 = 3;

#[derive(PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ImagePointCandidate {
    pub x: f64,
    pub y: f64,
    pub amplitude: f64,
    pub radius: f64,
    pub log_likelihood: f64,
    pub age: u64,
    pub latest_score: f64,
    pub matched_last_frame: bool,
}

impl ImagePointCandidate {
    pub fn update_with(&self, other: &ImagePointCandidate) -> Self {
        let update_scale = 0.25;//;other.log_likelihood.min(1.0) / self.log_likelihood.min(1.0);

        let diff_x = self.x - other.x;
        let update_x = diff_x * update_scale;
        let x = self.x + update_x;

        let diff_y = self.y - other.y;
        let update_y = diff_y * update_scale;
        let y = self.y + update_y;

        let diff_radius = self.radius - other.radius;
        let update_radius = diff_radius * update_scale;
        let radius = self.radius + update_radius;

        let diff_amplitude = self.amplitude - other.amplitude;
        let update_amplitude = diff_amplitude * update_scale;
        let amplitude = self.amplitude + update_amplitude;

        ImagePointCandidate {
            x,
            y,
            amplitude,
            radius,
            log_likelihood: self.log_likelihood + other.latest_score,
            age: self.age,
            latest_score: other.latest_score,
            matched_last_frame: other.matched_last_frame,
        }
    }
}

pub struct PointExtractorConsumer {
    state: Vec<ImagePointCandidate>,
    kdtree: Tree,
    window: WindowProxy,
}

impl PointExtractorConsumer {
    pub fn new() -> Self {

        let window = show_image::create_window("Annotated Frames", WindowOptions::default().set_size([1920, 1080])).expect("Could not create window");

        PointExtractorConsumer {
            state: vec![],
            kdtree: Tree::new(),
            window,
        }
    }

    pub fn update_state(&mut self, points: Vec<ImagePointCandidate>) {
        self.kdtree = Tree::with_capacity(self.state.len());
        for (idx, point) in self.state.iter().enumerate() {
            self.kdtree.add(&[point.x, point.y], idx);
        }

        if self.state.len() == 0 {
            self.state = points;
            for (idx, point) in self.state.iter().enumerate() {
                self.kdtree.add(&[point.x, point.y], idx);
            }
        } else {
            for point in points {
                if point.matched_last_frame {
                    continue;
                }
                let best_match = self.kdtree.nearest_one::<SquaredEuclidean>(&[point.x, point.y]);

                if best_match.distance < MAX_POINT_MATCH_DIST {
                    let matching = self.state.get_mut(best_match.item).unwrap();
                    let update_scale = 1.0;//(point.log_likelihood / matching.log_likelihood.max(1.0)).abs().max(0.25);

                    let diff_x = point.x - matching.x;
                    let update_x = diff_x * update_scale;
                    matching.x += update_x;

                    let diff_y = point.y - matching.y;
                    let update_y = diff_y * update_scale;
                    matching.y += update_y;

                    let diff_radius = point.radius - matching.radius;
                    let update_radius = diff_radius * update_scale;
                    matching.radius += update_radius;

                    let diff_amplitude = point.amplitude - matching.amplitude;
                    let update_amplitude = diff_amplitude * update_scale;
                    matching.amplitude += update_amplitude;

                    matching.log_likelihood += point.latest_score;
                    matching.matched_last_frame = true;
                } else {
                    self.state.push(point);
                }
            }

            for item in self.state.iter_mut() {
                item.age += 1;
                if !item.matched_last_frame {
                    item.log_likelihood -= UNMATCHED_POINT_PENALTY;
                }
            }
        }

        self.state = self.state.iter()
            .filter(|cand|cand.log_likelihood > POINT_DISCARD_THRESHOLD)
            .map(|x| x.clone())
            .collect();

        info!("state len: {:?}", self.state.len());
        self.state.sort_by_cached_key(|cand|OrderedFloat(cand.log_likelihood.neg()));

        let mut i = 0;
        while i < self.state.len() && self.state[i].log_likelihood > -5.0 {
            info!("{:?}", &self.state[i]);
            i += 1;
        }
    }

    pub fn preprocess_image(&self, img: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<GrayImage, Box<dyn Error>> {
        // TODO: have to perform expensive conversion from slice-based image to Vec-based image
        //       due to inflexibility of imageproc's median_filter and map_colors2 functions
        // let mut img_vec: Image<Luma<u8>> = Image::from_raw(img.width(), img.height(), img.to_vec()).unwrap();
        //
        // TODO: potentially eliminate this step by monte-carlo method
        //       of sampling a random 10-100 points inside a block,
        //       using the median of the points as the block median
        let img_median: GrayImage = median_filter(img, 30, 30);
        debug!("created median");
        img_median.save("img_median.png")?;

        // subtract median from original
        let subtracted = map_colors2(img, &img_median, |p, q| {
            Luma([(p[0] as u8).saturating_sub(q[0] as u8)])
        });
        debug!("subtracted median");

        subtracted.save("preprocessed.png")?;
        debug!("saved");

        Ok(subtracted)
    }

    pub fn process_image(&mut self, img: ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<(), Box<dyn Error>> {
        let subtracted = self.preprocess_image(&img)?;

        // reset matched point flag
        self.state = self.state.iter()
            .map(|x| {
                let mut x = x.clone();
                x.matched_last_frame = false;
                x
            })
            .collect();

        // Try to fit existing points first
        let fitted_existing_points: Vec<(usize, ImagePointCandidate)> = self.state
            .iter()
            .map(|candidate| self.fit_point(candidate, &subtracted))
            .enumerate()
            .filter(|(_, cand)|
                cand.x >= 0.0 && cand.y >= 0.0
                    && cand.x < img.width() as f64 && cand.y < img.height() as f64
                    && cand.radius < CANDIDATE_MAX_RADIUS && cand.radius > CANDIDATE_MIN_RADIUS
            )
            .collect();

        let mut count_refitted: usize = 0;
        let mut count_not_refitted: usize = 0;

        for (idx, refitted_point) in fitted_existing_points.iter() {
            if refitted_point.latest_score > MIN_EXISTING_MATCH_SCORE {
                self.state[*idx] = self.state[*idx].update_with(refitted_point);
                count_refitted += 1;
            } else {
                count_not_refitted += 1;
                self.state[*idx].latest_score = refitted_point.latest_score;
            }
        }
        info!(?count_refitted, ?count_not_refitted, "refit results");

        let point_extractor = PointDetectPeak {};

        let point_candidates: Vec<ImagePointCandidate> =
            point_extractor.extract_from_img(&subtracted, &self.state);
        info!(count_extracted = ?point_candidates.len(), "extracted");

        let points: Vec<ImagePointCandidate> = point_candidates
            .iter()
            .map(|candidate| self.fit_point(candidate, &subtracted))
            .filter(|cand|cand.x >= 0.0 && cand.y >= 0.0 && cand.x < img.width() as f64 && cand.y < img.height() as f64 && cand.radius < CANDIDATE_MAX_RADIUS && cand.radius > CANDIDATE_MIN_RADIUS)
            .collect();

        self.update_state(points);

        let img_query_annotated = self.annotate_image_query(&img.into(), &self.state);

        let _ = self.window.set_image("Frame", img_query_annotated);
        // img_query_annotated.save("query-annotated.png")?;

        Ok(())
    }

    pub fn fit_point(&self, point: &ImagePointCandidate, img: &GrayImage) -> ImagePointCandidate {
        let fitter = PointFitterGaussianNelderMead {};

        let result = fitter.fit_point(point, img);

        result
    }

    #[instrument(skip_all)]
    pub fn consume_frames_to_extracted_point_stream(&mut self, mut cons: Consumer<Arc<(isize, VideoFrame<Readable>)>>) {
        let mut idx = 0;
        let current_frame_index: isize = -1;

        loop {
            let mut processed_frame = false;
            sleep(Duration::from_millis(1));
            if let Ok(arc_frame_ref) = cons.peek() {
                let idx_and_frame = arc_frame_ref.clone();
                if idx_and_frame.0 > current_frame_index {
                    debug!("Consuming a frame");

                    let img_buf = self.video_frame_to_img_buf_cloned(&idx_and_frame.1);

                    let result = self.process_image(img_buf).unwrap();
                    debug!(?result, ?idx, "processed frame");
                    idx += 1;
                    processed_frame = true;
                }
            }

            // TODO: this will need to be more sophisticated when we have multiple consumers on the ring buffer
            if processed_frame {
                let _ = cons.pop();
            }
        }
    }

    #[allow(dead_code)]
    fn video_frame_to_img_buf<'a, P: Pixel>(&self, frame: &'a VideoFrame<Readable>) -> ImageBuffer<P, &'a [u8]>
        where for<'b> &'b [u8]: Deref<Target=[P::Subpixel]>,
              [P::Subpixel]: EncodableLayout,
              P: PixelWithColorType,
    {
        ImageBuffer::<P, &[u8]>::from_raw(
            frame.width(), frame.height(), frame.plane_data(0).unwrap()
        ).expect("Could not create ImageBuffer from VideoFrame")
    }

    fn video_frame_to_img_buf_cloned(&self, frame: &VideoFrame<Readable>) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let buf_cloned: Vec<u8> = frame.plane_data(0).unwrap().into();
        ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(
            frame.width(), frame.height(), buf_cloned
        ).expect("Could not create ImageBuffer from VideoFrame")
    }

    pub fn annotate_image_query(&self, img: &DynamicImage, query: &Vec<ImagePointCandidate>) -> DynamicImage {
        let blue = Rgba([0u8, 0u8, 255u8, 255u8]);
        let green = Rgba([0u8, 255u8, 0u8, 255u8]);
        let red = Rgba([255u8, 0u8, 0u8, 255u8]);

        let font = Vec::from(include_bytes!("/System/Library/Fonts/Monaco.ttf") as &[u8]);
        let font = Font::try_from_vec(font).unwrap();

        let height = 18f32;
        let scale = rusttype::Scale {
            x: height,
            y: height,
        };

        let mut new_img: DynamicImage = img.clone().into_rgba8().into();

        for (_idx, &point) in query.iter().enumerate() {
            // if point.log_likelihood < 5 && point.age < 5 {
            //     continue;
            // }

            let draw_color = if point.log_likelihood < 10.0 {
                red
            } else {
                if point.age < 10 {
                    blue
                } else {
                    green
                }
            };

            let centre = (point.x as i32, point.y as i32);
            draw_hollow_circle_mut(&mut new_img, centre, point.radius as i32 * 4, draw_color);

            let num = NumberFormat::new();
            let label = format!("${} LL{} ∅{} ↑{}", num.format(".2s", point.latest_score), num.format(".2s", point.log_likelihood), num.format(".2s", point.radius), num.format(".2s", point.amplitude));

            draw_text_mut(
                &mut new_img,
                draw_color,
                (point.x as u32).saturating_sub(5u32) as i32,
                (point.y as u32).saturating_sub(30u32) as i32,
                scale,
                &font,
                label.as_str(),
            );
        }

        new_img
    }
}
