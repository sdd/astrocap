use format_num::NumberFormat;
use gst_video::video_frame::Readable;
use gst_video::VideoFrame;
use image::io::Reader as ImageReader;
use image::{
    DynamicImage, EncodableLayout, GrayImage, ImageBuffer, Luma, Pixel, PixelWithColorType,
    RgbImage, Rgba,
};
use imageproc::drawing::{draw_hollow_circle_mut, draw_text_mut};
use itertools::Itertools;
use rtrb::Consumer;
use rusttype::Font;
use std::error::Error;
use std::ops::Deref;
use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;
use tracing::{debug, info, instrument};

use crate::cli::Args;
use astrocap_model::config::ModelConfig;
use show_image::{WindowOptions, WindowProxy};

use astrocap_model::point_detect_peak::PointDetectPeak;
use astrocap_model::point_fitter_nelder_mead::PointFitterGaussianNelderMead;
use astrocap_model::state::ModelState;

use crate::image_luma_extractor::Img;
use crate::map_colors_2::map_colors2;
use crate::median_filter::median_filter;
use crate::video_frame_to_image_buffer::VideoFrameExt;

// const ANNOTATED_IMG_MARKER_SRC_RADIUS: i32 = 10;
const MAX_POINT_MATCH_DIST: f64 = 7.5;
const MIN_EXISTING_MATCH_SCORE: f64 = -5.0;
const CANDIDATE_MIN_SCORE: f64 = 0.0;

const UNMATCHED_POINT_PENALTY: f64 = 5.0; //0.5;
                                          // const MATCHED_POINT_BENEFIT: f64 = 5.0;
const POINT_DISCARD_THRESHOLD: f64 = -10.0;
// const STARTING_LOG_LIKELIHOOD: i64 = 3;

const AMPLITUDE_PENALTY_THRESHOLD: f64 = 8.0;
const AMPLITUDE_PENALTY: f64 = 15.0;

pub struct PointExtractorConsumer {
    state: ModelState<f32>,
    window: WindowProxy,
    mask: Option<Arc<ImageBuffer<Luma<u8>, Vec<u8>>>>,
}

impl PointExtractorConsumer {
    pub fn new(args: Arc<Args>) -> Self {
        let window = show_image::create_window(
            "Annotated Frames",
            WindowOptions::default().set_size([1920, 1080]),
        )
        .expect("Could not create window");

        let mask = args.mask.clone().and_then(|mask_file_path| {
            let image = ImageReader::open(mask_file_path).unwrap();
            info!("loaded mask image");
            let decoded_image = image.decode().unwrap();
            info!("decoded mask image");
            let luma8_image = decoded_image.to_luma8();
            info!("converted mask image to luma8");

            Some(Arc::new(luma8_image))
        });

        PointExtractorConsumer {
            state: ModelState::new(ModelConfig::default()),
            window,
            mask,
        }
    }

    pub fn preprocess_frame(
        &self,
        img: &ImageBuffer<Luma<u8>, Arc<[u8]>>,
    ) -> Result<GrayImage, Box<dyn Error>> {
        // TODO: consider reusing the buffer for median
        //       (and poss subtacted)
        //       to save from re-allocating each time

        // TODO: potentially eliminate this step by monte-carlo method
        //       of sampling a random 10-100 points inside a block,
        //       using the median of the points as the block median
        let img_median: GrayImage = median_filter(img, 30, 30);
        debug!("created median");
        // img_median.save("../../img_median.png")?;

        // TODO: consider refactoring to subbing median from
        //       original in same step as creating median

        // subtract median from original
        let subtracted = map_colors2(img, &img_median, |p, q| {
            Luma([(p[0] as u8).saturating_sub(q[0] as u8)])
        });
        debug!("subtracted median");

        // subtracted.save("../../preprocessed.png")?;
        //debug!("saved");

        Ok(subtracted)
    }

    pub fn process_frame(
        &mut self,
        img: ImageBuffer<Luma<u8>, Arc<[u8]>>,
    ) -> Result<(), Box<dyn Error>> {
        let subtracted = self.preprocess_frame(&img)?;
        let arc_img = Arc::new(Img(subtracted));

        self.state
            .process_frame::<PointDetectPeak, PointFitterGaussianNelderMead>(arc_img)?;

        let img_query_annotated = self.annotate_image_query(&img);

        let _ = self.window.set_image("Frame", img_query_annotated);
        // img_query_annotated.save("query-annotated.png")?;

        Ok(())
    }

    #[instrument(skip_all)]
    pub fn consume_frames_to_extracted_point_stream(
        &mut self,
        mut cons: Consumer<Arc<(isize, VideoFrame<Readable>)>>,
    ) {
        let mut idx = 0;
        let current_frame_index: isize = -1;

        loop {
            let mut processed_frame = false;
            sleep(Duration::from_millis(1));
            if let Ok(arc_frame_ref) = cons.peek() {
                let idx_and_frame = arc_frame_ref.clone();
                if idx_and_frame.0 > current_frame_index {
                    debug!("Consuming a frame");

                    // let img_buf = self.video_frame_to_img_buf_cloned(&idx_and_frame.1);
                    let img_buf = VideoFrameExt(&idx_and_frame.1).as_img_buf_arc();

                    let result = self.process_frame(img_buf).unwrap();
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
    fn video_frame_to_img_buf<'a, P: Pixel>(
        &self,
        frame: &'a VideoFrame<Readable>,
    ) -> ImageBuffer<P, &'a [u8]>
    where
        for<'b> &'b [u8]: Deref<Target = [P::Subpixel]>,
        [P::Subpixel]: EncodableLayout,
        P: PixelWithColorType,
    {
        ImageBuffer::<P, &[u8]>::from_raw(
            frame.width(),
            frame.height(),
            frame.plane_data(0).unwrap(),
        )
        .expect("Could not create ImageBuffer from VideoFrame")
    }

    fn video_frame_to_img_buf_cloned(
        &self,
        frame: &VideoFrame<Readable>,
    ) -> ImageBuffer<Luma<u8>, Vec<u8>> {
        let buf_cloned: Vec<u8> = frame.plane_data(0).unwrap().into();
        ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(frame.width(), frame.height(), buf_cloned)
            .expect("Could not create ImageBuffer from VideoFrame")
    }

    pub fn annotate_image_query(&self, img: &ImageBuffer<Luma<u8>, Arc<[u8]>>) -> DynamicImage {
        let cyan = Rgba([0u8, 255u8, 255u8, 255u8]);
        let green = Rgba([0u8, 255u8, 0u8, 255u8]);
        let red = Rgba([255u8, 0u8, 0u8, 255u8]);

        #[cfg(target_os = "macos")]
        let font = Vec::from(include_bytes!("/System/Library/Fonts/Monaco.ttf") as &[u8]);

        #[cfg(not(target_os = "macos"))]
        let font = Vec::from(include_bytes!(
            "/home/scotty/.fonts/f/Fira_Code_Regular_Nerd_Font_Complete.otf"
        ) as &[u8]);

        let font = Font::try_from_vec(font).unwrap();

        let height = 18f32;
        let scale = rusttype::Scale {
            x: height,
            y: height,
        };

        let mut new_img = RgbImage::new(img.width(), img.height());
        img.pixels()
            .zip(new_img.pixels_mut())
            .for_each(|(from, to)| {
                to.channels_mut()[0] = from.channels()[0];
                to.channels_mut()[1] = from.channels()[0];
                to.channels_mut()[2] = from.channels()[0];
            });

        let mut new_img: DynamicImage = DynamicImage::ImageRgb8(new_img);

        for (_idx, &ref candidate) in self.state.star_candidates.iter().enumerate() {
            if candidate.log_likelihood < 10.0 || candidate.age < 5 {
                continue;
            }

            let draw_color = if candidate.log_likelihood < 30.0 {
                red
            } else {
                if candidate.age < 30 {
                    cyan
                } else {
                    green
                }
            };

            for (frame_idx, fitted_point_match) in candidate
                .fitted_point_match_history
                .iter()
                .rev()
                .enumerate()
            {
                if let Some(detected_point_idx) = fitted_point_match {
                    if let Some(&ref frame_state) = &self
                        .state
                        .recent_frame_states
                        .iter()
                        .rev()
                        .skip(frame_idx)
                        .next()
                    {
                        let fitted_point =
                            &frame_state.fitted_points_list[detected_point_idx.get()];

                        let centre = (fitted_point.x as i32, fitted_point.y as i32);
                        draw_hollow_circle_mut(
                            &mut new_img,
                            centre,
                            fitted_point.radius as i32 * 4,
                            draw_color,
                        );

                        let num = NumberFormat::new();
                        let label = format!(
                            "${} LL{} R{} A{}",
                            num.format(".2s", fitted_point.score),
                            num.format(".2s", candidate.log_likelihood),
                            num.format(".2s", fitted_point.radius),
                            num.format(".2s", fitted_point.amplitude)
                        );

                        draw_text_mut(
                            &mut new_img,
                            draw_color,
                            (fitted_point.x as u32).saturating_sub(5u32) as i32,
                            (fitted_point.y as u32).saturating_sub(30u32) as i32,
                            scale,
                            &font,
                            label.as_str(),
                        );

                        break;
                    }
                }
            }
        }

        new_img
    }
}
