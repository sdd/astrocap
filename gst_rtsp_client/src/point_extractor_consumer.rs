use gst_video::video_frame::Readable;
use gst_video::VideoFrame;
use image::io::Reader as ImageReader;
use image::{GrayImage, ImageBuffer, Luma};

use rtrb::Consumer;

use rerun::RecordingStream;
use std::error::Error;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use tracing::{debug, info, instrument};

use crate::cli::Args;
use astrocap_model::config::ModelConfig;
use astrocap_model::point_detect_adaptive_centroid::PointDetectAdaptiveCentroid;
use astrocap_model::point_detect_local_maxima::PointDetectLocalMaxima;
use astrocap_model::point_detect_peak::PointDetectPeak;
use astrocap_model::point_fitter_nelder_mead::PointFitterGaussianNelderMead;
use astrocap_model::state::ModelState;
use astrocap_model::traits::ImageLumaExtractor;

use crate::image_luma_extractor::Img;
use crate::map_colors_2::map_colors2;
use crate::median_filter::median_filter;
use crate::video_frame_to_image_buffer::VideoFrameExt;

pub struct PointExtractorConsumer {
    state: Arc<Mutex<ModelState<f32>>>,
    mask: Option<Arc<dyn ImageLumaExtractor>>,
    tx: Sender<(usize, ImageBuffer<Luma<u8>, Arc<[u8]>>)>,
    rec: RecordingStream,
    processed_frame_buffer: Vec<GrayImage>,
    median: Option<Arc<dyn ImageLumaExtractor>>,
    current_frame_index: usize,
    integration_frames: usize, // Will be 2^integration_power
    integration_power: u32,
    frames_written: usize,
    cached_integration_sum: Option<Vec<u16>>,
}

impl PointExtractorConsumer {
    pub fn new(args: Arc<Args>, rec: RecordingStream) -> Self {
        let mask = Self::create_mask(&args);

        let state = ModelState::new(
            ModelConfig::default(),
            args.star_index_path.clone(),
            rec.clone(),
        );
        let state = Arc::new(Mutex::new(state));

        let (tx, rx) = channel::<(usize, ImageBuffer<Luma<u8>, Arc<[u8]>>)>();

        let rec_cloned = rec.clone();
        thread::spawn(move || {
            while let Ok((idx, frame)) = rx.recv() {
                rec_cloned
                    .log(
                        "video/original",
                        &rerun::Image::from_pixel_format(
                            [1920, 1080],
                            rerun::PixelFormat::Y8_FullRange,
                            frame.as_ref(),
                        ),
                    )
                    .unwrap();
            }
        });

        // Get integration power from args, default to 2 (4 frames)
        let integration_power = 3u32; //args.integration_power.unwrap_or(2);
        let integration_frames = 1usize << integration_power; // 2^integration_power

        info!(
            "Frame integration: 2^{} = {} frames",
            integration_power, integration_frames
        );

        // Pre-allocate ring buffer with dummy frames - we'll overwrite them
        let dummy_frame = GrayImage::new(1920, 1080); // Adjust dimensions as needed
        let processed_frame_buffer = vec![dummy_frame; integration_frames];

        PointExtractorConsumer {
            state,
            mask,
            tx,
            rec,
            processed_frame_buffer,
            current_frame_index: 0,
            integration_frames,
            integration_power,
            frames_written: 0,
            cached_integration_sum: None,
            median: None,
        }
    }

    fn create_mask(args: &Arc<Args>) -> Option<Arc<dyn ImageLumaExtractor>> {
        let mask = args.mask.clone().map(|mask_file_path| {
            let image = ImageReader::open(mask_file_path).unwrap();
            info!("loaded mask image");
            let decoded_image = image.decode().unwrap();
            info!("decoded mask image");
            let luma8_image = decoded_image.to_luma8();
            info!("converted mask image to luma8");

            // See: https://stackoverflow.com/a/67470122/642703
            let arc = Arc::new(Img(luma8_image));
            let arc_cast: Arc<dyn ImageLumaExtractor> = Arc::<Img>::clone(&arc);
            arc_cast
        });
        mask
    }

    pub fn preprocess_frame(
        &mut self,
        img: ImageBuffer<Luma<u8>, Arc<[u8]>>,
    ) -> Result<GrayImage, Box<dyn Error>> {
        // TODO: consider reusing the buffer for median
        //       (and poss subtracted)
        //       to save from re-allocating each time

        // TODO: potentially eliminate this step by monte-carlo method
        //       of sampling a random 10-100 points inside a block,
        //       using the median of the points as the block median
        let img_median: GrayImage = median_filter(&img, 30, 30);
        debug!("created median");
        self.rec
            .log(
                "video/median",
                &rerun::Image::from_pixel_format(
                    [1920, 1080],
                    rerun::PixelFormat::Y8_FullRange,
                    img_median.as_ref(),
                ),
            )
            .unwrap();

        // TODO: consider refactoring to subbing median from
        //       original in same step as creating median

        // subtract median from original
        let subtracted = map_colors2(&img, &img_median, |p, q| {
            Luma([(p[0]).saturating_sub(q[0])])
        });
        debug!("subtracted median");
        self.rec
            .log(
                "video/subbed",
                &rerun::Image::from_pixel_format(
                    [1920, 1080],
                    rerun::PixelFormat::Y8_FullRange,
                    subtracted.as_ref(),
                ),
            )
            .unwrap();

        self.median = Some(Arc::new(Img(img_median)));

        Ok(subtracted)
    }

    pub fn process_frame(
        &mut self,
        img: ImageBuffer<Luma<u8>, Arc<[u8]>>,
        frame_index: usize,
    ) -> Result<(), Box<dyn Error>> {
        // First, do the median subtraction on this frame
        let median_subtracted = self.preprocess_frame(img.clone())?;

        // Store the processed frame in the ring buffer
        let buffer_index = self.current_frame_index & (self.integration_frames - 1);

        // Clone the old frame before we replace it (only if we need it)
        let old_frame = if self.frames_written >= self.integration_frames {
            Some(self.processed_frame_buffer[buffer_index].clone())
        } else {
            None
        };

        // Replace with new frame
        self.processed_frame_buffer[buffer_index] = median_subtracted.clone();

        // Update cached sum incrementally
        self.update_cached_integration_sum(old_frame.as_ref(), &median_subtracted);

        // Increment counters
        self.current_frame_index += 1;
        self.frames_written += 1;

        // Only do star detection when we have enough frames to integrate
        if self.frames_written >= self.integration_frames {
            let integrated_frame = self.get_integrated_frame();

            self.rec
                .log(
                    "video/integrated",
                    &rerun::Image::from_pixel_format(
                        [1920, 1080],
                        rerun::PixelFormat::Y8_FullRange,
                        integrated_frame.as_ref(),
                    ),
                )
                .unwrap();

            let arc_img_integrated = Arc::new(Img(integrated_frame));

            {
                self.state
                    .lock()
                    .unwrap()
                    .process_frame::<PointDetectAdaptiveCentroid, PointFitterGaussianNelderMead>(
                        arc_img_integrated,
                        self.median.clone(),
                        self.mask.clone(),
                        self.rec.clone(),
                    )?;
            }
        }

        self.tx.send((frame_index, img.clone()))?;
        Ok(())
    }

    fn update_cached_integration_sum(
        &mut self,
        old_frame: Option<&GrayImage>,
        new_frame: &GrayImage,
    ) {
        let (width, height) = new_frame.dimensions();
        let pixel_count = (width * height) as usize;

        // Initialize cache if needed
        if self.cached_integration_sum.is_none() {
            self.cached_integration_sum = Some(vec![0u16; pixel_count]);
        }

        let sum = self.cached_integration_sum.as_mut().unwrap();

        if let Some(old_frame) = old_frame {
            // Buffer full - subtract old, add new
            for (i, (&old_pixel, &new_pixel)) in old_frame.iter().zip(new_frame.iter()).enumerate()
            {
                sum[i] = sum[i]
                    .saturating_sub(old_pixel as u16)
                    .saturating_add(new_pixel as u16);
            }
        } else {
            // Still filling buffer - just add new frame
            for (i, &pixel) in new_frame.iter().enumerate() {
                sum[i] += pixel as u16;
            }
        }
    }

    fn get_integrated_frame(&self) -> GrayImage {
        let first_frame = &self.processed_frame_buffer[0];
        let (width, height) = first_frame.dimensions();

        let sum = self.cached_integration_sum.as_ref().unwrap();
        let scale_factor = (self.integration_frames as f32).sqrt();

        let result: Vec<u8> = sum
            .iter()
            .map(|&sum_val| ((sum_val as f32 / scale_factor).round() as u16).min(255) as u8)
            .collect();

        debug!(
            "Retrieved cached integrated frame with √N scaling (factor: {:.2})",
            scale_factor
        );

        GrayImage::from_raw(width, height, result).unwrap()
    }

    fn integrate_processed_frames(&self) -> GrayImage {
        let first_frame = &self.processed_frame_buffer[0];
        let (width, height) = first_frame.dimensions();

        // Use u16 for intermediate sums
        let mut integrated = vec![0u16; (width * height) as usize];

        // Sum all frames
        for frame in &self.processed_frame_buffer {
            for (i, &pixel) in frame.iter().enumerate() {
                integrated[i] += pixel as u16;
            }
        }

        // For astrometry/photometry, we want to maximize SNR while preserving
        // relative brightness relationships. Scale by √N maintains noise
        // characteristics while boosting signal.
        let scale_factor = (self.integration_frames as f32).sqrt();

        let result: Vec<u8> = integrated
            .into_iter()
            .map(|sum| {
                // Round to preserve quantization for photometry
                ((sum as f32 / scale_factor).round() as u16).min(255) as u8
            })
            .collect();

        debug!(
            "Integrated {} frames (2^{}) for astrometry/photometry - SNR boost: {:.2}x",
            self.integration_frames, self.integration_power, scale_factor
        );

        GrayImage::from_raw(width, height, result).unwrap()
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

                    let img_buf = VideoFrameExt(&idx_and_frame.1).as_img_buf_arc();

                    self.process_frame(img_buf, idx).unwrap();

                    debug!(?idx, "processed frame");
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
}
