use gst_video::video_frame::Readable;
use gst_video::VideoFrame;
use image::io::Reader as ImageReader;
use image::{GrayImage, ImageBuffer, Luma};

use rtrb::Consumer;

use std::error::Error;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::sleep;
use std::time::Duration;
use tracing::{debug, info, instrument};

use crate::cli::Args;
use astrocap_model::config::ModelConfig;

use astrocap_model::point_detect_peak::PointDetectPeak;
use astrocap_model::point_fitter_nelder_mead::PointFitterGaussianNelderMead;
use astrocap_model::state::ModelState;

use crate::image_luma_extractor::Img;
use crate::map_colors_2::map_colors2;
use crate::median_filter::median_filter;
use crate::video_frame_to_image_buffer::VideoFrameExt;
use crate::window_renderer::WindowRenderer;

// const ANNOTATED_IMG_MARKER_SRC_RADIUS: i32 = 10;

pub struct PointExtractorConsumer {
    state: Arc<Mutex<ModelState<f32>>>,
    mask: Option<Arc<Img>>,
    tx: Sender<ImageBuffer<Luma<u8>, Arc<[u8]>>>,
}

impl PointExtractorConsumer {
    pub fn new(args: Arc<Args>) -> Self {
        let mask = args.mask.clone().map(|mask_file_path| {
            let image = ImageReader::open(mask_file_path).unwrap();
            info!("loaded mask image");
            let decoded_image = image.decode().unwrap();
            info!("decoded mask image");
            let luma8_image = decoded_image.to_luma8();
            info!("converted mask image to luma8");

            Arc::new(Img(luma8_image))
        });

        let state = ModelState::new(ModelConfig::default());
        let state = Arc::new(Mutex::new(state));
        let state_cloned = state.clone();

        let (tx, rx) = channel();

        thread::spawn(move || {
            WindowRenderer::<f32>::new(state_cloned, rx).run();
        });

        PointExtractorConsumer {
            state: state.clone(),
            mask,
            tx,
        }
    }

    pub fn preprocess_frame(
        &self,
        img: ImageBuffer<Luma<u8>, Arc<[u8]>>,
    ) -> Result<GrayImage, Box<dyn Error>> {
        // TODO: consider reusing the buffer for median
        //       (and poss subtacted)
        //       to save from re-allocating each time

        // TODO: potentially eliminate this step by monte-carlo method
        //       of sampling a random 10-100 points inside a block,
        //       using the median of the points as the block median
        let img_median: GrayImage = median_filter(&img, 30, 30);
        debug!("created median");
        // img_median.save("../../img_median.png")?;

        // TODO: consider refactoring to subbing median from
        //       original in same step as creating median

        // subtract median from original
        let subtracted = map_colors2(&img, &img_median, |p, q| {
            Luma([(p[0]).saturating_sub(q[0])])
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
        let subtracted = self.preprocess_frame(img.clone())?;

        let arc_img_subtracted = Arc::new(Img(subtracted));
        let arc_mask = self.mask.clone().unwrap();

        {
            self.state
                .lock()
                .unwrap()
                .process_frame::<PointDetectPeak, PointFitterGaussianNelderMead>(
                    arc_img_subtracted,
                    arc_mask,
                )?;
        }

        self.tx.send(img.clone())?;

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

                    let img_buf = VideoFrameExt(&idx_and_frame.1).as_img_buf_arc();

                    self.process_frame(img_buf).unwrap();

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
