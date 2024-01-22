use argmin::core::ArgminFloat;
use astrocap_model::state::{FittedPoint, ModelState};
use az::{Az, Cast};
use format_num::NumberFormat;
use image::{DynamicImage, ImageBuffer, Luma, Pixel, RgbImage, Rgba};
use imageproc::drawing::{draw_hollow_circle_mut, draw_text_mut};
use kiddo::float::kdtree::Axis;
use rusttype::{Font, Scale};
use show_image::{WindowOptions, WindowProxy};
use std::error::Error;
use std::iter::Sum;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

pub struct WindowRenderer<'a, F: Axis + ArgminFloat + Sum> {
    state: Arc<Mutex<ModelState<F>>>,
    font: Font<'a>,
    cyan: Rgba<u8>,
    red: Rgba<u8>,
    green: Rgba<u8>,
    scale: Scale,
    window: WindowProxy,
    rx: Receiver<ImageBuffer<Luma<u8>, Arc<[u8]>>>,
}

pub struct StarCandidateAnnotation<F: Axis + ArgminFloat + Sum> {
    x: F,
    y: F,
    radius: F,
    amplitude: F,
    score: F,
    log_likelihood: F,
    age: usize,
}

impl<'a, F: Axis + ArgminFloat + Sum + Cast<u32> + Cast<f32> + Cast<i32>> WindowRenderer<'a, F>
where
    f64: Cast<F>,
    f32: Cast<F>,
{
    pub fn new(
        state: Arc<Mutex<ModelState<F>>>,
        rx: Receiver<ImageBuffer<Luma<u8>, Arc<[u8]>>>,
    ) -> Self {
        let window = show_image::create_window(
            "Annotated Frames",
            WindowOptions::default().set_size([1920, 1080]),
        )
        .expect("Could not create window");

        #[cfg(target_os = "macos")]
        let font = Vec::from(include_bytes!("/System/Library/Fonts/Monaco.ttf") as &[u8]);

        #[cfg(not(target_os = "macos"))]
        let font = Vec::from(include_bytes!(
            "/home/scotty/.fonts/f/Fira_Code_Regular_Nerd_Font_Complete.otf"
        ) as &[u8]);

        let font = Font::try_from_vec(font).unwrap();

        let cyan = Rgba([0u8, 255u8, 255u8, 255u8]);
        let green = Rgba([0u8, 255u8, 0u8, 255u8]);
        let red = Rgba([255u8, 0u8, 0u8, 255u8]);

        let height = 18f32;
        let scale = Scale {
            x: height,
            y: height,
        };

        WindowRenderer {
            state,
            font,
            scale,
            cyan,
            green,
            red,
            window,
            rx,
        }
    }

    pub fn run(&mut self) {
        while let Ok(img) = self.rx.recv() {
            let _ = self.process_frame(img);
        }
    }

    pub fn process_frame(
        &mut self,
        img: ImageBuffer<Luma<u8>, Arc<[u8]>>,
    ) -> Result<(), Box<dyn Error>> {
        let annotations = self.get_annotation_data();

        let img_query_annotated = self.annotate_image_query(&img, annotations);

        self.window.set_image("Frame", img_query_annotated)?;
        // img_query_annotated.save("query-annotated.png")?;

        Ok(())
    }

    pub fn get_annotation_data(&self) -> Vec<StarCandidateAnnotation<F>> {
        let state_ref = &self.state.lock().unwrap();

        let mut annotations = vec![];

        state_ref
            .star_candidates
            .iter()
            .filter(|cand| cand.log_likelihood >= 10.0f32.az::<F>() && cand.age >= 5)
            .for_each(|cand| {
                let mut fitted_point: Option<&FittedPoint<F>> = None;
                for (frame_idx, fitted_point_match) in
                    cand.detected_point_match_history.iter().rev().enumerate()
                {
                    if let Some(detected_point_idx) = fitted_point_match {
                        if let Some(frame_state) =
                            state_ref.recent_frame_states.iter().rev().nth(frame_idx)
                        {
                            let detected_point =
                                &frame_state.detected_points_list[detected_point_idx.get()];

                            fitted_point = Some(detected_point.fitted_point.as_ref().unwrap());
                        }
                    }
                }

                if let Some(fitted_point) = fitted_point {
                    annotations.push(StarCandidateAnnotation {
                        x: fitted_point.x,
                        y: fitted_point.y,
                        radius: fitted_point.radius,
                        amplitude: fitted_point.amplitude,
                        score: fitted_point.score,
                        log_likelihood: cand.log_likelihood,
                        age: cand.age,
                    });
                }
            });

        annotations
    }

    pub fn annotate_image_query(
        &self,
        img: &ImageBuffer<Luma<u8>, Arc<[u8]>>,
        annotations: Vec<StarCandidateAnnotation<F>>,
    ) -> DynamicImage {
        let mut new_img = RgbImage::new(img.width(), img.height());
        img.pixels()
            .zip(new_img.pixels_mut())
            .for_each(|(from, to)| {
                to.channels_mut()[0] = from.channels()[0];
                to.channels_mut()[1] = from.channels()[0];
                to.channels_mut()[2] = from.channels()[0];
            });

        let mut new_img: DynamicImage = DynamicImage::ImageRgb8(new_img);

        for annotation in annotations {
            if annotation.log_likelihood < 10.0f64.az::<F>() || annotation.age < 5 {
                continue;
            }

            let draw_color = if annotation.log_likelihood < 30.0f32.az::<F>() {
                &self.red
            } else if annotation.age < 30 {
                &self.cyan
            } else {
                &self.green
            };

            let centre = (annotation.x.az::<i32>(), annotation.y.az::<i32>());
            draw_hollow_circle_mut(
                &mut new_img,
                centre,
                annotation.radius.az::<i32>() * 4,
                *draw_color,
            );

            let num = NumberFormat::new();
            let label = format!(
                "${} LL{} R{} A{}",
                num.format(".2s", annotation.score.az::<f32>()),
                num.format(".2s", annotation.log_likelihood.az::<f32>()),
                num.format(".2s", annotation.radius.az::<f32>()),
                num.format(".2s", annotation.amplitude.az::<f32>())
            );

            draw_text_mut(
                &mut new_img,
                *draw_color,
                (annotation.x.az::<u32>()).saturating_sub(5u32) as i32,
                (annotation.y.az::<u32>()).saturating_sub(30u32) as i32,
                self.scale,
                &self.font,
                label.as_str(),
            );
        }

        new_img
    }
}
