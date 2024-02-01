use argmin::core::ArgminFloat;
use astrocap_model::state::{FittedPoint, ModelState};
use astrocap_model::traits::AstroFloat;
use az::{Az, Cast};
use format_num::NumberFormat;
use image::{DynamicImage, ImageBuffer, Luma, Pixel, RgbImage, Rgba};
use imageproc::drawing::{draw_hollow_circle_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::rect::Rect;
use kiddo::float::kdtree::Axis;
use nalgebra::Vector2;
use rusttype::{Font, Scale};
use show_image::{WindowOptions, WindowProxy};
use std::error::Error;
use std::iter::Sum;
use std::ops::{DivAssign, MulAssign, SubAssign};
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Mutex};

pub struct WindowRenderer<'a, F: AstroFloat> {
    state: Arc<Mutex<ModelState<F>>>,
    font: Font<'a>,
    cyan: Rgba<u8>,
    red: Rgba<u8>,
    green: Rgba<u8>,
    yellow: Rgba<u8>,
    orange: Rgba<u8>,
    scale: Scale,
    big_scale: Scale,
    window: WindowProxy,
    rx: Receiver<(usize, ImageBuffer<Luma<u8>, Arc<[u8]>>)>,
}

#[derive(Debug)]
pub struct StarCandidateAnnotation<F: AstroFloat> {
    x: F,
    y: F,
    radius: F,
    amplitude: F,
    score: F,
    log_likelihood: F,
    age: usize,
    name: Option<String>,
}

#[derive(Debug)]
pub struct MovingTargetAnnotation<F: AstroFloat> {
    score: F,
    log_likelihood: F,
    age: usize,
    position: Vector2<F>,
    velocity: Vector2<F>,
    pos_history: Vec<Vector2<F>>,
}

pub struct FrameAnnotations<F: AstroFloat> {
    star_candidates: Vec<StarCandidateAnnotation<F>>,
    moving_targets: Vec<MovingTargetAnnotation<F>>,
}

impl<'a, F: AstroFloat> WindowRenderer<'a, F>
where
    f64: Cast<F>,
    f32: Cast<F>,
    F: Cast<u32>,
    F: Cast<f32>,
    F: Cast<i32>,
{
    pub fn new(
        state: Arc<Mutex<ModelState<F>>>,
        rx: Receiver<(usize, ImageBuffer<Luma<u8>, Arc<[u8]>>)>,
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
        let yellow = Rgba([255u8, 255u8, 0u8, 255u8]);
        let orange = Rgba([255u8, 127u8, 0u8, 255u8]);

        let height = 18f32;
        let scale = Scale {
            x: height,
            y: height,
        };

        let big_scale = Scale {
            x: height * 1.5,
            y: height * 1.5,
        };

        WindowRenderer {
            state,
            font,
            scale,
            big_scale,
            cyan,
            green,
            red,
            yellow,
            orange,
            window,
            rx,
        }
    }

    pub fn run(&mut self) {
        while let Ok((frame_index, img)) = self.rx.recv() {
            let _ = self.process_frame(frame_index, img);
        }
    }

    pub fn process_frame(
        &mut self,
        frame_index: usize,
        img: ImageBuffer<Luma<u8>, Arc<[u8]>>,
    ) -> Result<(), Box<dyn Error>> {
        let annotations = self.get_annotation_data();

        let img_query_annotated = self.annotate_image_query(&img, annotations);

        self.window.set_image(
            format!("query-annotated-frame-{}.png", frame_index),
            img_query_annotated,
        )?;
        // img_query_annotated.save("query-annotated.png")?;

        Ok(())
    }

    pub fn get_annotation_data(&self) -> FrameAnnotations<F> {
        let state_ref = &self.state.lock().unwrap();

        let mut sc_annotations = vec![];

        state_ref
            .star_candidates
            .iter()
            // .filter(|cand| cand.log_likelihood >= 10.0f32.az::<F>() && cand.age >= 5)
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
                            break;
                        }
                    }
                }

                if let Some(fitted_point) = fitted_point {
                    sc_annotations.push(StarCandidateAnnotation {
                        x: fitted_point.x,
                        y: fitted_point.y,
                        radius: fitted_point.radius,
                        amplitude: fitted_point.amplitude,
                        score: fitted_point.score,
                        log_likelihood: cand.log_likelihood,
                        age: cand.age,
                        name: cand.match_name.clone(),
                    });
                } else {
                    sc_annotations.push(StarCandidateAnnotation {
                        x: cand.x,
                        y: cand.y,
                        radius: F::zero(),
                        amplitude: F::zero(),
                        score: F::zero(),
                        log_likelihood: cand.log_likelihood,
                        age: cand.age,
                        name: cand.match_name.clone(),
                    });
                }
            });

        let mt_annotations = state_ref
            .moving_targets
            .iter()
            .map(|mt| MovingTargetAnnotation {
                position: mt.position,
                velocity: mt.velocity,
                score: mt.latest_score,
                log_likelihood: mt.log_odds,
                age: mt.age,
                pos_history: mt.pos_history.clone(),
            })
            .collect();

        FrameAnnotations {
            star_candidates: sc_annotations,
            moving_targets: mt_annotations,
        }
    }

    pub fn annotate_image_query(
        &self,
        img: &ImageBuffer<Luma<u8>, Arc<[u8]>>,
        annotations: FrameAnnotations<F>,
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

        for (idx, annotation) in annotations.star_candidates.iter().enumerate() {
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

            let name = annotation.name.clone().unwrap_or("".to_string());

            let label = format!(
                "{} #{} ${} LL{} R{} A{}",
                name,
                idx,
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

        for (idx, annotation) in annotations.moving_targets.iter().enumerate() {
            let draw_color = if annotation.log_likelihood < 30.0f32.az::<F>() {
                &self.orange
            } else {
                &self.yellow
            };

            let centre = (
                annotation.position[0].az::<i32>(),
                annotation.position[1].az::<i32>(),
            );
            draw_hollow_rect_mut(
                &mut new_img,
                Rect::at(centre.0 - 10, centre.1 - 10).of_size(20, 20),
                *draw_color,
            );

            let num = NumberFormat::new();
            let label = format!(
                "#{} ${} LL{} D{},{}",
                idx,
                num.format(".2s", annotation.score.az::<f32>()),
                num.format(".2s", annotation.log_likelihood.az::<f32>()),
                num.format(".2s", annotation.velocity[0].az::<f32>()),
                num.format(".2s", annotation.velocity[1].az::<f32>()),
            );

            // TODO: draw pos history

            draw_text_mut(
                &mut new_img,
                *draw_color,
                (annotation.position[0].az::<u32>()).saturating_sub(5u32) as i32,
                (annotation.position[1].az::<u32>()).saturating_sub(30u32) as i32,
                self.big_scale,
                &self.font,
                label.as_str(),
            );
        }

        new_img
    }
}
