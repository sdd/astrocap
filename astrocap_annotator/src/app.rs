use eframe::egui;
use std::time::Duration;

use crate::{
    annotations::AnnotationSession, annotations::Keyframe, annotations::ObjectType,
    annotations::TrackedObject, video::VideoFrameCache,
};

pub struct AnnotatorApp {
    video_cache: Option<VideoFrameCache>,
    current_frame: u32,
    is_playing: bool,
    play_speed: f32,
    last_frame_time: std::time::Instant,

    // New annotation fields
    annotations: Option<AnnotationSession>,
    selected_object_id: Option<u32>,
}

impl Default for AnnotatorApp {
    fn default() -> Self {
        Self {
            video_cache: None,
            current_frame: 0,
            is_playing: false,
            play_speed: 1.0,
            last_frame_time: std::time::Instant::now(),
            annotations: None,
            selected_object_id: None,
        }
    }
}

impl AnnotatorApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Default::default()
    }

    pub fn load_video(&mut self, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let cache = VideoFrameCache::new(file_path)?;

        // Initialize annotations for this video
        let annotations = AnnotationSession::new(
            file_path.to_string(),
            cache.width(),
            cache.height(),
            cache.frame_count() as u32, // Cast to u32
            cache.fps() as f32,
        );

        self.annotations = Some(annotations);
        self.video_cache = Some(cache);
        self.current_frame = 0;
        self.selected_object_id = None;
        Ok(())
    }

    fn show_zoom_tooltip(&self, ui: &mut egui::Ui, image_rect: egui::Rect, cursor_pos: egui::Pos2) {
        // Get current frame texture from video cache
        if let Some(video_cache) = &self.video_cache {
            if let Ok(texture) =
                video_cache.get_frame_texture(ui.ctx(), self.current_frame as usize)
            {
                // Check if cursor is over the video area
                if image_rect.contains(cursor_pos) {
                    // Calculate relative position in the image (0.0 to 1.0)
                    let rel_x = (cursor_pos.x - image_rect.min.x) / image_rect.width();
                    let rel_y = (cursor_pos.y - image_rect.min.y) / image_rect.height();

                    // Define zoom parameters
                    let zoom_factor = 40.0;
                    let zoom_window_size = 100.0;

                    // Calculate UV coordinates for the small area we want to zoom
                    // For 4x zoom, we sample 1/4 the area in each dimension
                    let sample_half_size = 0.5 / zoom_factor; // Half the size of what we sample

                    let source_rect = egui::Rect::from_center_size(
                        egui::pos2(rel_x, rel_y),
                        egui::vec2(sample_half_size * 2.0, sample_half_size * 2.0),
                    );

                    // Clamp to texture bounds
                    let clamped_source = egui::Rect::from_min_max(
                        egui::pos2(source_rect.min.x.max(0.0), source_rect.min.y.max(0.0)),
                        egui::pos2(source_rect.max.x.min(1.0), source_rect.max.y.min(1.0)),
                    );

                    // Position tooltip near cursor but avoid edges
                    let tooltip_offset = egui::vec2(20.0, -120.0);
                    let mut tooltip_pos = cursor_pos + tooltip_offset;

                    // Keep tooltip on screen
                    let screen_rect = ui.ctx().screen_rect();
                    if tooltip_pos.x + zoom_window_size + 40.0 > screen_rect.max.x {
                        tooltip_pos.x = cursor_pos.x - zoom_window_size - 40.0;
                    }
                    if tooltip_pos.y < screen_rect.min.y {
                        tooltip_pos.y = cursor_pos.y + 20.0;
                    }

                    // Show tooltip as a fixed-size window
                    egui::Area::new(egui::Id::new("zoom_tooltip"))
                        .fixed_pos(tooltip_pos)
                        .order(egui::Order::Tooltip)
                        .show(ui.ctx(), |ui| {
                            egui::Frame::popup(ui.style())
                                .fill(ui.style().visuals.panel_fill)
                                .stroke(ui.style().visuals.window_stroke)
                                .inner_margin(egui::Margin::same(5))
                                .show(ui, |ui| {
                                    // Set explicit size constraints
                                    ui.set_max_width(zoom_window_size + 10.0);
                                    ui.set_min_width(zoom_window_size + 10.0);

                                    // Create the zoomed image
                                    let image =
                                        egui::Image::new((texture.id(), texture.size_vec2()))
                                            .uv(clamped_source)
                                            .fit_to_exact_size(egui::vec2(
                                                zoom_window_size,
                                                zoom_window_size,
                                            ));

                                    let response = ui.add(image);
                                    let zoom_rect = response.rect;

                                    // Draw crosshair at the center of the zoomed area
                                    let center = zoom_rect.center();
                                    let crosshair_size = 8.0;

                                    ui.painter().line_segment(
                                        [
                                            center + egui::vec2(-crosshair_size, 0.0),
                                            center + egui::vec2(crosshair_size, 0.0),
                                        ],
                                        egui::Stroke::new(1.5, egui::Color32::RED),
                                    );
                                    ui.painter().line_segment(
                                        [
                                            center + egui::vec2(0.0, -crosshair_size),
                                            center + egui::vec2(0.0, crosshair_size),
                                        ],
                                        egui::Stroke::new(1.5, egui::Color32::RED),
                                    );
                                });
                        });
                }
            }
        }
    }

    fn handle_video_click(&mut self, image_rect: egui::Rect, click_pos: egui::Pos2) {
        if let (Some(annotations), Some(selected_id)) =
            (&mut self.annotations, self.selected_object_id)
        {
            // Convert screen coordinates to image coordinates
            let rel_x = (click_pos.x - image_rect.min.x) / image_rect.width();
            let rel_y = (click_pos.y - image_rect.min.y) / image_rect.height();

            let image_x = rel_x * annotations.video_metadata.width as f32;
            let image_y = rel_y * annotations.video_metadata.height as f32;

            // Find the selected object and add keyframe
            if let Some(selected_object) = annotations
                .objects
                .iter_mut()
                .find(|obj| obj.id == selected_id)
            {
                let keyframe = Keyframe {
                    frame_number: self.current_frame,
                    x: image_x as f32,
                    y: image_y as f32,
                    confidence: 1.0,
                    notes: None,
                };

                selected_object.add_keyframe(keyframe);
            }
        }
    }

    fn draw_annotations(
        &self,
        ui: &mut egui::Ui,
        image_rect: egui::Rect,
        annotations: &AnnotationSession,
    ) {
        let painter = ui.painter();

        for object in &annotations.objects {
            if object.keyframes.len() == 0 {
                continue;
            }

            // Sort keyframes by frame number for proper line drawing
            let mut sorted_keyframes = object.keyframes.clone();
            sorted_keyframes.sort_by_key(|k| k.frame_number);

            // Draw magenta crosses for each keyframe
            for keyframe in &sorted_keyframes {
                let screen_x = image_rect.min.x
                    + (keyframe.x as f32 / annotations.video_metadata.width as f32)
                        * image_rect.width();
                let screen_y = image_rect.min.y
                    + (keyframe.y as f32 / annotations.video_metadata.height as f32)
                        * image_rect.height();
                let screen_pos = egui::pos2(screen_x, screen_y);

                let cross_size = 4.0;
                let magenta = egui::Color32::from_rgb(255, 0, 255);

                // Draw cross
                painter.line_segment(
                    [
                        screen_pos + egui::vec2(-cross_size, 0.0),
                        screen_pos + egui::vec2(cross_size, 0.0),
                    ],
                    egui::Stroke::new(2.0, magenta),
                );
                painter.line_segment(
                    [
                        screen_pos + egui::vec2(0.0, -cross_size),
                        screen_pos + egui::vec2(0.0, cross_size),
                    ],
                    egui::Stroke::new(2.0, magenta),
                );
            }

            // Draw lines between keyframes and interpolated position
            for i in 0..sorted_keyframes.len().saturating_sub(1) {
                let kf1 = &sorted_keyframes[i];
                let kf2 = &sorted_keyframes[i + 1];

                let screen_pos1 = egui::pos2(
                    image_rect.min.x
                        + (kf1.x as f32 / annotations.video_metadata.width as f32)
                            * image_rect.width(),
                    image_rect.min.y
                        + (kf1.y as f32 / annotations.video_metadata.height as f32)
                            * image_rect.height(),
                );
                let screen_pos2 = egui::pos2(
                    image_rect.min.x
                        + (kf2.x as f32 / annotations.video_metadata.width as f32)
                            * image_rect.width(),
                    image_rect.min.y
                        + (kf2.y as f32 / annotations.video_metadata.height as f32)
                            * image_rect.height(),
                );

                // Determine line color based on current frame
                let line_color = if self.current_frame >= kf1.frame_number
                    && self.current_frame <= kf2.frame_number
                {
                    egui::Color32::from_rgb(255, 180, 255) // light magenta
                } else {
                    egui::Color32::from_rgb(128, 0, 128) // dark magenta
                };

                // Draw line between keyframes
                painter.line_segment(
                    [screen_pos1, screen_pos2],
                    egui::Stroke::new(1.5, line_color),
                );

                // Draw green circle at interpolated position if current frame is between these keyframes
                if self.current_frame >= kf1.frame_number && self.current_frame <= kf2.frame_number
                {
                    let t = if kf2.frame_number == kf1.frame_number {
                        0.0
                    } else {
                        (self.current_frame - kf1.frame_number) as f32
                            / (kf2.frame_number - kf1.frame_number) as f32
                    };

                    let interp_pos = screen_pos1 + t * (screen_pos2 - screen_pos1);
                    painter.circle_filled(interp_pos, 3.0, egui::Color32::GREEN);
                }
            }
        }
    }
}

impl eframe::App for AnnotatorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Force default cursor at the start of every frame to prevent cursor crashes
        ctx.set_cursor_icon(egui::CursorIcon::Default);

        // Handle continuous playback
        if self.is_playing {
            if let Some(ref cache) = self.video_cache {
                let now = std::time::Instant::now();
                let elapsed = now.duration_since(self.last_frame_time);
                let target_duration = Duration::from_secs_f64(self.play_speed as f64 / cache.fps());

                if elapsed >= target_duration {
                    let max_frame = cache.frame_count().saturating_sub(1);
                    if self.current_frame < (max_frame as u32) {
                        self.current_frame += 1;
                        self.last_frame_time = now;
                    } else {
                        self.is_playing = false; // Stop at end
                    }
                    ctx.request_repaint();
                }
            }
        }

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Open Video...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Video files", &["mp4", "avi", "mov", "mkv"])
                            .pick_file()
                        {
                            if let Err(e) = self.load_video(&path.display().to_string()) {
                                eprintln!("Failed to load video: {}", e);
                            }
                        }
                        ui.close_menu();
                    }
                    if ui.button("Save Annotations...").clicked() {
                        if let Some(annotations) = &self.annotations {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("JSON files", &["json"])
                                .save_file()
                            {
                                if let Err(e) =
                                    annotations.save_to_file(&path.display().to_string())
                                {
                                    eprintln!("Failed to save annotations: {}", e);
                                }
                            }
                        }
                        ui.close_menu();
                    }
                    if ui.button("Load Annotations...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("JSON files", &["json"])
                            .pick_file()
                        {
                            match AnnotationSession::load_from_file(&path.display().to_string()) {
                                Ok(annotations) => {
                                    self.annotations = Some(annotations);
                                    self.selected_object_id = None;
                                }
                                Err(e) => eprintln!("Failed to load annotations: {}", e),
                            }
                        }
                        ui.close_menu();
                    }
                });
            });
        });

        egui::SidePanel::left("objects_panel")
            .resizable(true)
            .default_width(200.0)
            .show(ctx, |ui| {
                ui.heading("Objects");

                if ui.button("Add Object").clicked() {
                    if let Some(annotations) = &mut self.annotations {
                        let object_count = annotations.objects.len();
                        let new_id = annotations
                            .add_object(format!("Object {}", object_count + 1), ObjectType::Star);
                        self.selected_object_id = Some(new_id);
                    }
                }

                ui.separator();

                if let Some(annotations) = &mut self.annotations {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for object in &mut annotations.objects {
                            let is_selected = self.selected_object_id == Some(object.id);

                            let response = ui.selectable_label(is_selected, &object.name);

                            if response.clicked() {
                                self.selected_object_id = Some(object.id);
                            }

                            if is_selected {
                                ui.indent("object_details", |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label("Name:");
                                        ui.text_edit_singleline(&mut object.name);
                                    });

                                    ui.label(format!("Keyframes: {}", object.keyframes.len()));

                                    if let Some((start, end)) = object.frame_range() {
                                        ui.label(format!("Range: {} - {}", start, end));
                                    }
                                });
                            }
                        }
                    });
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(ref cache) = self.video_cache {
                // Add bounds checking for current_frame before any texture operations
                let frame_count = cache.frame_count();
                if frame_count == 0 {
                    ui.centered_and_justified(|ui| {
                        ui.label("Video loaded but no frames available");
                    });
                    return;
                }

                // Ensure current_frame is within bounds
                let max_frame = (frame_count - 1) as u32;
                self.current_frame = self.current_frame.clamp(0, max_frame);

                // Check if we have sufficient space to render
                let available_size = ui.available_size();
                if available_size.x < 10.0 || available_size.y < 10.0 {
                    // Window is too small or being resized, skip rendering
                    ui.centered_and_justified(|ui| {
                        ui.label("Resizing...");
                    });
                    return;
                }

                match cache.get_frame_texture(ctx, self.current_frame as usize) {
                    Ok(texture) => {
                        let aspect_ratio = cache.width() as f32 / cache.height() as f32;

                        let display_size = if available_size.x / available_size.y > aspect_ratio {
                            egui::Vec2::new(available_size.y * aspect_ratio, available_size.y)
                        } else {
                            egui::Vec2::new(available_size.x, available_size.x / aspect_ratio)
                        };

                        // Additional safety check for display size
                        if display_size.x <= 0.0 || display_size.y <= 0.0 {
                            ui.centered_and_justified(|ui| {
                                ui.label("Invalid display size");
                            });
                            return;
                        }

                        let image_rect = egui::Rect::from_center_size(
                            ui.available_rect_before_wrap().center(),
                            display_size,
                        );

                        // Ensure image_rect is valid
                        if !image_rect.is_positive() {
                            ui.centered_and_justified(|ui| {
                                ui.label("Invalid image rectangle");
                            });
                            return;
                        }

                        // Allocate space for the video but be more careful about interaction
                        let response = ui.allocate_rect(image_rect, egui::Sense::click());

                        // Only paint if the image rect is valid and visible
                        if image_rect.is_positive() && ui.is_rect_visible(image_rect) {
                            ui.painter().image(
                                texture.id(),
                                image_rect,
                                egui::Rect::from_min_max(
                                    egui::pos2(0.0, 0.0),
                                    egui::pos2(1.0, 1.0),
                                ),
                                egui::Color32::WHITE,
                            );

                            // Draw annotations on top of the video
                            if let Some(annotations) = &self.annotations {
                                self.draw_annotations(ui, image_rect, annotations);
                            }
                        }

                        // Handle clicks on the video (only if clicked, not hovered)
                        if response.clicked() {
                            if let Some(click_pos) = response.interact_pointer_pos() {
                                // Additional check that click is within bounds
                                if image_rect.contains(click_pos) {
                                    self.handle_video_click(image_rect, click_pos);
                                }
                            }
                        }

                        // Show zoom tooltip when hovering over the video area
                        if let Some(hover_pos) = ui.input(|i| i.pointer.hover_pos()) {
                            if image_rect.contains(hover_pos) {
                                self.show_zoom_tooltip(ui, image_rect, hover_pos);
                            }
                        }

                        // Draw the annotations on top
                        if let Some(annotations) = &self.annotations {
                            self.draw_annotations(ui, image_rect, annotations);
                        }
                    }
                    Err(e) => {
                        ui.centered_and_justified(|ui| {
                            ui.label(format!("Error loading frame: {}", e));
                        });
                    }
                }
            } else {
                ui.centered_and_justified(|ui| {
                    ui.label("No video loaded. Use File > Open Video to get started.");
                });
            }
        });

        egui::TopBottomPanel::bottom("controls_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                // Play/pause controls
                if ui.button(if self.is_playing { "⏸" } else { "▶" }).clicked() {
                    self.is_playing = !self.is_playing;
                    if self.is_playing {
                        self.last_frame_time = std::time::Instant::now();
                    }
                }

                if ui.button("⏹").clicked() {
                    self.is_playing = false;
                    self.current_frame = 0;
                }

                ui.add(
                    egui::Slider::new(&mut self.play_speed, 0.1..=3.0)
                        .text("Speed")
                        .show_value(false),
                );

                ui.separator();

                // Frame navigation
                if let Some(ref cache) = self.video_cache {
                    let frame_count = cache.frame_count() as u32; // Cast to u32
                    if frame_count > 0 {
                        let max_frame = frame_count - 1;

                        // Store the original value to detect changes
                        let original_frame = self.current_frame;

                        // Ensure current_frame is within bounds before creating slider
                        self.current_frame = self.current_frame.clamp(0, max_frame);

                        // Create a local copy for the slider to modify
                        let mut slider_value = self.current_frame;

                        let slider_response = ui.add(
                            egui::Slider::new(&mut slider_value, 0..=max_frame)
                                .text("Frame")
                                .clamp_to_range(true), // Force clamping
                        );

                        // Only update if the slider was actually interacted with and value is valid
                        if slider_response.changed() && slider_value <= max_frame {
                            self.current_frame = slider_value.clamp(0, max_frame);
                        } else if slider_value > max_frame {
                            // If somehow slider went out of bounds, reset it
                            self.current_frame = max_frame;
                        }

                        // Double-check bounds one more time
                        self.current_frame = self.current_frame.clamp(0, max_frame);

                        ui.separator();

                        let current_time = if cache.fps() > 0.0 {
                            self.current_frame as f64 / cache.fps()
                        } else {
                            0.0
                        };

                        let total_time = cache.duration_seconds();

                        ui.label(format!(
                            "Frame {} / {} ({:.1}s / {:.1}s)",
                            self.current_frame, frame_count, current_time, total_time
                        ));
                    } else {
                        ui.label("No frames loaded");
                    }
                }
            });
        });

        // Force default cursor again at the end
        ctx.set_cursor_icon(egui::CursorIcon::Default);
    }
}
