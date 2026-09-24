mod detection;
mod openai_detection;

use std::{
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, Sender},
    thread,
};

use eframe::egui::{
    self, Color32, ColorImage, CursorIcon, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle,
    TextureOptions, Vec2,
};
use image::DynamicImage;

use crate::detection::{detect_photos, PhotoRect};

const HANDLE_RADIUS: f32 = 7.0;

#[derive(Clone, Copy, PartialEq, Eq)]
enum DragMode {
    Move,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

struct ActiveDrag {
    index: usize,
    mode: DragMode,
    start_pointer: Pos2,
    start_rect: PhotoRect,
}

struct DetectionJob {
    id: u64,
    image: DynamicImage,
    threshold: u8,
    margin: u32,
}

struct DetectionResult {
    id: u64,
    boxes: Vec<PhotoRect>,
    engine: &'static str,
    warning: Option<String>,
}

struct SlicerApp {
    image: Option<DynamicImage>,
    texture: Option<TextureHandle>,
    image_path: Option<PathBuf>,
    boxes: Vec<PhotoRect>,
    selected: Option<usize>,
    drag: Option<ActiveDrag>,
    threshold: u8,
    margin: u32,
    status: String,
    detection_tx: Sender<DetectionJob>,
    detection_rx: Receiver<DetectionResult>,
    detection_id: u64,
    detecting: bool,
}

impl SlicerApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<DetectionJob>();
        let (result_tx, result_rx) = mpsc::channel::<DetectionResult>();

        thread::spawn(move || {
            while let Ok(job) = job_rx.recv() {
                let output = detect_photos(&job.image, job.threshold, job.margin);
                if result_tx
                    .send(DetectionResult {
                        id: job.id,
                        boxes: output.boxes,
                        engine: output.engine,
                        warning: output.warning,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });

        Self {
            image: None,
            texture: None,
            image_path: None,
            boxes: Vec::new(),
            selected: None,
            drag: None,
            threshold: 22,
            margin: 0,
            status: "Open a scan to begin.".into(),
            detection_tx: job_tx,
            detection_rx: result_rx,
            detection_id: 0,
            detecting: false,
        }
    }

    fn open_image(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff"])
            .pick_file()
        else {
            return;
        };

        match image::open(&path) {
            Ok(image) => {
                let rgba = image.to_rgba8();
                let size = [rgba.width() as usize, rgba.height() as usize];
                let color = ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                self.texture = Some(ctx.load_texture("scan", color, TextureOptions::LINEAR));
                self.image = Some(image);
                self.image_path = Some(path.clone());
                self.selected = None;
                self.drag = None;
                self.status = format!("Loaded {}", path.display());
                self.redetect();
            }
            Err(error) => self.status = format!("Could not open image: {error}"),
        }
    }

    fn redetect(&mut self) {
        let Some(image) = self.image.as_ref() else {
            return;
        };

        self.detection_id = self.detection_id.wrapping_add(1);
        let job = DetectionJob {
            id: self.detection_id,
            image: image.clone(),
            threshold: self.threshold,
            margin: self.margin,
        };

        match self.detection_tx.send(job) {
            Ok(()) => {
                self.detecting = true;
                self.status = "Detecting photos…".into();
            }
            Err(error) => {
                self.detecting = false;
                self.status = format!("Could not start detection: {error}");
            }
        }
    }

    fn poll_detection(&mut self) {
        while let Ok(result) = self.detection_rx.try_recv() {
            if result.id != self.detection_id {
                continue;
            }

            self.boxes = result.boxes;
            self.selected = None;
            self.detecting = false;
            self.status = match result.warning {
                Some(warning) => format!(
                    "Detected {} photo(s) with {} — {}",
                    self.boxes.len(),
                    result.engine,
                    warning
                ),
                None => format!(
                    "Detected {} photo(s) with {}.",
                    self.boxes.len(),
                    result.engine
                ),
            };
        }
    }

    fn export(&mut self) {
        let Some(image) = self.image.as_ref() else {
            return;
        };
        if self.boxes.is_empty() {
            self.status = "Nothing to export.".into();
            return;
        }

        let default_dir = self
            .image_path
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf);

        let mut dialog = rfd::FileDialog::new();
        if let Some(dir) = default_dir {
            dialog = dialog.set_directory(dir);
        }

        let Some(dir) = dialog.pick_folder() else {
            return;
        };

        let stem = self
            .image_path
            .as_deref()
            .and_then(Path::file_stem)
            .and_then(|s| s.to_str())
            .unwrap_or("scan");

        let mut exported = 0usize;
        for (index, rect) in self.boxes.iter().enumerate() {
            let rect = rect.clamped(image.width(), image.height());
            if rect.w < 2 || rect.h < 2 {
                continue;
            }

            let crop = image.crop_imm(rect.x, rect.y, rect.w, rect.h);
            let path = dir.join(format!("{stem}_{:02}.png", index + 1));
            match crop.save(&path) {
                Ok(()) => exported += 1,
                Err(error) => {
                    self.status = format!("Export failed at {}: {error}", path.display());
                    return;
                }
            }
        }

        self.status = format!("Exported {exported} photo(s) to {}", dir.display());
    }

    fn add_box(&mut self) {
        let Some(image) = self.image.as_ref() else {
            return;
        };

        let w = (image.width() / 3).max(100);
        let h = (image.height() / 3).max(100);
        let rect = PhotoRect {
            x: (image.width().saturating_sub(w)) / 2,
            y: (image.height().saturating_sub(h)) / 2,
            w,
            h,
            corners: None,
        };
        self.boxes.push(rect);
        self.selected = Some(self.boxes.len() - 1);
    }

    fn remove_selected(&mut self) {
        if let Some(index) = self.selected.take() {
            if index < self.boxes.len() {
                self.boxes.remove(index);
            }
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let (Some(texture), Some(image)) = (self.texture.as_ref(), self.image.as_ref()) else {
            ui.centered_and_justified(|ui| {
                ui.label("Open a scan to start slicing.");
            });
            return;
        };

        let available = ui.available_size();
        let source = Vec2::new(image.width() as f32, image.height() as f32);
        let scale = (available.x / source.x)
            .min(available.y / source.y)
            .min(1.0)
            .max(0.01);
        let display = source * scale;

        let response = ui.add(
            egui::Image::new((texture.id(), display))
                .sense(Sense::click_and_drag())
                .maintain_aspect_ratio(true),
        );
        let canvas = response.rect;

        if let Some(pointer) = response.interact_pointer_pos() {
            let image_pos = |p: Pos2| -> Pos2 {
                Pos2::new(
                    ((p.x - canvas.left()) / scale).clamp(0.0, image.width() as f32),
                    ((p.y - canvas.top()) / scale).clamp(0.0, image.height() as f32),
                )
            };

            if response.drag_started() {
                if let Some((index, mode)) = self.hit_test(pointer, canvas, scale) {
                    self.selected = Some(index);
                    self.drag = Some(ActiveDrag {
                        index,
                        mode,
                        start_pointer: image_pos(pointer),
                        start_rect: self.boxes[index],
                    });
                } else {
                    self.selected = None;
                }
            }

            if response.dragged() {
                if let Some(drag) = self.drag.as_ref() {
                    let now = image_pos(pointer);
                    let dx = now.x - drag.start_pointer.x;
                    let dy = now.y - drag.start_pointer.y;
                    let mut rect = drag.start_rect;
                    apply_drag(&mut rect, drag.mode, dx, dy, image.width(), image.height());
                    self.boxes[drag.index] = rect;
                    ui.ctx().set_cursor_icon(match drag.mode {
                        DragMode::Move => CursorIcon::Grabbing,
                        _ => CursorIcon::ResizeNwSe,
                    });
                }
            }

            if response.drag_stopped() {
                self.drag = None;
            }

            if response.clicked() && self.drag.is_none() {
                self.selected = self.hit_test(pointer, canvas, scale).map(|(i, _)| i);
            }
        }

        let painter = ui.painter_at(canvas);
        for (index, rect) in self.boxes.iter().enumerate() {
            let screen = rect_to_screen(*rect, canvas, scale);
            let selected = self.selected == Some(index);
            let stroke = if selected {
                Stroke::new(2.5_f32, Color32::YELLOW)
            } else {
                Stroke::new(2.0_f32, Color32::from_rgb(255, 80, 80))
            };

            let points = rect_screen_corners(*rect, canvas, scale);
            if rect.corners.is_some() {
                for i in 0..4 {
                    painter.line_segment([points[i], points[(i + 1) % 4]], stroke);
                }
            } else {
                painter.rect_stroke(screen, 0.0, stroke, StrokeKind::Outside);
            }

            let badge_origin = points
                .iter()
                .copied()
                .min_by(|a, b| {
                    a.y.partial_cmp(&b.y)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
                })
                .unwrap_or(screen.min);
            let badge = Rect::from_min_size(
                badge_origin + Vec2::new(4.0, 4.0),
                Vec2::new(26.0, 22.0),
            );
            painter.rect_filled(badge, 4.0, Color32::from_black_alpha(170));
            painter.text(
                badge.center(),
                egui::Align2::CENTER_CENTER,
                format!("{}", index + 1),
                egui::FontId::proportional(14.0),
                Color32::WHITE,
            );

            if selected {
                for point in points {
                    painter.circle_filled(point, HANDLE_RADIUS, Color32::YELLOW);
                    painter.circle_stroke(
                        point,
                        HANDLE_RADIUS,
                        Stroke::new(1.0_f32, Color32::BLACK),
                    );
                }
            }
        }
    }

    fn hit_test(&self, pointer: Pos2, canvas: Rect, scale: f32) -> Option<(usize, DragMode)> {
        for (index, rect) in self.boxes.iter().enumerate().rev() {
            let screen = rect_to_screen(*rect, canvas, scale);
            let handles = [
                (screen.left_top(), DragMode::TopLeft),
                (screen.right_top(), DragMode::TopRight),
                (screen.left_bottom(), DragMode::BottomLeft),
                (screen.right_bottom(), DragMode::BottomRight),
            ];

            for (point, mode) in handles {
                if point.distance(pointer) <= HANDLE_RADIUS + 5.0 {
                    return Some((index, mode));
                }
            }

            if screen.contains(pointer) {
                return Some((index, DragMode::Move));
            }
        }
        None
    }
}

impl eframe::App for SlicerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_detection();
        if self.detecting {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Open scan").clicked() {
                    self.open_image(ctx);
                }
                if ui
                    .add_enabled(
                        self.image.is_some() && !self.detecting,
                        egui::Button::new("Detect photos"),
                    )
                    .clicked()
                {
                    self.redetect();
                }
                if ui
                    .add_enabled(self.image.is_some(), egui::Button::new("+ Add frame"))
                    .clicked()
                {
                    self.add_box();
                }
                if ui
                    .add_enabled(self.selected.is_some(), egui::Button::new("Delete frame"))
                    .clicked()
                {
                    self.remove_selected();
                }
                if ui
                    .add_enabled(!self.boxes.is_empty(), egui::Button::new("Export PNGs"))
                    .clicked()
                {
                    self.export();
                }

                ui.separator();
                ui.label("Detection");
                let threshold_changed = ui
                    .add(egui::Slider::new(&mut self.threshold, 5..=80).text("threshold"))
                    .changed();
                let margin_changed = ui
                    .add(egui::Slider::new(&mut self.margin, 0..=100).text("margin px"))
                    .changed();

                if (threshold_changed || margin_changed) && self.image.is_some() {
                    self.status = "Detection settings changed — click Detect photos.".into();
                }
            });
        });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                if let Some(index) = self.selected {
                    if let Some(rect) = self.boxes.get(index) {
                        ui.separator();
                        ui.label(format!(
                            "Frame {}: {}×{} at {},{}",
                            index + 1,
                            rect.w,
                            rect.h,
                            rect.x,
                            rect.y
                        ));
                    }
                }
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| self.canvas(ui));
    }
}

fn rect_to_screen(rect: PhotoRect, canvas: Rect, scale: f32) -> Rect {
    Rect::from_min_size(
        Pos2::new(
            canvas.left() + rect.x as f32 * scale,
            canvas.top() + rect.y as f32 * scale,
        ),
        Vec2::new(rect.w as f32 * scale, rect.h as f32 * scale),
    )
}

fn rect_screen_corners(rect: PhotoRect, canvas: Rect, scale: f32) -> [Pos2; 4] {
    if let Some(corners) = rect.corners {
        corners.map(|[x, y]| {
            Pos2::new(
                canvas.left() + x * scale,
                canvas.top() + y * scale,
            )
        })
    } else {
        let screen = rect_to_screen(rect, canvas, scale);
        [
            screen.left_top(),
            screen.right_top(),
            screen.right_bottom(),
            screen.left_bottom(),
        ]
    }
}

fn apply_drag(
    rect: &mut PhotoRect,
    mode: DragMode,
    dx: f32,
    dy: f32,
    image_w: u32,
    image_h: u32,
) {
    let min_size = 20i32;
    let mut left = rect.x as i32;
    let mut top = rect.y as i32;
    let mut right = (rect.x + rect.w) as i32;
    let mut bottom = (rect.y + rect.h) as i32;
    let dx = dx.round() as i32;
    let dy = dy.round() as i32;

    match mode {
        DragMode::Move => {
            let width = right - left;
            let height = bottom - top;
            left = (left + dx).clamp(0, image_w as i32 - width);
            top = (top + dy).clamp(0, image_h as i32 - height);
            right = left + width;
            bottom = top + height;
        }
        DragMode::TopLeft => {
            left = (left + dx).clamp(0, right - min_size);
            top = (top + dy).clamp(0, bottom - min_size);
        }
        DragMode::TopRight => {
            right = (right + dx).clamp(left + min_size, image_w as i32);
            top = (top + dy).clamp(0, bottom - min_size);
        }
        DragMode::BottomLeft => {
            left = (left + dx).clamp(0, right - min_size);
            bottom = (bottom + dy).clamp(top + min_size, image_h as i32);
        }
        DragMode::BottomRight => {
            right = (right + dx).clamp(left + min_size, image_w as i32);
            bottom = (bottom + dy).clamp(top + min_size, image_h as i32);
        }
    }

    rect.x = left as u32;
    rect.y = top as u32;
    rect.w = (right - left) as u32;
    rect.h = (bottom - top) as u32;
    rect.corners = None;
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Slicer")
            .with_inner_size([1200.0, 820.0])
            .with_min_inner_size([720.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Slicer",
        options,
        Box::new(|cc| Ok(Box::new(SlicerApp::new(cc)))),
    )
}
