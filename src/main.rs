mod detection;
mod openai_detection;

use std::{
    path::{Path, PathBuf},
    sync::{
        mpsc::{self, Receiver, Sender},
        Arc,
    },
    thread,
};

use eframe::egui::{
    self, Align, Color32, ColorImage, CursorIcon, FontId, Layout, Pos2, Rect, RichText, Sense,
    Stroke, StrokeKind, TextureHandle, TextureOptions, Vec2,
};
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::geometric_transformations::{warp_into, Border, Interpolation, Projection};

use crate::detection::{detect_photos, PhotoRect};

const HANDLE_RADIUS: f32 = 7.0;

fn accent() -> Color32 {
    Color32::from_rgb(96, 137, 255)
}

fn accent_hover() -> Color32 {
    Color32::from_rgb(114, 151, 255)
}

fn surface() -> Color32 {
    Color32::from_rgb(29, 32, 40)
}

fn workspace() -> Color32 {
    Color32::from_rgb(17, 19, 24)
}

fn border() -> Color32 {
    Color32::from_rgb(55, 60, 72)
}

fn muted() -> Color32 {
    Color32::from_rgb(154, 161, 177)
}

fn action_button(ui: &mut egui::Ui, label: &str, enabled: bool, primary: bool) -> bool {
    let fill = if primary { accent() } else { surface() };
    let stroke = if primary {
        Stroke::new(1.0, accent())
    } else {
        Stroke::new(1.0, border())
    };

    ui.add_enabled(
        enabled,
        egui::Button::new(RichText::new(label).size(14.0).strong())
            .min_size(Vec2::new(ui.available_width(), 42.0))
            .fill(fill)
            .stroke(stroke)
            .corner_radius(9),
    )
    .clicked()
}

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
    image: Arc<DynamicImage>,
    margin: u32,
}

struct DetectionResult {
    id: u64,
    boxes: Vec<PhotoRect>,
    engine: &'static str,
    warning: Option<String>,
}

struct LoadResult {
    path: PathBuf,
    result: Result<DynamicImage, String>,
}

enum ExportEvent {
    Progress {
        completed: usize,
        total: usize,
        exported: usize,
    },
    Finished {
        exported: usize,
        dir: PathBuf,
    },
    Failed(String),
}

struct SlicerApp {
    image: Option<Arc<DynamicImage>>,
    texture: Option<TextureHandle>,
    image_path: Option<PathBuf>,
    boxes: Vec<PhotoRect>,
    selected: Option<usize>,
    drag: Option<ActiveDrag>,
    margin: u32,
    status: String,
    detection_tx: Sender<DetectionJob>,
    detection_rx: Receiver<DetectionResult>,
    detection_id: u64,
    detecting: bool,
    load_tx: Sender<LoadResult>,
    load_rx: Receiver<LoadResult>,
    loading: bool,
    export_tx: Sender<ExportEvent>,
    export_rx: Receiver<ExportEvent>,
    exporting: bool,
}

impl SlicerApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<DetectionJob>();
        let (result_tx, result_rx) = mpsc::channel::<DetectionResult>();
        let (load_tx, load_rx) = mpsc::channel::<LoadResult>();
        let (export_tx, export_rx) = mpsc::channel::<ExportEvent>();

        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = Color32::from_rgb(23, 25, 31);
        visuals.window_fill = Color32::from_rgb(23, 25, 31);
        visuals.extreme_bg_color = workspace();
        visuals.faint_bg_color = surface();
        visuals.selection.bg_fill = accent();
        visuals.selection.stroke = Stroke::new(1.0, Color32::WHITE);
        visuals.hyperlink_color = accent_hover();
        _cc.egui_ctx.set_visuals(visuals);

        let mut style = (*_cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(10.0, 10.0);
        style.spacing.button_padding = Vec2::new(14.0, 9.0);
        style.spacing.interact_size.y = 38.0;
        _cc.egui_ctx.set_style(style);

        thread::spawn(move || {
            while let Ok(job) = job_rx.recv() {
                let output = detect_photos(job.image.as_ref(), job.margin);
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
            margin: 0,
            status: "Open a scan to begin.".into(),
            detection_tx: job_tx,
            detection_rx: result_rx,
            detection_id: 0,
            detecting: false,
            load_tx,
            load_rx,
            loading: false,
            export_tx,
            export_rx,
            exporting: false,
        }
    }

    fn open_image(&mut self, ctx: &egui::Context) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff"])
            .pick_file()
        else {
            return;
        };

        self.loading = true;
        self.status = format!("Loading {}…", path.display());

        let tx = self.load_tx.clone();
        let repaint = ctx.clone();
        thread::spawn(move || {
            let result = image::open(&path).map_err(|error| error.to_string());
            let _ = tx.send(LoadResult { path, result });
            repaint.request_repaint();
        });
    }

    fn poll_load(&mut self, ctx: &egui::Context) {
        let Ok(loaded) = self.load_rx.try_recv() else {
            return;
        };
        self.loading = false;

        match loaded.result {
            Ok(image) => {
                let image = Arc::new(image);
                // The original stays full-resolution for detection/export. The GPU only
                // needs a preview large enough for the editor window.
                const MAX_UI_PREVIEW_DIM: u32 = 2400;
                let preview = image.thumbnail(MAX_UI_PREVIEW_DIM, MAX_UI_PREVIEW_DIM).to_rgba8();
                let size = [preview.width() as usize, preview.height() as usize];
                let color = ColorImage::from_rgba_unmultiplied(size, preview.as_raw());
                self.texture = Some(ctx.load_texture("scan", color, TextureOptions::LINEAR));
                self.image = Some(image);
                self.image_path = Some(loaded.path.clone());
                self.boxes.clear();
                self.selected = None;
                self.drag = None;
                self.status = format!("Loaded {}", loaded.path.display());
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
            image: Arc::clone(image),
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

    fn export(&mut self, ctx: &egui::Context) {
        if self.exporting {
            return;
        }

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
            .unwrap_or("scan")
            .to_owned();

        let image = Arc::clone(image);
        let boxes = self.boxes.clone();
        let tx = self.export_tx.clone();
        let repaint = ctx.clone();

        self.exporting = true;
        self.status = format!("Exporting {} photo(s)…", boxes.len());

        thread::spawn(move || {
            let total = boxes.len();
            let rgba = image.to_rgba8();
            let mut exported = 0usize;

            for (index, rect) in boxes.iter().enumerate() {
                let rect = rect.clamped(image.width(), image.height());
                if rect.w < 2 || rect.h < 2 {
                    let _ = tx.send(ExportEvent::Progress {
                        completed: index + 1,
                        total,
                        exported,
                    });
                    repaint.request_repaint();
                    continue;
                }

                let crop = match rect.corners {
                    Some(corners) => match perspective_crop(&rgba, corners) {
                        Ok(crop) => crop,
                        Err(error) => {
                            let _ = tx.send(ExportEvent::Failed(format!(
                                "Export failed for frame {}: {error}",
                                index + 1
                            )));
                            repaint.request_repaint();
                            return;
                        }
                    },
                    None => {
                        image::imageops::crop_imm(&rgba, rect.x, rect.y, rect.w, rect.h).to_image()
                    }
                };

                let path = dir.join(format!("{stem}_{:02}.png", index + 1));
                if let Err(error) = crop.save(&path) {
                    let _ = tx.send(ExportEvent::Failed(format!(
                        "Export failed at {}: {error}",
                        path.display()
                    )));
                    repaint.request_repaint();
                    return;
                }

                exported += 1;
                let _ = tx.send(ExportEvent::Progress {
                    completed: index + 1,
                    total,
                    exported,
                });
                repaint.request_repaint();
            }

            let _ = tx.send(ExportEvent::Finished { exported, dir });
            repaint.request_repaint();
        });
    }

    fn poll_export(&mut self) {
        while let Ok(event) = self.export_rx.try_recv() {
            match event {
                ExportEvent::Progress {
                    completed,
                    total,
                    exported,
                } => {
                    self.status =
                        format!("Exporting {completed}/{total}… {exported} photo(s) saved.");
                }
                ExportEvent::Finished { exported, dir } => {
                    self.exporting = false;
                    self.status = format!("Exported {exported} photo(s) to {}", dir.display());
                }
                ExportEvent::Failed(error) => {
                    self.exporting = false;
                    self.status = error;
                }
            }
        }
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
        let available = ui.available_size();
        let (workspace_rect, _) = ui.allocate_exact_size(available, Sense::hover());
        let painter = ui.painter_at(workspace_rect);
        painter.rect_filled(workspace_rect, 0.0, workspace());

        let (Some(texture), Some(image)) = (self.texture.as_ref(), self.image.as_ref()) else {
            let card = Rect::from_center_size(
                workspace_rect.center(),
                Vec2::new(
                    390.0_f32.min(workspace_rect.width() - 40.0).max(240.0),
                    190.0_f32.min(workspace_rect.height() - 40.0).max(140.0),
                ),
            );
            painter.rect_filled(card, 18.0, surface());
            painter.rect_stroke(
                card,
                18.0,
                Stroke::new(1.0, border()),
                StrokeKind::Inside,
            );
            painter.text(
                card.center() - Vec2::new(0.0, 24.0),
                egui::Align2::CENTER_CENTER,
                "Open a scanned sheet",
                FontId::proportional(20.0),
                Color32::WHITE,
            );
            painter.text(
                card.center() + Vec2::new(0.0, 14.0),
                egui::Align2::CENTER_CENTER,
                "Photos will be detected automatically.",
                FontId::proportional(14.0),
                muted(),
            );
            return;
        };

        let source = Vec2::new(image.width() as f32, image.height() as f32);
        let padded = Vec2::new(
            (workspace_rect.width() - 64.0).max(1.0),
            (workspace_rect.height() - 64.0).max(1.0),
        );
        let scale = (padded.x / source.x)
            .min(padded.y / source.y)
            .min(1.0)
            .max(0.01);
        let display = source * scale;
        let canvas = Rect::from_center_size(workspace_rect.center(), display);

        painter.rect_filled(
            canvas.expand(12.0),
            14.0,
            Color32::from_black_alpha(85),
        );
        painter.image(
            texture.id(),
            canvas,
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::WHITE,
        );
        painter.rect_stroke(
            canvas,
            2.0,
            Stroke::new(1.0, Color32::from_white_alpha(35)),
            StrokeKind::Outside,
        );

        let response = ui.interact(
            canvas,
            ui.id().with("scan_canvas"),
            Sense::click_and_drag(),
        );

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
            } else if let Some((_, mode)) = self.hit_test(pointer, canvas, scale) {
                ui.ctx().set_cursor_icon(match mode {
                    DragMode::Move => CursorIcon::Grab,
                    _ => CursorIcon::ResizeNwSe,
                });
            }

            if response.drag_stopped() {
                self.drag = None;
            }

            if response.clicked() && self.drag.is_none() {
                self.selected = self.hit_test(pointer, canvas, scale).map(|(i, _)| i);
            }
        }

        for (index, rect) in self.boxes.iter().enumerate() {
            let screen = rect_to_screen(*rect, canvas, scale);
            let selected = self.selected == Some(index);
            let box_color = if selected {
                accent()
            } else {
                Color32::from_rgb(116, 158, 255)
            };
            let stroke = Stroke::new(if selected { 2.5 } else { 2.0 }, box_color);
            let outline = Stroke::new(
                if selected { 5.0 } else { 4.0 },
                Color32::from_black_alpha(190),
            );

            let points = rect_screen_corners(*rect, canvas, scale);
            if rect.corners.is_some() {
                for i in 0..4 {
                    painter.line_segment([points[i], points[(i + 1) % 4]], outline);
                    painter.line_segment([points[i], points[(i + 1) % 4]], stroke);
                }
            } else {
                painter.rect_stroke(screen, 2.0, outline, StrokeKind::Outside);
                painter.rect_stroke(screen, 2.0, stroke, StrokeKind::Outside);
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
                badge_origin + Vec2::new(7.0, 7.0),
                Vec2::new(28.0, 24.0),
            );
            painter.rect_filled(
                badge,
                7.0,
                if selected {
                    accent()
                } else {
                    Color32::from_black_alpha(190)
                },
            );
            painter.text(
                badge.center(),
                egui::Align2::CENTER_CENTER,
                format!("{}", index + 1),
                FontId::proportional(13.0),
                Color32::WHITE,
            );

            if selected {
                for point in points {
                    painter.circle_filled(point, HANDLE_RADIUS, accent());
                    painter.circle_stroke(
                        point,
                        HANDLE_RADIUS,
                        Stroke::new(2.0, Color32::from_black_alpha(220)),
                    );
                }
            }
        }
    }

    fn hit_test(&self, pointer: Pos2, canvas: Rect, scale: f32) -> Option<(usize, DragMode)> {
        for (index, rect) in self.boxes.iter().enumerate().rev() {
            let points = rect_screen_corners(*rect, canvas, scale);
            let handles = [
                (points[0], DragMode::TopLeft),
                (points[1], DragMode::TopRight),
                (points[2], DragMode::BottomRight),
                (points[3], DragMode::BottomLeft),
            ];

            for (point, mode) in handles {
                if point.distance(pointer) <= HANDLE_RADIUS + 5.0 {
                    return Some((index, mode));
                }
            }

            if point_in_quad(pointer, points) {
                return Some((index, DragMode::Move));
            }
        }
        None
    }
}

impl eframe::App for SlicerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_load(ctx);
        self.poll_detection();
        self.poll_export();
        if self.detecting || self.loading || self.exporting {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::TopBottomPanel::top("app_header")
            .exact_height(72.0)
            .show(ctx, |ui| {
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.add_space(18.0);
                    ui.vertical(|ui| {
                        ui.label(RichText::new("Scan Slicer").size(21.0).strong());
                        ui.label(
                            RichText::new("Split scanned photo sheets into clean image files")
                                .size(12.5)
                                .color(muted()),
                        );
                    });

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.add_space(18.0);
                        if self.loading || self.detecting || self.exporting {
                            ui.add(egui::Spinner::new().size(16.0));
                        }
                        ui.label(
                            RichText::new(format!(
                                "{} frame{}",
                                self.boxes.len(),
                                if self.boxes.len() == 1 { "" } else { "s" }
                            ))
                            .size(13.0)
                            .color(muted()),
                        );
                    });
                });
            });

        egui::SidePanel::left("sidebar")
            .exact_width(280.0)
            .resizable(false)
            .show(ctx, |ui| {
                ui.add_space(18.0);
                ui.label(RichText::new("SOURCE").size(11.0).strong().color(muted()));
                ui.add_space(6.0);

                if let (Some(path), Some(image)) = (self.image_path.as_ref(), self.image.as_ref()) {
                    ui.label(
                        RichText::new(
                            path.file_name()
                                .and_then(|name| name.to_str())
                                .unwrap_or("Loaded scan"),
                        )
                        .size(15.0)
                        .strong(),
                    );
                    ui.label(
                        RichText::new(format!("{} × {} px", image.width(), image.height()))
                            .size(12.5)
                            .color(muted()),
                    );
                } else {
                    ui.label(RichText::new("No scan loaded").size(15.0).strong());
                    ui.label(
                        RichText::new("PNG, JPG or TIFF")
                            .size(12.5)
                            .color(muted()),
                    );
                }

                ui.add_space(18.0);
                if action_button(ui, "Open scan", !self.loading && !self.exporting, false) {
                    self.open_image(ctx);
                }
                if action_button(
                    ui,
                    if self.detecting { "Detecting…" } else { "Detect photos" },
                    self.image.is_some() && !self.detecting && !self.loading && !self.exporting,
                    false,
                ) {
                    self.redetect();
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(10.0);

                ui.label(RichText::new("FRAMES").size(11.0).strong().color(muted()));
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    let half = (ui.available_width() - 8.0) / 2.0;
                    if ui
                        .add_enabled(
                            self.image.is_some() && !self.exporting,
                            egui::Button::new(RichText::new("+ Add").strong())
                                .min_size(Vec2::new(half, 38.0))
                                .fill(surface())
                                .stroke(Stroke::new(1.0, border()))
                                .corner_radius(9),
                        )
                        .clicked()
                    {
                        self.add_box();
                    }

                    if ui
                        .add_enabled(
                            self.selected.is_some() && !self.exporting,
                            egui::Button::new("Delete")
                                .min_size(Vec2::new(half, 38.0))
                                .fill(surface())
                                .stroke(Stroke::new(1.0, border()))
                                .corner_radius(9),
                        )
                        .clicked()
                    {
                        self.remove_selected();
                    }
                });

                ui.add_space(16.0);
                ui.label(RichText::new("Crop padding").size(13.0).strong());
                ui.label(
                    RichText::new("Add a little space around every detected photo.")
                        .size(12.0)
                        .color(muted()),
                );
                let margin_changed = ui
                    .add(egui::Slider::new(&mut self.margin, 0..=100).suffix(" px"))
                    .changed();
                if margin_changed && self.image.is_some() {
                    self.status = "Crop padding changed — run detection again.".into();
                }

                if let Some(index) = self.selected {
                    if let Some(rect) = self.boxes.get(index) {
                        ui.add_space(16.0);
                        ui.separator();
                        ui.add_space(12.0);
                        ui.label(
                            RichText::new(format!("Frame {}", index + 1))
                                .size(13.0)
                                .strong(),
                        );
                        ui.label(
                            RichText::new(format!(
                                "{} × {} px  ·  x {}, y {}",
                                rect.w, rect.h, rect.x, rect.y
                            ))
                            .size(12.0)
                            .color(muted()),
                        );
                    }
                }

                ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
                    ui.add_space(18.0);
                    if action_button(
                        ui,
                        "Export PNGs",
                        !self.boxes.is_empty()
                            && !self.loading
                            && !self.detecting
                            && !self.exporting,
                        true,
                    ) {
                        self.export(ctx);
                    }
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new("Drag a frame to move it. Drag its corners to resize.")
                            .size(11.5)
                            .color(muted()),
                    );
                });
            });

        egui::TopBottomPanel::bottom("status")
            .exact_height(42.0)
            .show(ctx, |ui| {
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    ui.add_space(12.0);
                    if self.loading || self.detecting || self.exporting {
                        ui.add(egui::Spinner::new().size(16.0));
                    }
                    ui.label(RichText::new(&self.status).size(12.5).color(muted()));
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(workspace()))
            .show(ctx, |ui| self.canvas(ui));
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
    let mut corners = rect_image_corners(*rect);

    match mode {
        DragMode::Move => {
            let min_x = corners.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
            let min_y = corners.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
            let max_x = corners
                .iter()
                .map(|p| p[0])
                .fold(f32::NEG_INFINITY, f32::max);
            let max_y = corners
                .iter()
                .map(|p| p[1])
                .fold(f32::NEG_INFINITY, f32::max);

            let dx = dx.clamp(-min_x, image_w as f32 - max_x);
            let dy = dy.clamp(-min_y, image_h as f32 - max_y);
            for point in &mut corners {
                point[0] += dx;
                point[1] += dy;
            }
        }
        mode => {
            let index = match mode {
                DragMode::TopLeft => 0,
                DragMode::TopRight => 1,
                DragMode::BottomRight => 2,
                DragMode::BottomLeft => 3,
                DragMode::Move => unreachable!(),
            };
            corners[index][0] = (corners[index][0] + dx).clamp(0.0, image_w as f32);
            corners[index][1] = (corners[index][1] + dy).clamp(0.0, image_h as f32);

            if !is_valid_quad(corners) {
                return;
            }
        }
    }

    set_rect_corners(rect, corners);
}

fn rect_image_corners(rect: PhotoRect) -> [[f32; 2]; 4] {
    rect.corners.unwrap_or([
        [rect.x as f32, rect.y as f32],
        [(rect.x + rect.w) as f32, rect.y as f32],
        [(rect.x + rect.w) as f32, (rect.y + rect.h) as f32],
        [rect.x as f32, (rect.y + rect.h) as f32],
    ])
}

fn set_rect_corners(rect: &mut PhotoRect, corners: [[f32; 2]; 4]) {
    let min_x = corners.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let min_y = corners.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let max_x = corners
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let max_y = corners
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max);

    let left = min_x.floor().max(0.0) as u32;
    let top = min_y.floor().max(0.0) as u32;
    let right = max_x.ceil().max(left as f32) as u32;
    let bottom = max_y.ceil().max(top as f32) as u32;

    rect.x = left;
    rect.y = top;
    rect.w = right.saturating_sub(left);
    rect.h = bottom.saturating_sub(top);
    rect.corners = Some(corners);
}

fn is_valid_quad(points: [[f32; 2]; 4]) -> bool {
    const MIN_EDGE: f32 = 20.0;

    for i in 0..4 {
        let a = points[i];
        let b = points[(i + 1) % 4];
        if point_distance(a, b) < MIN_EDGE {
            return false;
        }
    }

    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = points[i];
        let b = points[(i + 1) % 4];
        let c = points[(i + 2) % 4];
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() < 1.0 {
            return false;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if sign * cross < 0.0 {
            return false;
        }
    }

    true
}

fn point_in_quad(point: Pos2, points: [Pos2; 4]) -> bool {
    let mut has_positive = false;
    let mut has_negative = false;

    for i in 0..4 {
        let a = points[i];
        let b = points[(i + 1) % 4];
        let cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
        has_positive |= cross > 0.0;
        has_negative |= cross < 0.0;
        if has_positive && has_negative {
            return false;
        }
    }

    true
}

fn point_distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn perspective_crop(
    image: &RgbaImage,
    corners: [[f32; 2]; 4],
) -> Result<RgbaImage, String> {
    if !is_valid_quad(corners) {
        return Err("the crop corners do not form a valid quadrilateral".into());
    }

    let width = point_distance(corners[0], corners[1])
        .max(point_distance(corners[3], corners[2]))
        .round()
        .max(2.0) as u32;
    let height = point_distance(corners[0], corners[3])
        .max(point_distance(corners[1], corners[2]))
        .round()
        .max(2.0) as u32;

    let from = corners.map(|[x, y]| (x, y));
    let to = [
        (0.0, 0.0),
        ((width - 1) as f32, 0.0),
        ((width - 1) as f32, (height - 1) as f32),
        (0.0, (height - 1) as f32),
    ];
    let projection = Projection::from_control_points(from, to)
        .ok_or_else(|| "could not calculate perspective transform".to_string())?;

    let mut output = RgbaImage::new(width, height);
    warp_into(
        image,
        projection,
        Interpolation::Bicubic,
        Border::Constant(Rgba([0, 0, 0, 0])),
        &mut output,
    );
    Ok(output)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dragging_one_corner_keeps_the_quadrilateral() {
        let mut rect = PhotoRect {
            x: 10,
            y: 10,
            w: 100,
            h: 90,
            corners: Some([
                [10.0, 20.0],
                [100.0, 10.0],
                [110.0, 90.0],
                [20.0, 100.0],
            ]),
        };

        apply_drag(&mut rect, DragMode::TopLeft, 8.0, -5.0, 200, 200);

        let corners = rect.corners.expect("quad must be preserved");
        assert_eq!(corners[0], [18.0, 15.0]);
        assert_eq!(corners[1], [100.0, 10.0]);
        assert_eq!(corners[2], [110.0, 90.0]);
        assert_eq!(corners[3], [20.0, 100.0]);
    }

    #[test]
    fn moving_a_quad_preserves_its_shape() {
        let mut rect = PhotoRect {
            x: 10,
            y: 10,
            w: 100,
            h: 90,
            corners: Some([
                [10.0, 20.0],
                [100.0, 10.0],
                [110.0, 90.0],
                [20.0, 100.0],
            ]),
        };

        apply_drag(&mut rect, DragMode::Move, 15.0, 12.0, 200, 200);

        assert_eq!(
            rect.corners.unwrap(),
            [
                [25.0, 32.0],
                [115.0, 22.0],
                [125.0, 102.0],
                [35.0, 112.0],
            ]
        );
    }

    #[test]
    fn hit_area_follows_rotated_quad_instead_of_bbox() {
        let quad = [
            Pos2::new(50.0, 10.0),
            Pos2::new(90.0, 50.0),
            Pos2::new(50.0, 90.0),
            Pos2::new(10.0, 50.0),
        ];

        assert!(point_in_quad(Pos2::new(50.0, 50.0), quad));
        assert!(!point_in_quad(Pos2::new(15.0, 15.0), quad));
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Scan Slicer")
            .with_inner_size([1200.0, 820.0])
            .with_min_inner_size([720.0, 480.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Scan Slicer",
        options,
        Box::new(|cc| Ok(Box::new(SlicerApp::new(cc)))),
    )
}
