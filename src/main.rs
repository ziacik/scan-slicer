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
const EDGE_HANDLE_RADIUS: f32 = 5.0;
const MAGNIFIER_SIZE: f32 = 150.0;
const MAGNIFIER_ZOOM: f32 = 5.0;

fn accent() -> Color32 {
    // GNOME/libadwaita blue.
    Color32::from_rgb(53, 132, 228)
}

fn accent_hover() -> Color32 {
    Color32::from_rgb(28, 113, 216)
}

fn panel_bg() -> Color32 {
    Color32::from_rgb(246, 245, 244)
}

fn surface() -> Color32 {
    Color32::WHITE
}

fn workspace() -> Color32 {
    Color32::from_rgb(36, 36, 36)
}

fn border() -> Color32 {
    Color32::from_rgb(218, 216, 214)
}

fn muted() -> Color32 {
    Color32::from_rgb(119, 118, 123)
}

fn canvas_muted() -> Color32 {
    Color32::from_rgb(190, 190, 190)
}

fn destructive() -> Color32 {
    Color32::from_rgb(192, 28, 40)
}

fn action_button(ui: &mut egui::Ui, label: &str, enabled: bool, primary: bool) -> bool {
    let fill = if !enabled {
        Color32::from_rgb(224, 222, 220)
    } else if primary {
        accent()
    } else {
        surface()
    };
    let stroke = if primary && enabled {
        Stroke::NONE
    } else {
        Stroke::new(1.0, border())
    };
    let text_color = if !enabled {
        Color32::from_rgb(146, 144, 141)
    } else if primary {
        Color32::WHITE
    } else {
        Color32::from_rgb(45, 45, 45)
    };

    ui.add_enabled(
        enabled,
        egui::Button::new(
            RichText::new(label)
                .size(13.5)
                .strong()
                .color(text_color),
        )
        .min_size(Vec2::new(ui.available_width(), 38.0))
        .fill(fill)
        .stroke(stroke)
        .corner_radius(8),
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
    TopEdge,
    RightEdge,
    BottomEdge,
    LeftEdge,
}

struct ActiveDrag {
    index: usize,
    mode: DragMode,
    start_pointer: Pos2,
    start_rect: PhotoRect,
}

#[derive(Clone)]
struct EditorSnapshot {
    boxes: Vec<PhotoRect>,
    selected: Option<usize>,
    selected_mode: Option<DragMode>,
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
    selected_mode: Option<DragMode>,
    drag: Option<ActiveDrag>,
    zoom: f32,
    pan: Vec2,
    undo_stack: Vec<EditorSnapshot>,
    redo_stack: Vec<EditorSnapshot>,
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

        let mut visuals = egui::Visuals::light();
        visuals.panel_fill = panel_bg();
        visuals.window_fill = panel_bg();
        visuals.extreme_bg_color = Color32::from_rgb(235, 233, 231);
        visuals.faint_bg_color = Color32::from_rgb(238, 237, 235);
        visuals.selection.bg_fill = accent();
        visuals.selection.stroke = Stroke::new(1.0, accent());
        visuals.hyperlink_color = accent_hover();
        visuals.widgets.inactive.bg_fill = Color32::from_rgb(238, 237, 235);
        visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(238, 237, 235);
        visuals.widgets.hovered.bg_fill = Color32::from_rgb(229, 227, 224);
        visuals.widgets.active.bg_fill = Color32::from_rgb(220, 218, 215);
        _cc.egui_ctx.set_visuals(visuals);

        let mut style = (*_cc.egui_ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(8.0, 8.0);
        style.spacing.button_padding = Vec2::new(12.0, 7.0);
        style.spacing.interact_size.y = 34.0;
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
            selected_mode: None,
            drag: None,
            zoom: 1.0,
            pan: Vec2::ZERO,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
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
                self.selected_mode = None;
                self.drag = None;
                self.zoom = 1.0;
                self.pan = Vec2::ZERO;
                self.undo_stack.clear();
                self.redo_stack.clear();
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
            self.selected_mode = None;
            self.undo_stack.clear();
            self.redo_stack.clear();
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

    fn snapshot(&self) -> EditorSnapshot {
        EditorSnapshot {
            boxes: self.boxes.clone(),
            selected: self.selected,
            selected_mode: self.selected_mode,
        }
    }

    fn restore_snapshot(&mut self, snapshot: EditorSnapshot) {
        self.boxes = snapshot.boxes;
        self.selected = snapshot.selected.filter(|&index| index < self.boxes.len());
        self.selected_mode = if self.selected.is_some() {
            snapshot.selected_mode
        } else {
            None
        };
        self.drag = None;
    }

    fn push_undo(&mut self) {
        const HISTORY_LIMIT: usize = 100;
        if self.undo_stack.len() >= HISTORY_LIMIT {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(self.snapshot());
        self.redo_stack.clear();
    }

    fn undo(&mut self) {
        let Some(snapshot) = self.undo_stack.pop() else {
            return;
        };
        let current = self.snapshot();
        self.redo_stack.push(current);
        self.restore_snapshot(snapshot);
        self.status = "Undid edit.".into();
    }

    fn redo(&mut self) {
        let Some(snapshot) = self.redo_stack.pop() else {
            return;
        };
        let current = self.snapshot();
        self.undo_stack.push(current);
        self.restore_snapshot(snapshot);
        self.status = "Redid edit.".into();
    }

    fn nudge_selected(&mut self, dx: f32, dy: f32) {
        let Some(index) = self.selected else {
            return;
        };
        let Some(image) = self.image.as_ref() else {
            return;
        };
        let image_w = image.width();
        let image_h = image.height();
        if index >= self.boxes.len() {
            return;
        }

        let mode = self.selected_mode.unwrap_or(DragMode::Move);
        self.push_undo();
        let mut rect = self.boxes[index];
        apply_drag(&mut rect, mode, dx, dy, image_w, image_h);
        self.boxes[index] = rect;
    }

    fn handle_shortcuts(&mut self, ctx: &egui::Context) {
        if self.exporting || self.loading || self.detecting || self.drag.is_some() {
            return;
        }

        let (undo, redo, nudge) = ctx.input(|i| {
            let command = i.modifiers.command || i.modifiers.ctrl;
            let redo = command
                && ((i.modifiers.shift && i.key_pressed(egui::Key::Z))
                    || i.key_pressed(egui::Key::Y));
            let undo = command && !i.modifiers.shift && i.key_pressed(egui::Key::Z);

            let step = if i.modifiers.shift { 10.0 } else { 1.0 };
            let nudge = if !command && !i.modifiers.alt {
                if i.key_pressed(egui::Key::ArrowLeft) {
                    Some((-step, 0.0))
                } else if i.key_pressed(egui::Key::ArrowRight) {
                    Some((step, 0.0))
                } else if i.key_pressed(egui::Key::ArrowUp) {
                    Some((0.0, -step))
                } else if i.key_pressed(egui::Key::ArrowDown) {
                    Some((0.0, step))
                } else {
                    None
                }
            } else {
                None
            };

            (undo, redo, nudge)
        });

        if redo {
            self.redo();
        } else if undo {
            self.undo();
        } else if let Some((dx, dy)) = nudge {
            self.nudge_selected(dx, dy);
        }
    }

    fn add_box(&mut self) {
        let Some(image) = self.image.as_ref() else {
            return;
        };
        let image_w = image.width();
        let image_h = image.height();

        self.push_undo();

        let w = (image_w / 3).max(100);
        let h = (image_h / 3).max(100);
        let rect = PhotoRect {
            x: (image_w.saturating_sub(w)) / 2,
            y: (image_h.saturating_sub(h)) / 2,
            w,
            h,
            corners: None,
        };
        self.boxes.push(rect);
        self.selected = Some(self.boxes.len() - 1);
        self.selected_mode = Some(DragMode::Move);
    }

    fn remove_selected(&mut self) {
        if let Some(index) = self.selected {
            if index < self.boxes.len() {
                self.push_undo();
                self.boxes.remove(index);
                self.selected = None;
                self.selected_mode = None;
            }
        }
    }

    fn canvas(&mut self, ui: &mut egui::Ui) {
        let available = ui.available_size();
        let (workspace_rect, _) = ui.allocate_exact_size(available, Sense::hover());
        let painter = ui.painter_at(workspace_rect);
        painter.rect_filled(workspace_rect, 0.0, workspace());

        let (Some(texture), Some(image)) = (
            self.texture.clone(),
            self.image.as_ref().map(Arc::clone),
        ) else {
            let center = workspace_rect.center();
            let icon = Rect::from_center_size(
                center - Vec2::new(0.0, 54.0),
                Vec2::new(52.0, 40.0),
            );
            painter.rect_stroke(
                icon,
                7.0,
                Stroke::new(2.0, Color32::from_rgb(150, 150, 150)),
                StrokeKind::Inside,
            );
            painter.circle_filled(
                icon.left_top() + Vec2::new(14.0, 13.0),
                4.0,
                Color32::from_rgb(150, 150, 150),
            );
            painter.line_segment(
                [
                    Pos2::new(icon.left() + 7.0, icon.bottom() - 9.0),
                    Pos2::new(icon.left() + 21.0, icon.top() + 22.0),
                ],
                Stroke::new(2.0, Color32::from_rgb(150, 150, 150)),
            );
            painter.line_segment(
                [
                    Pos2::new(icon.left() + 21.0, icon.top() + 22.0),
                    Pos2::new(icon.left() + 30.0, icon.bottom() - 14.0),
                ],
                Stroke::new(2.0, Color32::from_rgb(150, 150, 150)),
            );
            painter.line_segment(
                [
                    Pos2::new(icon.left() + 30.0, icon.bottom() - 14.0),
                    Pos2::new(icon.right() - 7.0, icon.bottom() - 9.0),
                ],
                Stroke::new(2.0, Color32::from_rgb(150, 150, 150)),
            );
            painter.text(
                center + Vec2::new(0.0, 4.0),
                egui::Align2::CENTER_CENTER,
                "Open a scan to begin",
                FontId::proportional(19.0),
                Color32::from_rgb(238, 238, 238),
            );
            painter.text(
                center + Vec2::new(0.0, 34.0),
                egui::Align2::CENTER_CENTER,
                "PNG, JPEG or TIFF",
                FontId::proportional(13.0),
                canvas_muted(),
            );
            return;
        };

        let source = Vec2::new(image.width() as f32, image.height() as f32);
        let padded = Vec2::new(
            (workspace_rect.width() - 64.0).max(1.0),
            (workspace_rect.height() - 64.0).max(1.0),
        );
        let base_scale = (padded.x / source.x)
            .min(padded.y / source.y)
            .min(1.0)
            .max(0.01);

        let (scroll_y, hover_pos, middle_down, pointer_delta) = ui.input(|i| {
            (
                i.raw_scroll_delta.y,
                i.pointer.hover_pos(),
                i.pointer.middle_down(),
                i.pointer.delta(),
            )
        });

        if let Some(pointer) = hover_pos.filter(|p| workspace_rect.contains(*p)) {
            if scroll_y.abs() > f32::EPSILON {
                let old_zoom = self.zoom;
                let new_zoom = (old_zoom * (scroll_y * 0.0025).exp()).clamp(1.0, 12.0);
                if (new_zoom - old_zoom).abs() > f32::EPSILON {
                    let old_scale = base_scale * old_zoom;
                    let old_display = source * old_scale;
                    let old_canvas =
                        Rect::from_center_size(workspace_rect.center() + self.pan, old_display);

                    let image_x = (pointer.x - old_canvas.left()) / old_scale;
                    let image_y = (pointer.y - old_canvas.top()) / old_scale;
                    let new_scale = base_scale * new_zoom;
                    let new_display = source * new_scale;
                    let new_left = pointer.x - image_x * new_scale;
                    let new_top = pointer.y - image_y * new_scale;
                    let new_center =
                        Pos2::new(new_left + new_display.x * 0.5, new_top + new_display.y * 0.5);

                    self.zoom = new_zoom;
                    self.pan = new_center - workspace_rect.center();
                }
            }

            if middle_down {
                self.pan += pointer_delta;
                ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            }
        }

        let scale = base_scale * self.zoom;
        let display = source * scale;
        clamp_pan(&mut self.pan, display, workspace_rect.size());
        let canvas = Rect::from_center_size(workspace_rect.center() + self.pan, display);

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
            workspace_rect,
            ui.id().with("scan_canvas"),
            Sense::click_and_drag(),
        );

        let mut magnifier = None;

        if let Some(pointer) = response.interact_pointer_pos() {
            let image_pos = |p: Pos2| -> Pos2 {
                Pos2::new(
                    ((p.x - canvas.left()) / scale).clamp(0.0, image.width() as f32),
                    ((p.y - canvas.top()) / scale).clamp(0.0, image.height() as f32),
                )
            };

            if response.drag_started_by(egui::PointerButton::Primary) {
                if let Some((index, mode)) = self.hit_test(pointer, canvas, scale) {
                    self.push_undo();
                    self.selected = Some(index);
                    self.selected_mode = Some(mode);
                    self.drag = Some(ActiveDrag {
                        index,
                        mode,
                        start_pointer: image_pos(pointer),
                        start_rect: self.boxes[index],
                    });
                } else {
                    self.selected = None;
                    self.selected_mode = None;
                }
            }

            if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(drag) = self.drag.as_ref() {
                    let now = image_pos(pointer);
                    let dx = now.x - drag.start_pointer.x;
                    let dy = now.y - drag.start_pointer.y;
                    let mut rect = drag.start_rect;
                    apply_drag(&mut rect, drag.mode, dx, dy, image.width(), image.height());
                    self.boxes[drag.index] = rect;
                    ui.ctx().set_cursor_icon(drag_cursor(drag.mode, true));

                    if drag.mode != DragMode::Move {
                        magnifier = drag_target(rect, drag.mode).map(|target| (pointer, target));
                    }
                }
            } else if let Some((_, mode)) = self.hit_test(pointer, canvas, scale) {
                ui.ctx().set_cursor_icon(drag_cursor(mode, false));
            }

            if response.drag_stopped_by(egui::PointerButton::Primary) {
                self.drag = None;
            }

            if response.clicked_by(egui::PointerButton::Primary) && self.drag.is_none() {
                if let Some((index, mode)) = self.hit_test(pointer, canvas, scale) {
                    self.selected = Some(index);
                    self.selected_mode = Some(mode);
                } else {
                    self.selected = None;
                    self.selected_mode = None;
                }
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
                let corner_modes = [
                    DragMode::TopLeft,
                    DragMode::TopRight,
                    DragMode::BottomRight,
                    DragMode::BottomLeft,
                ];
                for (point, mode) in points.into_iter().zip(corner_modes) {
                    painter.circle_filled(point, HANDLE_RADIUS, accent());
                    painter.circle_stroke(
                        point,
                        HANDLE_RADIUS,
                        Stroke::new(
                            if self.selected_mode == Some(mode) { 3.0 } else { 2.0 },
                            if self.selected_mode == Some(mode) {
                                Color32::WHITE
                            } else {
                                Color32::from_black_alpha(220)
                            },
                        ),
                    );
                }

                let edge_modes = [
                    DragMode::TopEdge,
                    DragMode::RightEdge,
                    DragMode::BottomEdge,
                    DragMode::LeftEdge,
                ];
                for (point, mode) in edge_midpoints(points).into_iter().zip(edge_modes) {
                    painter.circle_filled(
                        point,
                        EDGE_HANDLE_RADIUS,
                        if self.selected_mode == Some(mode) {
                            accent()
                        } else {
                            Color32::WHITE
                        },
                    );
                    painter.circle_stroke(
                        point,
                        EDGE_HANDLE_RADIUS,
                        Stroke::new(2.0, Color32::from_black_alpha(220)),
                    );
                }
            }
        }

        if let Some((pointer, target)) = magnifier {
            draw_magnifier(
                &painter,
                &texture,
                workspace_rect,
                canvas,
                scale,
                image.width(),
                image.height(),
                pointer,
                target,
            );
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

            let edge_handles = [
                (midpoint(points[0], points[1]), DragMode::TopEdge),
                (midpoint(points[1], points[2]), DragMode::RightEdge),
                (midpoint(points[2], points[3]), DragMode::BottomEdge),
                (midpoint(points[3], points[0]), DragMode::LeftEdge),
            ];
            for (point, mode) in edge_handles {
                if point.distance(pointer) <= EDGE_HANDLE_RADIUS + 6.0 {
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
        self.handle_shortcuts(ctx);
        if self.detecting || self.loading || self.exporting {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        egui::SidePanel::left("sidebar")
            .exact_width(300.0)
            .resizable(false)
            .frame(
                egui::Frame::new()
                    .fill(panel_bg())
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| {
                ui.add_space(4.0);
                ui.label(
                    RichText::new("Scan")
                        .size(18.0)
                        .strong()
                        .color(Color32::from_rgb(45, 45, 45)),
                );
                ui.add_space(6.0);

                egui::Frame::new()
                    .fill(surface())
                    .stroke(Stroke::new(1.0, border()))
                    .corner_radius(12)
                    .inner_margin(egui::Margin::same(13))
                    .show(ui, |ui| {
                        if let (Some(path), Some(image)) =
                            (self.image_path.as_ref(), self.image.as_ref())
                        {
                            ui.label(
                                RichText::new(
                                    path.file_name()
                                        .and_then(|name| name.to_str())
                                        .unwrap_or("Loaded scan"),
                                )
                                .size(14.0)
                                .strong()
                                .color(Color32::from_rgb(45, 45, 45)),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "{} × {} px",
                                    image.width(),
                                    image.height()
                                ))
                                .size(12.0)
                                .color(muted()),
                            );
                        } else {
                            ui.label(
                                RichText::new("No scan loaded")
                                    .size(14.0)
                                    .strong()
                                    .color(Color32::from_rgb(45, 45, 45)),
                            );
                            ui.label(
                                RichText::new("Choose an image to get started")
                                    .size(12.0)
                                    .color(muted()),
                            );
                        }

                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            let available = ui.available_width();
                            let gap = 8.0;
                            let width = (available - gap) / 2.0;

                            if ui
                                .add_enabled(
                                    !self.loading && !self.exporting,
                                    egui::Button::new(
                                        RichText::new("Open…")
                                            .size(13.0)
                                            .strong()
                                            .color(Color32::from_rgb(45, 45, 45)),
                                    )
                                    .min_size(Vec2::new(width, 34.0))
                                    .fill(Color32::from_rgb(238, 237, 235))
                                    .stroke(Stroke::NONE)
                                    .corner_radius(8),
                                )
                                .clicked()
                            {
                                self.open_image(ctx);
                            }

                            if ui
                                .add_enabled(
                                    self.image.is_some()
                                        && !self.detecting
                                        && !self.loading
                                        && !self.exporting,
                                    egui::Button::new(
                                        RichText::new(if self.detecting {
                                            "Detecting…"
                                        } else {
                                            "Detect"
                                        })
                                        .size(13.0)
                                        .strong()
                                        .color(Color32::from_rgb(45, 45, 45)),
                                    )
                                    .min_size(Vec2::new(width, 34.0))
                                    .fill(Color32::from_rgb(238, 237, 235))
                                    .stroke(Stroke::NONE)
                                    .corner_radius(8),
                                )
                                .clicked()
                            {
                                self.redetect();
                            }
                        });
                    });

                ui.add_space(18.0);
                ui.label(
                    RichText::new("Frames")
                        .size(18.0)
                        .strong()
                        .color(Color32::from_rgb(45, 45, 45)),
                );
                ui.add_space(6.0);

                egui::Frame::new()
                    .fill(surface())
                    .stroke(Stroke::new(1.0, border()))
                    .corner_radius(12)
                    .inner_margin(egui::Margin::same(10))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let width = (ui.available_width() - 18.0) / 4.0;

                            if ui
                                .add_enabled(
                                    self.image.is_some() && !self.exporting,
                                    egui::Button::new("+")
                                        .min_size(Vec2::new(width, 34.0))
                                        .fill(Color32::from_rgb(238, 237, 235))
                                        .stroke(Stroke::NONE)
                                        .corner_radius(8),
                                )
                                .on_hover_text("Add frame")
                                .clicked()
                            {
                                self.add_box();
                            }

                            if ui
                                .add_enabled(
                                    self.selected.is_some() && !self.exporting,
                                    egui::Button::new(
                                        RichText::new("−").color(destructive()).strong(),
                                    )
                                    .min_size(Vec2::new(width, 34.0))
                                    .fill(Color32::from_rgb(238, 237, 235))
                                    .stroke(Stroke::NONE)
                                    .corner_radius(8),
                                )
                                .on_hover_text("Delete selected frame")
                                .clicked()
                            {
                                self.remove_selected();
                            }

                            if ui
                                .add_enabled(
                                    !self.undo_stack.is_empty() && !self.exporting,
                                    egui::Button::new("↶")
                                        .min_size(Vec2::new(width, 34.0))
                                        .fill(Color32::from_rgb(238, 237, 235))
                                        .stroke(Stroke::NONE)
                                        .corner_radius(8),
                                )
                                .on_hover_text("Undo")
                                .clicked()
                            {
                                self.undo();
                            }

                            if ui
                                .add_enabled(
                                    !self.redo_stack.is_empty() && !self.exporting,
                                    egui::Button::new("↷")
                                        .min_size(Vec2::new(width, 34.0))
                                        .fill(Color32::from_rgb(238, 237, 235))
                                        .stroke(Stroke::NONE)
                                        .corner_radius(8),
                                )
                                .on_hover_text("Redo")
                                .clicked()
                            {
                                self.redo();
                            }
                        });

                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!(
                                    "{} frame{}",
                                    self.boxes.len(),
                                    if self.boxes.len() == 1 { "" } else { "s" }
                                ))
                                .size(12.0)
                                .color(muted()),
                            );
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                if ui
                                    .add_enabled(
                                        self.zoom > 1.001 || self.pan.length_sq() > 0.5,
                                        egui::Button::new(
                                            RichText::new("Reset view")
                                                .size(12.0)
                                                .color(accent_hover()),
                                        )
                                        .frame(false),
                                    )
                                    .clicked()
                                {
                                    self.zoom = 1.0;
                                    self.pan = Vec2::ZERO;
                                }

                                ui.label(
                                    RichText::new(format!(
                                        "{}%",
                                        (self.zoom * 100.0).round() as u32
                                    ))
                                    .size(12.0)
                                    .color(muted()),
                                );
                            });
                        });
                    });

                ui.add_space(18.0);
                ui.label(
                    RichText::new("Crop")
                        .size(18.0)
                        .strong()
                        .color(Color32::from_rgb(45, 45, 45)),
                );
                ui.add_space(6.0);

                egui::Frame::new()
                    .fill(surface())
                    .stroke(Stroke::new(1.0, border()))
                    .corner_radius(12)
                    .inner_margin(egui::Margin::same(13))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("Padding")
                                    .size(13.0)
                                    .strong()
                                    .color(Color32::from_rgb(45, 45, 45)),
                            );
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.label(
                                    RichText::new(format!("{} px", self.margin))
                                        .size(12.0)
                                        .color(muted()),
                                );
                            });
                        });
                        let margin_changed = ui
                            .add(egui::Slider::new(&mut self.margin, 0..=100).show_value(false))
                            .changed();
                        if margin_changed && self.image.is_some() {
                            self.status = "Crop padding changed — run detection again.".into();
                        }

                        if let Some(index) = self.selected {
                            if let Some(rect) = self.boxes.get(index) {
                                ui.add_space(8.0);
                                ui.separator();
                                ui.add_space(7.0);
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(format!("Frame {}", index + 1))
                                            .size(12.5)
                                            .strong()
                                            .color(Color32::from_rgb(45, 45, 45)),
                                    );
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        ui.label(
                                            RichText::new(format!("{} × {} px", rect.w, rect.h))
                                                .size(12.0)
                                                .color(muted()),
                                        );
                                    });
                                });
                            }
                        }
                    });

                ui.with_layout(Layout::bottom_up(Align::Min), |ui| {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new(
                            "Wheel zooms · middle-drag pans · arrows nudge · Ctrl+Z/Y undo/redo",
                        )
                        .size(10.5)
                        .color(muted()),
                    );
                    ui.add_space(8.0);
                    if action_button(
                        ui,
                        if self.exporting { "Exporting…" } else { "Export PNGs" },
                        !self.boxes.is_empty()
                            && !self.loading
                            && !self.detecting
                            && !self.exporting,
                        true,
                    ) {
                        self.export(ctx);
                    }
                });
            });

        egui::TopBottomPanel::bottom("status")
            .exact_height(34.0)
            .frame(
                egui::Frame::new()
                    .fill(panel_bg())
                    .inner_margin(egui::Margin::symmetric(10, 0)),
            )
            .show(ctx, |ui| {
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    if self.loading || self.detecting || self.exporting {
                        ui.add(egui::Spinner::new().size(14.0));
                    }
                    ui.label(RichText::new(&self.status).size(11.5).color(muted()));
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
        DragMode::TopLeft
        | DragMode::TopRight
        | DragMode::BottomRight
        | DragMode::BottomLeft => {
            let index = match mode {
                DragMode::TopLeft => 0,
                DragMode::TopRight => 1,
                DragMode::BottomRight => 2,
                DragMode::BottomLeft => 3,
                _ => unreachable!(),
            };
            corners[index][0] = (corners[index][0] + dx).clamp(0.0, image_w as f32);
            corners[index][1] = (corners[index][1] + dy).clamp(0.0, image_h as f32);

            if !is_valid_quad(corners) {
                return;
            }
        }
        DragMode::TopEdge
        | DragMode::RightEdge
        | DragMode::BottomEdge
        | DragMode::LeftEdge => {
            let (a_index, b_index) = match mode {
                DragMode::TopEdge => (0, 1),
                DragMode::RightEdge => (1, 2),
                DragMode::BottomEdge => (2, 3),
                DragMode::LeftEdge => (3, 0),
                _ => unreachable!(),
            };

            let a = corners[a_index];
            let b = corners[b_index];
            let edge_x = b[0] - a[0];
            let edge_y = b[1] - a[1];
            let edge_len = edge_x.hypot(edge_y);
            if edge_len < 1.0 {
                return;
            }

            let normal = [-edge_y / edge_len, edge_x / edge_len];
            let desired_offset = dx * normal[0] + dy * normal[1];
            let offset = clamp_edge_offset(
                a,
                b,
                normal,
                desired_offset,
                image_w as f32,
                image_h as f32,
            );

            corners[a_index][0] += normal[0] * offset;
            corners[a_index][1] += normal[1] * offset;
            corners[b_index][0] += normal[0] * offset;
            corners[b_index][1] += normal[1] * offset;

            if !is_valid_quad(corners) {
                return;
            }
        }
    }

    set_rect_corners(rect, corners);
}

fn clamp_pan(pan: &mut Vec2, display: Vec2, viewport: Vec2) {
    const MIN_VISIBLE: f32 = 80.0;

    if display.x <= viewport.x {
        pan.x = 0.0;
    } else {
        let max_pan = ((display.x + viewport.x) * 0.5 - MIN_VISIBLE).max(0.0);
        pan.x = pan.x.clamp(-max_pan, max_pan);
    }

    if display.y <= viewport.y {
        pan.y = 0.0;
    } else {
        let max_pan = ((display.y + viewport.y) * 0.5 - MIN_VISIBLE).max(0.0);
        pan.y = pan.y.clamp(-max_pan, max_pan);
    }
}

fn drag_target(rect: PhotoRect, mode: DragMode) -> Option<[f32; 2]> {
    let corners = rect_image_corners(rect);
    match mode {
        DragMode::Move => None,
        DragMode::TopLeft => Some(corners[0]),
        DragMode::TopRight => Some(corners[1]),
        DragMode::BottomRight => Some(corners[2]),
        DragMode::BottomLeft => Some(corners[3]),
        DragMode::TopEdge => Some(array_midpoint(corners[0], corners[1])),
        DragMode::RightEdge => Some(array_midpoint(corners[1], corners[2])),
        DragMode::BottomEdge => Some(array_midpoint(corners[2], corners[3])),
        DragMode::LeftEdge => Some(array_midpoint(corners[3], corners[0])),
    }
}

fn array_midpoint(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

fn draw_magnifier(
    painter: &egui::Painter,
    texture: &TextureHandle,
    workspace_rect: Rect,
    canvas: Rect,
    scale: f32,
    image_w: u32,
    image_h: u32,
    pointer: Pos2,
    target: [f32; 2],
) {
    let size = Vec2::splat(MAGNIFIER_SIZE);
    let gap = 22.0;

    let mut lens_min = pointer + Vec2::new(gap, -MAGNIFIER_SIZE - gap);
    if lens_min.x + MAGNIFIER_SIZE > workspace_rect.right() - 8.0 {
        lens_min.x = pointer.x - MAGNIFIER_SIZE - gap;
    }
    if lens_min.y < workspace_rect.top() + 8.0 {
        lens_min.y = pointer.y + gap;
    }
    lens_min.x = lens_min
        .x
        .clamp(workspace_rect.left() + 8.0, workspace_rect.right() - MAGNIFIER_SIZE - 8.0);
    lens_min.y = lens_min
        .y
        .clamp(workspace_rect.top() + 8.0, workspace_rect.bottom() - MAGNIFIER_SIZE - 8.0);

    let lens = Rect::from_min_size(lens_min, size);
    painter.rect_filled(lens.expand(4.0), 10.0, Color32::from_black_alpha(220));

    let sample_w = (MAGNIFIER_SIZE / (scale * MAGNIFIER_ZOOM)).max(4.0);
    let sample_h = sample_w;
    let half_w = sample_w * 0.5;
    let half_h = sample_h * 0.5;

    let max_x = image_w as f32;
    let max_y = image_h as f32;
    let left = (target[0] - half_w).clamp(0.0, (max_x - sample_w).max(0.0));
    let top = (target[1] - half_h).clamp(0.0, (max_y - sample_h).max(0.0));
    let right = (left + sample_w).min(max_x);
    let bottom = (top + sample_h).min(max_y);

    let uv = Rect::from_min_max(
        Pos2::new(left / max_x.max(1.0), top / max_y.max(1.0)),
        Pos2::new(right / max_x.max(1.0), bottom / max_y.max(1.0)),
    );
    painter.image(texture.id(), lens, uv, Color32::WHITE);
    painter.rect_stroke(
        lens,
        8.0,
        Stroke::new(2.0, Color32::WHITE),
        StrokeKind::Inside,
    );

    let target_x = lens.left() + ((target[0] - left) / (right - left).max(1.0)) * lens.width();
    let target_y = lens.top() + ((target[1] - top) / (bottom - top).max(1.0)) * lens.height();
    let cross = Pos2::new(target_x, target_y);
    painter.line_segment(
        [cross - Vec2::new(12.0, 0.0), cross + Vec2::new(12.0, 0.0)],
        Stroke::new(1.5, Color32::WHITE),
    );
    painter.line_segment(
        [cross - Vec2::new(0.0, 12.0), cross + Vec2::new(0.0, 12.0)],
        Stroke::new(1.5, Color32::WHITE),
    );
    painter.circle_stroke(cross, 4.0, Stroke::new(1.5, accent()));

    let source_target = Pos2::new(
        canvas.left() + target[0] * scale,
        canvas.top() + target[1] * scale,
    );
    painter.circle_stroke(source_target, HANDLE_RADIUS + 3.0, Stroke::new(1.5, Color32::WHITE));
}

fn drag_cursor(mode: DragMode, active: bool) -> CursorIcon {
    match mode {
        DragMode::Move => {
            if active {
                CursorIcon::Grabbing
            } else {
                CursorIcon::Grab
            }
        }
        DragMode::TopEdge | DragMode::BottomEdge => CursorIcon::ResizeVertical,
        DragMode::LeftEdge | DragMode::RightEdge => CursorIcon::ResizeHorizontal,
        DragMode::TopLeft | DragMode::BottomRight => CursorIcon::ResizeNwSe,
        DragMode::TopRight | DragMode::BottomLeft => CursorIcon::ResizeNeSw,
    }
}

fn midpoint(a: Pos2, b: Pos2) -> Pos2 {
    Pos2::new((a.x + b.x) * 0.5, (a.y + b.y) * 0.5)
}

fn edge_midpoints(points: [Pos2; 4]) -> [Pos2; 4] {
    [
        midpoint(points[0], points[1]),
        midpoint(points[1], points[2]),
        midpoint(points[2], points[3]),
        midpoint(points[3], points[0]),
    ]
}

fn clamp_edge_offset(
    a: [f32; 2],
    b: [f32; 2],
    normal: [f32; 2],
    desired: f32,
    image_w: f32,
    image_h: f32,
) -> f32 {
    let mut min_offset = f32::NEG_INFINITY;
    let mut max_offset = f32::INFINITY;

    for point in [a, b] {
        constrain_offset_axis(
            point[0],
            normal[0],
            image_w,
            &mut min_offset,
            &mut max_offset,
        );
        constrain_offset_axis(
            point[1],
            normal[1],
            image_h,
            &mut min_offset,
            &mut max_offset,
        );
    }

    desired.clamp(min_offset, max_offset)
}

fn constrain_offset_axis(
    position: f32,
    direction: f32,
    max_position: f32,
    min_offset: &mut f32,
    max_offset: &mut f32,
) {
    if direction.abs() < 1e-6 {
        return;
    }

    let first = -position / direction;
    let second = (max_position - position) / direction;
    *min_offset = (*min_offset).max(first.min(second));
    *max_offset = (*max_offset).min(first.max(second));
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
    fn dragging_an_edge_moves_both_endpoints_in_parallel() {
        let mut rect = PhotoRect {
            x: 10,
            y: 10,
            w: 120,
            h: 100,
            corners: Some([
                [20.0, 30.0],
                [120.0, 20.0],
                [130.0, 100.0],
                [30.0, 110.0],
            ]),
        };

        let before = rect.corners.unwrap();
        let before_vector = [
            before[1][0] - before[0][0],
            before[1][1] - before[0][1],
        ];

        apply_drag(&mut rect, DragMode::TopEdge, 5.0, -15.0, 200, 200);

        let after = rect.corners.unwrap();
        let after_vector = [
            after[1][0] - after[0][0],
            after[1][1] - after[0][1],
        ];
        let shift_a = [after[0][0] - before[0][0], after[0][1] - before[0][1]];
        let shift_b = [after[1][0] - before[1][0], after[1][1] - before[1][1]];

        assert!((after_vector[0] - before_vector[0]).abs() < 0.001);
        assert!((after_vector[1] - before_vector[1]).abs() < 0.001);
        assert!((shift_a[0] - shift_b[0]).abs() < 0.001);
        assert!((shift_a[1] - shift_b[1]).abs() < 0.001);
        assert!((shift_a[0] * before_vector[0] + shift_a[1] * before_vector[1]).abs() < 0.01);
    }

    #[test]
    fn magnifier_target_follows_dragged_edge_midpoint() {
        let rect = PhotoRect {
            x: 10,
            y: 10,
            w: 120,
            h: 100,
            corners: Some([
                [20.0, 30.0],
                [120.0, 20.0],
                [130.0, 100.0],
                [30.0, 110.0],
            ]),
        };

        assert_eq!(drag_target(rect, DragMode::TopLeft), Some([20.0, 30.0]));
        assert_eq!(drag_target(rect, DragMode::TopEdge), Some([70.0, 25.0]));
        assert_eq!(drag_target(rect, DragMode::Move), None);
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
