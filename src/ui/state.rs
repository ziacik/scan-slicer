use std::{path::PathBuf, sync::Arc};

use gdk_pixbuf::Pixbuf;
use image::DynamicImage;

use crate::{
    detection::PhotoRect,
    editor::{self, DragMode},
};

#[derive(Clone)]
pub(super) struct EditorSnapshot {
    pub(super) boxes: Vec<PhotoRect>,
    pub(super) selected: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Busy {
    None,
    Loading,
    Scanning,
    Detecting,
    Exporting,
}


#[derive(Clone, Copy)]
pub(super) struct ActiveDrag {
    pub(super) index: usize,
    pub(super) mode: DragMode,
    pub(super) start_rect: PhotoRect,
    pub(super) start_pointer: [f64; 2],
    pub(super) pointer: [f64; 2],
}

pub(super) struct AppState {
    pub(super) image: Option<Arc<DynamicImage>>,
    pub(super) preview: Option<Pixbuf>,
    pub(super) image_path: Option<PathBuf>,
    pub(super) boxes: Vec<PhotoRect>,
    pub(super) selected: Option<usize>,
    pub(super) selected_mode: Option<DragMode>,
    pub(super) drag: Option<ActiveDrag>,
    pub(super) pan_drag_start: Option<(f32, f32)>,
    pub(super) zoom: f32,
    pub(super) pan: (f32, f32),
    pub(super) hover: [f64; 2],
    pub(super) undo_stack: Vec<EditorSnapshot>,
    pub(super) redo_stack: Vec<EditorSnapshot>,
    pub(super) margin: u32,
    pub(super) busy: Busy,
    pub(super) status: String,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            image: None,
            preview: None,
            image_path: None,
            boxes: Vec::new(),
            selected: None,
            selected_mode: None,
            drag: None,
            pan_drag_start: None,
            zoom: 1.0,
            pan: (0.0, 0.0),
            hover: [0.0, 0.0],
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            margin: 0,
            busy: Busy::None,
            status: "Open an image or scan one to begin.".into(),
        }
    }
}

impl AppState {
    fn snapshot(&self) -> EditorSnapshot {
        EditorSnapshot {
            boxes: self.boxes.clone(),
            selected: self.selected,
        }
    }

    pub(super) fn push_undo(&mut self) {
        const HISTORY_LIMIT: usize = 100;
        if self.undo_stack.len() >= HISTORY_LIMIT {
            self.undo_stack.remove(0);
        }
        self.undo_stack.push(self.snapshot());
        self.redo_stack.clear();
    }

    pub(super) fn undo(&mut self) {
        let Some(snapshot) = self.undo_stack.pop() else {
            return;
        };
        self.redo_stack.push(self.snapshot());
        self.boxes = snapshot.boxes;
        self.selected = snapshot.selected.filter(|&i| i < self.boxes.len());
        self.selected_mode = None;
        self.drag = None;
        self.status = "Undid edit.".into();
    }

    pub(super) fn redo(&mut self) {
        let Some(snapshot) = self.redo_stack.pop() else {
            return;
        };
        self.undo_stack.push(self.snapshot());
        self.boxes = snapshot.boxes;
        self.selected = snapshot.selected.filter(|&i| i < self.boxes.len());
        self.selected_mode = None;
        self.drag = None;
        self.status = "Redid edit.".into();
    }

    pub(super) fn add_box(&mut self) {
        let Some(image) = self.image.as_ref() else {
            return;
        };
        let image_w = image.width();
        let image_h = image.height();

        self.push_undo();
        let w = (image_w / 3).max(100);
        let h = (image_h / 3).max(100);
        self.boxes.push(PhotoRect {
            x: (image_w.saturating_sub(w)) / 2,
            y: (image_h.saturating_sub(h)) / 2,
            w,
            h,
            corners: None,
        });
        self.selected = Some(self.boxes.len() - 1);
        self.status = "Added a frame.".into();
    }

    pub(super) fn remove_selected(&mut self) {
        let Some(index) = self.selected else {
            return;
        };
        if index >= self.boxes.len() {
            return;
        }

        self.push_undo();
        self.boxes.remove(index);
        self.selected = None;
        self.selected_mode = None;
        self.status = "Deleted frame.".into();
    }

    pub(super) fn nudge_selected(&mut self, dx: f32, dy: f32) {
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

        self.push_undo();
        let mut rect = self.boxes[index];
        editor::apply_drag(
            &mut rect,
            DragMode::Move,
            dx,
            dy,
            image_w,
            image_h,
        );
        self.boxes[index] = rect;
        self.status = "Moved frame.".into();
    }
}
