use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
    sync::mpsc,
    thread,
    time::Duration,
};

use adw::prelude::*;
use gdk_pixbuf::{Colorspace, Pixbuf};
use gtk::{
    gdk,
    gdk::prelude::GdkCairoContextExt,
    glib,
};

use image::DynamicImage;

use crate::{
    detection::{PhotoRect, detect_photos},
    editor::{self, DragMode},
};

const HANDLE_RADIUS: f64 = 7.0;
const EDGE_HANDLE_RADIUS: f64 = 5.0;
const MAGNIFIER_SIZE: f64 = 150.0;
const MAGNIFIER_ZOOM: f64 = 5.0;

#[derive(Clone)]
struct EditorSnapshot {
    boxes: Vec<PhotoRect>,
    selected: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Busy {
    None,
    Loading,
    Detecting,
    Exporting,
}

#[derive(Clone, Copy)]
struct ViewTransform {
    x: f64,
    y: f64,
    scale: f64,
}

#[derive(Clone, Copy)]
struct ActiveDrag {
    index: usize,
    mode: DragMode,
    start_rect: PhotoRect,
    start_pointer: [f64; 2],
    pointer: [f64; 2],
}

struct AppState {
    image: Option<Arc<DynamicImage>>,
    preview: Option<Pixbuf>,
    image_path: Option<PathBuf>,
    boxes: Vec<PhotoRect>,
    selected: Option<usize>,
    selected_mode: Option<DragMode>,
    drag: Option<ActiveDrag>,
    pan_drag_start: Option<(f32, f32)>,
    zoom: f32,
    pan: (f32, f32),
    hover: [f64; 2],
    undo_stack: Vec<EditorSnapshot>,
    redo_stack: Vec<EditorSnapshot>,
    margin: u32,
    busy: Busy,
    status: String,
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
            status: "Open a scan to begin.".into(),
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
        self.redo_stack.push(self.snapshot());
        self.boxes = snapshot.boxes;
        self.selected = snapshot.selected.filter(|&i| i < self.boxes.len());
        self.selected_mode = None;
        self.drag = None;
        self.status = "Undid edit.".into();
    }

    fn redo(&mut self) {
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

    fn add_box(&mut self) {
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

    fn remove_selected(&mut self) {
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

#[derive(Clone)]
struct Ui {
    window: adw::ApplicationWindow,
    drawing: gtk::DrawingArea,
    empty_page: adw::StatusPage,
    source_row: adw::ActionRow,
    frames_row: adw::ActionRow,
    selected_row: adw::ActionRow,
    padding_spin: gtk::SpinButton,
    open_button: gtk::Button,
    detect_button: gtk::Button,
    export_button: gtk::Button,
    add_button: gtk::Button,
    delete_button: gtk::Button,
    undo_button: gtk::Button,
    redo_button: gtk::Button,
    fit_button: gtk::Button,
    spinner: gtk::Spinner,
    status_label: gtk::Label,
    zoom_label: gtk::Label,
    toast_overlay: adw::ToastOverlay,
}

pub fn run() -> glib::ExitCode {
    let app = adw::Application::builder()
        .application_id("com.github.ziacik.ScanSlicer")
        .build();

    app.connect_activate(build_ui);
    app.run()
}

fn build_ui(app: &adw::Application) {
    let state = Rc::new(RefCell::new(AppState::default()));

    let window = adw::ApplicationWindow::builder()
        .application(app)
        .title("Scan Slicer")
        .default_width(1200)
        .default_height(820)
        .width_request(780)
        .height_request(520)
        .build();

    let header = adw::HeaderBar::new();
    let title = adw::WindowTitle::new("Scan Slicer", "Photo sheet editor");
    header.set_title_widget(Some(&title));

    let open_button = icon_button("document-open-symbolic", "Open scan");
    let detect_button = icon_button("view-refresh-symbolic", "Detect photos");
    let undo_button = icon_button("edit-undo-symbolic", "Undo");
    let redo_button = icon_button("edit-redo-symbolic", "Redo");
    let fit_button = icon_button("zoom-fit-best-symbolic", "Fit image to window");
    let export_button = gtk::Button::with_label("Export");
    export_button.add_css_class("suggested-action");

    header.pack_start(&open_button);
    header.pack_start(&detect_button);
    header.pack_end(&export_button);
    header.pack_end(&fit_button);
    header.pack_end(&redo_button);
    header.pack_end(&undo_button);

    let source_group = adw::PreferencesGroup::builder().title("Source").build();
    let source_row = adw::ActionRow::builder()
        .title("No scan loaded")
        .subtitle("PNG, JPEG or TIFF")
        .build();
    source_group.add(&source_row);

    let frames_group = adw::PreferencesGroup::builder().title("Frames").build();
    let frames_row = adw::ActionRow::builder()
        .title("Detected frames")
        .subtitle("No frames")
        .build();

    let add_button = icon_button("list-add-symbolic", "Add frame");
    let delete_button = icon_button("edit-delete-symbolic", "Delete selected frame");
    delete_button.add_css_class("destructive-action");
    frames_row.add_suffix(&add_button);
    frames_row.add_suffix(&delete_button);
    frames_group.add(&frames_row);

    let selected_row = adw::ActionRow::builder()
        .title("Selected frame")
        .subtitle("")
        .build();
    selected_row.set_visible(false);
    frames_group.add(&selected_row);

    let crop_group = adw::PreferencesGroup::builder().title("Crop").build();
    let padding_row = adw::ActionRow::builder()
        .title("Padding")
        .subtitle("Extra pixels around detected photos")
        .build();
    let padding_spin = gtk::SpinButton::with_range(0.0, 100.0, 1.0);
    padding_spin.set_width_chars(4);
    padding_spin.set_valign(gtk::Align::Center);
    padding_row.add_suffix(&padding_spin);
    crop_group.add(&padding_row);

    let view_group = adw::PreferencesGroup::builder().title("View").build();
    let view_row = adw::ActionRow::builder()
        .title("Zoom")
        .subtitle("Scroll over the canvas to zoom")
        .build();
    let zoom_label = gtk::Label::new(Some("100%"));
    zoom_label.add_css_class("dim-label");
    view_row.add_suffix(&zoom_label);
    view_group.add(&view_row);

    let sidebar = gtk::Box::new(gtk::Orientation::Vertical, 18);
    sidebar.set_size_request(300, -1);
    sidebar.set_margin_top(18);
    sidebar.set_margin_bottom(14);
    sidebar.set_margin_start(16);
    sidebar.set_margin_end(16);
    sidebar.add_css_class("sidebar");
    sidebar.append(&source_group);
    sidebar.append(&frames_group);
    sidebar.append(&crop_group);
    sidebar.append(&view_group);

    let status_box = gtk::Box::new(gtk::Orientation::Horizontal, 8);
    status_box.set_valign(gtk::Align::End);
    status_box.set_vexpand(true);
    let spinner = gtk::Spinner::new();
    spinner.set_visible(false);
    let status_label = gtk::Label::new(Some("Open a scan to begin."));
    status_label.set_wrap(true);
    status_label.set_xalign(0.0);
    status_label.add_css_class("dim-label");
    status_box.append(&spinner);
    status_box.append(&status_label);
    sidebar.append(&status_box);

    let drawing = gtk::DrawingArea::new();
    drawing.set_hexpand(true);
    drawing.set_vexpand(true);
    drawing.set_focusable(true);
    drawing.add_css_class("scan-canvas");

    let empty_page = adw::StatusPage::builder()
        .icon_name("image-x-generic-symbolic")
        .title("Open a scan to begin")
        .description("PNG, JPEG or TIFF")
        .build();
    empty_page.set_can_target(false);

    let canvas_overlay = gtk::Overlay::new();
    canvas_overlay.set_child(Some(&drawing));
    canvas_overlay.add_overlay(&empty_page);

    let paned = gtk::Paned::new(gtk::Orientation::Horizontal);
    paned.set_start_child(Some(&sidebar));
    paned.set_end_child(Some(&canvas_overlay));
    paned.set_position(300);
    paned.set_resize_start_child(false);
    paned.set_shrink_start_child(false);
    paned.set_resize_end_child(true);
    paned.set_shrink_end_child(false);

    let toast_overlay = adw::ToastOverlay::new();
    toast_overlay.set_child(Some(&paned));

    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&header);
    toolbar.set_top_bar_style(adw::ToolbarStyle::Flat);
    toolbar.set_content(Some(&toast_overlay));
    window.set_content(Some(&toolbar));

    install_css();

    let ui = Ui {
        window,
        drawing,
        empty_page,
        source_row,
        frames_row,
        selected_row,
        padding_spin,
        open_button,
        detect_button,
        export_button,
        add_button,
        delete_button,
        undo_button,
        redo_button,
        fit_button,
        spinner,
        status_label,
        zoom_label,
        toast_overlay,
    };

    connect_canvas(&state, &ui);
    connect_actions(&state, &ui);
    refresh_ui(&state.borrow(), &ui);

    ui.window.present();
}

fn icon_button(icon_name: &str, tooltip: &str) -> gtk::Button {
    let button = gtk::Button::builder().icon_name(icon_name).build();
    button.set_tooltip_text(Some(tooltip));
    button
}

fn install_css() {
    let provider = gtk::CssProvider::new();
    provider.load_from_string(
        "
        .scan-canvas {
            background: #242424;
        }

        .sidebar {
            padding: 0;
        }
        ",
    );

    if let Some(display) = gdk::Display::default() {
        gtk::style_context_add_provider_for_display(
            &display,
            &provider,
            gtk::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}

fn connect_actions(state: &Rc<RefCell<AppState>>, ui: &Ui) {
    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.open_button.clone();
        button.connect_clicked(move |_| {
            choose_and_load(state.clone(), ui.clone());
        });
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.detect_button.clone();
        button.connect_clicked(move |_| {
            start_detection(state.clone(), ui.clone());
        });
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.export_button.clone();
        button.connect_clicked(move |_| {
            start_export(state.clone(), ui.clone());
        });
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.add_button.clone();
        button.connect_clicked(move |_| {
            state.borrow_mut().add_box();
            refresh_ui(&state.borrow(), &ui);
            ui.drawing.queue_draw();
        });
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.delete_button.clone();
        button.connect_clicked(move |_| {
            state.borrow_mut().remove_selected();
            refresh_ui(&state.borrow(), &ui);
            ui.drawing.queue_draw();
        });
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.undo_button.clone();
        button.connect_clicked(move |_| {
            state.borrow_mut().undo();
            refresh_ui(&state.borrow(), &ui);
            ui.drawing.queue_draw();
        });
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.redo_button.clone();
        button.connect_clicked(move |_| {
            state.borrow_mut().redo();
            refresh_ui(&state.borrow(), &ui);
            ui.drawing.queue_draw();
        });
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.fit_button.clone();
        button.connect_clicked(move |_| {
            {
                let mut st = state.borrow_mut();
                st.zoom = 1.0;
                st.pan = (0.0, 0.0);
            }
            refresh_ui(&state.borrow(), &ui);
            ui.drawing.queue_draw();
        });
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let spin_widget = ui.padding_spin.clone();
        spin_widget.connect_value_changed(move |spin| {
            {
                let mut st = state.borrow_mut();
                st.margin = spin.value_as_int().max(0) as u32;
                if st.image.is_some() {
                    st.status = "Padding changed — run detection again.".into();
                }
            }
            refresh_ui(&state.borrow(), &ui);
        });
    }
}

fn connect_canvas(state: &Rc<RefCell<AppState>>, ui: &Ui) {
    {
        let state = state.clone();
        ui.drawing.set_draw_func(move |_, cr, width, height| {
            draw_canvas(&state.borrow(), cr, width, height);
        });
    }

    {
        let state = state.clone();
        let drawing = ui.drawing.clone();
        let controller_target = drawing.clone();
        let ui = ui.clone();
        let motion = gtk::EventControllerMotion::new();
        motion.connect_motion(move |_, x, y| {
            {
                let mut state = state.borrow_mut();
                state.hover = [x, y];

                let cursor = if state.drag.is_some() {
                    "grabbing"
                } else if let Some(transform) =
                    view_transform(&state, drawing.width(), drawing.height())
                {
                    match hit_test(&state, [x, y], transform).map(|(_, mode)| mode) {
                        Some(DragMode::Move) => "grab",
                        Some(DragMode::TopEdge | DragMode::BottomEdge) => "ns-resize",
                        Some(DragMode::LeftEdge | DragMode::RightEdge) => "ew-resize",
                        Some(DragMode::TopLeft | DragMode::BottomRight) => "nwse-resize",
                        Some(DragMode::TopRight | DragMode::BottomLeft) => "nesw-resize",
                        None => "default",
                    }
                } else {
                    "default"
                };
                drawing.set_cursor_from_name(Some(cursor));
            }
            if state.borrow().drag.is_some() {
                drawing.queue_draw();
            }
            refresh_ui(&state.borrow(), &ui);
        });
        controller_target.add_controller(motion);
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let gesture = gtk::GestureDrag::new();
        gesture.set_button(1);

        {
            let state = state.clone();
            let ui = ui.clone();
            gesture.connect_drag_begin(move |gesture, x, y| {
                ui.drawing.grab_focus();
                {
                    let mut st = state.borrow_mut();
                    let Some(transform) =
                        view_transform(&st, ui.drawing.width(), ui.drawing.height())
                    else {
                        return;
                    };

                    if let Some((index, mode)) = hit_test(&st, [x, y], transform) {
                        st.push_undo();
                        let start_rect = st.boxes[index];
                        st.selected = Some(index);
                        st.selected_mode = Some(mode);
                        st.drag = Some(ActiveDrag {
                        index,
                        mode,
                        start_rect,
                        start_pointer: [x, y],
                        pointer: [x, y],
                    });
                        gesture.set_state(gtk::EventSequenceState::Claimed);
                    } else {
                        st.selected = None;
                        st.selected_mode = None;
                    }
                }

                refresh_ui(&state.borrow(), &ui);
                ui.drawing.queue_draw();
            });
        }

        {
            let state = state.clone();
            let ui = ui.clone();
            gesture.connect_drag_update(move |_, dx, dy| {
                let mut st = state.borrow_mut();
                let Some(active) = st.drag else {
                    return;
                };
                let Some(image) = st.image.as_ref().cloned() else {
                    return;
                };
                let Some(transform) =
                    view_transform(&st, ui.drawing.width(), ui.drawing.height())
                else {
                    return;
                };

                let mut rect = active.start_rect;
                editor::apply_drag(
                    &mut rect,
                    active.mode,
                    (dx / transform.scale) as f32,
                    (dy / transform.scale) as f32,
                    image.width(),
                    image.height(),
                );
                if active.index < st.boxes.len() {
                    st.boxes[active.index] = rect;
                }
                st.drag = Some(ActiveDrag {
                    pointer: [
                        active.start_pointer[0] + dx,
                        active.start_pointer[1] + dy,
                    ],
                    ..active
                });

                drop(st);
                refresh_ui(&state.borrow(), &ui);
                ui.drawing.queue_draw();
            });
        }

        {
            let state = state.clone();
            let ui = ui.clone();
            gesture.connect_drag_end(move |_, _, _| {
                {
                    let mut st = state.borrow_mut();
                    st.drag = None;
                    st.selected_mode = None;
                    st.status = "Frame adjusted.".into();
                }
                refresh_ui(&state.borrow(), &ui);
                ui.drawing.queue_draw();
            });
        }

        ui.drawing.add_controller(gesture);
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let pan = gtk::GestureDrag::new();
        pan.set_button(2);

        {
            let state = state.clone();
            pan.connect_drag_begin(move |_, _, _| {
                let mut state = state.borrow_mut();
                state.pan_drag_start = Some(state.pan);
            });
        }

        {
            let state = state.clone();
            let drawing = ui.drawing.clone();
            let ui = ui.clone();
            pan.connect_drag_update(move |_, dx, dy| {
                {
                    let mut st = state.borrow_mut();
                    if let Some((start_x, start_y)) = st.pan_drag_start {
                        st.pan = (start_x + dx as f32, start_y + dy as f32);
                    }
                }
                refresh_ui(&state.borrow(), &ui);
                drawing.queue_draw();
            });
        }

        {
            let state = state.clone();
            pan.connect_drag_end(move |_, _, _| {
                state.borrow_mut().pan_drag_start = None;
            });
        }

        ui.drawing.add_controller(pan);
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let controller_target = ui.drawing.clone();
        let scroll = gtk::EventControllerScroll::new(gtk::EventControllerScrollFlags::VERTICAL);
        scroll.connect_scroll(move |_, _, dy| {
            let mut st = state.borrow_mut();
            if st.image.is_none() {
                return glib::Propagation::Proceed;
            }

            let Some(before) = view_transform(&st, ui.drawing.width(), ui.drawing.height()) else {
                return glib::Propagation::Proceed;
            };
            let pointer = st.hover;
            let image_point = [
                (pointer[0] - before.x) / before.scale,
                (pointer[1] - before.y) / before.scale,
            ];

            let factor = (-dy * 0.14).exp() as f32;
            st.zoom = (st.zoom * factor).clamp(1.0, 8.0);

            if let Some(after) = view_transform(&st, ui.drawing.width(), ui.drawing.height()) {
                let screen_after = [
                    after.x + image_point[0] * after.scale,
                    after.y + image_point[1] * after.scale,
                ];
                st.pan.0 += (pointer[0] - screen_after[0]) as f32;
                st.pan.1 += (pointer[1] - screen_after[1]) as f32;
            }

            drop(st);
            refresh_ui(&state.borrow(), &ui);
            ui.drawing.queue_draw();
            glib::Propagation::Stop
        });
        controller_target.add_controller(scroll);
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let controller_target = ui.drawing.clone();
        let keys = gtk::EventControllerKey::new();
        keys.connect_key_pressed(move |_, key, _, modifiers| {
            let control = modifiers.contains(gdk::ModifierType::CONTROL_MASK);
            let shift = modifiers.contains(gdk::ModifierType::SHIFT_MASK);

            let handled = if control && key == gdk::Key::z {
                if shift {
                    state.borrow_mut().redo();
                } else {
                    state.borrow_mut().undo();
                }
                true
            } else if control && key == gdk::Key::y {
                state.borrow_mut().redo();
                true
            } else {
                let step = if shift { 10.0 } else { 1.0 };
                let delta = if key == gdk::Key::Left {
                    Some((-step, 0.0))
                } else if key == gdk::Key::Right {
                    Some((step, 0.0))
                } else if key == gdk::Key::Up {
                    Some((0.0, -step))
                } else if key == gdk::Key::Down {
                    Some((0.0, step))
                } else {
                    None
                };

                if let Some((dx, dy)) = delta {
                    state.borrow_mut().nudge_selected(dx, dy);
                    true
                } else {
                    false
                }
            };

            if handled {
                refresh_ui(&state.borrow(), &ui);
                ui.drawing.queue_draw();
                glib::Propagation::Stop
            } else {
                glib::Propagation::Proceed
            }
        });
        controller_target.add_controller(keys);
    }
}

fn choose_and_load(state: Rc<RefCell<AppState>>, ui: Ui) {
    if state.borrow().busy != Busy::None {
        return;
    }

    let Some(path) = rfd::FileDialog::new()
        .add_filter("Images", &["png", "jpg", "jpeg", "tif", "tiff"])
        .pick_file()
    else {
        return;
    };

    {
        let mut state = state.borrow_mut();
        state.busy = Busy::Loading;
        state.status = format!("Loading {}…", path.display());
    }
    refresh_ui(&state.borrow(), &ui);

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let result = image::open(&path).map_err(|error| error.to_string());
        let _ = tx.send((path, result));
    });

    glib::timeout_add_local(Duration::from_millis(40), move || match rx.try_recv() {
        Ok((path, result)) => {
            match result {
                Ok(image) => {
                    let preview = make_preview(&image);
                    let image = Arc::new(image);
                    {
                        let mut state = state.borrow_mut();
                        state.image = Some(image);
                        state.preview = Some(preview);
                        state.image_path = Some(path.clone());
                        state.boxes.clear();
                        state.selected = None;
                        state.selected_mode = None;
                        state.drag = None;
                        state.zoom = 1.0;
                        state.pan = (0.0, 0.0);
                        state.undo_stack.clear();
                        state.redo_stack.clear();
                        state.busy = Busy::None;
                        state.status = format!("Loaded {}", path.display());
                    }
                    refresh_ui(&state.borrow(), &ui);
                    ui.drawing.queue_draw();
                    start_detection(state.clone(), ui.clone());
                }
                Err(error) => {
                    {
                        let mut st = state.borrow_mut();
                        st.busy = Busy::None;
                        st.status = format!("Could not open image: {error}");
                    }
                    refresh_ui(&state.borrow(), &ui);
                    ui.toast_overlay
                        .add_toast(adw::Toast::new("Could not open the image"));
                }
            }
            glib::ControlFlow::Break
        }
        Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
        Err(mpsc::TryRecvError::Disconnected) => {
            {
                let mut st = state.borrow_mut();
                st.busy = Busy::None;
                st.status = "Image loading failed.".into();
            }
            refresh_ui(&state.borrow(), &ui);
            glib::ControlFlow::Break
        }
    });
}

fn start_detection(state: Rc<RefCell<AppState>>, ui: Ui) {
    if state.borrow().busy != Busy::None {
        return;
    }

    let (image, margin) = {
        let mut state = state.borrow_mut();
        let Some(image) = state.image.as_ref().cloned() else {
            return;
        };
        state.busy = Busy::Detecting;
        state.status = "Detecting photos…".into();
        (image, state.margin)
    };
    refresh_ui(&state.borrow(), &ui);

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(detect_photos(image.as_ref(), margin));
    });

    glib::timeout_add_local(Duration::from_millis(50), move || match rx.try_recv() {
        Ok(result) => {
            let toast_text;
            {
                let mut state = state.borrow_mut();
                state.boxes = result.boxes;
                state.selected = None;
                state.selected_mode = None;
                state.undo_stack.clear();
                state.redo_stack.clear();
                state.busy = Busy::None;

                state.status = match result.warning {
                    Some(warning) => {
                        toast_text = "Photo detection failed";
                        format!("{warning}")
                    }
                    None => {
                        toast_text = "Photos detected";
                        format!(
                            "Detected {} photo{} with {}.",
                            state.boxes.len(),
                            if state.boxes.len() == 1 { "" } else { "s" },
                            result.engine
                        )
                    }
                };
            }

            refresh_ui(&state.borrow(), &ui);
            ui.drawing.queue_draw();
            ui.toast_overlay.add_toast(adw::Toast::new(toast_text));
            glib::ControlFlow::Break
        }
        Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
        Err(mpsc::TryRecvError::Disconnected) => {
            {
                let mut st = state.borrow_mut();
                st.busy = Busy::None;
                st.status = "Photo detection failed.".into();
            }
            refresh_ui(&state.borrow(), &ui);
            glib::ControlFlow::Break
        }
    });
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

fn start_export(state: Rc<RefCell<AppState>>, ui: Ui) {
    if state.borrow().busy != Busy::None {
        return;
    }

    let (image, boxes, stem, default_dir) = {
        let state_ref = state.borrow();
        let Some(image) = state_ref.image.as_ref().cloned() else {
            return;
        };
        if state_ref.boxes.is_empty() {
            return;
        }

        let default_dir = state_ref
            .image_path
            .as_deref()
            .and_then(Path::parent)
            .map(Path::to_path_buf);
        let stem = state_ref
            .image_path
            .as_deref()
            .and_then(Path::file_stem)
            .and_then(|s| s.to_str())
            .unwrap_or("scan")
            .to_owned();

        (image, state_ref.boxes.clone(), stem, default_dir)
    };

    let mut dialog = rfd::FileDialog::new();
    if let Some(dir) = default_dir {
        dialog = dialog.set_directory(dir);
    }

    let Some(dir) = dialog.pick_folder() else {
        return;
    };

    {
        let mut state = state.borrow_mut();
        state.busy = Busy::Exporting;
        state.status = format!("Exporting {} photos…", boxes.len());
    }
    refresh_ui(&state.borrow(), &ui);

    let (tx, rx) = mpsc::channel();
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
                continue;
            }

            let crop = match rect.corners {
                Some(corners) => match editor::perspective_crop(&rgba, corners) {
                    Ok(crop) => crop,
                    Err(error) => {
                        let _ = tx.send(ExportEvent::Failed(format!(
                            "Export failed for frame {}: {error}",
                            index + 1
                        )));
                        return;
                    }
                },
                None => image::imageops::crop_imm(&rgba, rect.x, rect.y, rect.w, rect.h).to_image(),
            };

            let path = dir.join(format!("{stem}_{:02}.png", index + 1));
            if let Err(error) = crop.save(&path) {
                let _ = tx.send(ExportEvent::Failed(format!(
                    "Export failed at {}: {error}",
                    path.display()
                )));
                return;
            }

            exported += 1;
            let _ = tx.send(ExportEvent::Progress {
                completed: index + 1,
                total,
                exported,
            });
        }

        let _ = tx.send(ExportEvent::Finished { exported, dir });
    });

    glib::timeout_add_local(Duration::from_millis(50), move || {
        let mut finished = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                ExportEvent::Progress {
                    completed,
                    total,
                    exported,
                } => {
                    state.borrow_mut().status =
                        format!("Exporting {completed}/{total}… {exported} saved.");
                }
                ExportEvent::Finished { exported, dir } => {
                    let mut state = state.borrow_mut();
                    state.busy = Busy::None;
                    state.status = format!("Exported {exported} photos to {}", dir.display());
                    drop(state);
                    ui.toast_overlay
                        .add_toast(adw::Toast::new("Export finished"));
                    finished = true;
                }
                ExportEvent::Failed(error) => {
                    let mut state = state.borrow_mut();
                    state.busy = Busy::None;
                    state.status = error;
                    drop(state);
                    ui.toast_overlay
                        .add_toast(adw::Toast::new("Export failed"));
                    finished = true;
                }
            }
        }

        refresh_ui(&state.borrow(), &ui);
        if finished {
            glib::ControlFlow::Break
        } else {
            glib::ControlFlow::Continue
        }
    });
}

fn make_preview(image: &DynamicImage) -> Pixbuf {
    const MAX_UI_PREVIEW_DIM: u32 = 2400;
    let preview = image
        .thumbnail(MAX_UI_PREVIEW_DIM, MAX_UI_PREVIEW_DIM)
        .to_rgba8();
    let width = preview.width() as i32;
    let height = preview.height() as i32;
    let bytes = glib::Bytes::from_owned(preview.into_raw());
    Pixbuf::from_bytes(
        &bytes,
        Colorspace::Rgb,
        true,
        8,
        width,
        height,
        width * 4,
    )
}

fn refresh_ui(state: &AppState, ui: &Ui) {
    if let (Some(path), Some(image)) = (state.image_path.as_ref(), state.image.as_ref()) {
        ui.source_row.set_title(
            path.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Loaded scan"),
        );
        ui.source_row
            .set_subtitle(&format!("{} × {} px", image.width(), image.height()));
    } else {
        ui.source_row.set_title("No scan loaded");
        ui.source_row.set_subtitle("PNG, JPEG or TIFF");
    }

    ui.frames_row.set_subtitle(&format!(
        "{} frame{}",
        state.boxes.len(),
        if state.boxes.len() == 1 { "" } else { "s" }
    ));

    if let Some(index) = state.selected {
        if let Some(rect) = state.boxes.get(index) {
            ui.selected_row.set_visible(true);
            ui.selected_row
                .set_title(&format!("Frame {}", index + 1));
            ui.selected_row
                .set_subtitle(&format!("{} × {} px · x {}, y {}", rect.w, rect.h, rect.x, rect.y));
        } else {
            ui.selected_row.set_visible(false);
        }
    } else {
        ui.selected_row.set_visible(false);
    }

    ui.zoom_label
        .set_label(&format!("{}%", (state.zoom * 100.0).round() as u32));

    if ui.padding_spin.value_as_int().max(0) as u32 != state.margin {
        ui.padding_spin.set_value(state.margin as f64);
    }

    let idle = state.busy == Busy::None;
    ui.open_button.set_sensitive(idle);
    ui.detect_button
        .set_sensitive(idle && state.image.is_some());
    ui.export_button
        .set_sensitive(idle && state.image.is_some() && !state.boxes.is_empty());
    ui.add_button
        .set_sensitive(idle && state.image.is_some());
    ui.delete_button
        .set_sensitive(idle && state.selected.is_some());
    ui.undo_button
        .set_sensitive(idle && !state.undo_stack.is_empty());
    ui.redo_button
        .set_sensitive(idle && !state.redo_stack.is_empty());
    ui.fit_button.set_sensitive(
        state.image.is_some() && (state.zoom > 1.001 || state.pan.0.abs() > 0.5 || state.pan.1.abs() > 0.5),
    );

    let busy = state.busy != Busy::None;
    ui.spinner.set_visible(busy);
    ui.spinner.set_spinning(busy);
    ui.status_label.set_label(&state.status);
    ui.empty_page.set_visible(state.image.is_none());
}

fn view_transform(state: &AppState, width: i32, height: i32) -> Option<ViewTransform> {
    let image = state.image.as_ref()?;
    let source_w = image.width() as f64;
    let source_h = image.height() as f64;
    let viewport_w = (width as f64 - 64.0).max(1.0);
    let viewport_h = (height as f64 - 64.0).max(1.0);
    let base = (viewport_w / source_w)
        .min(viewport_h / source_h)
        .min(1.0)
        .max(0.01);
    let scale = base * state.zoom as f64;
    let display_w = source_w * scale;
    let display_h = source_h * scale;

    Some(ViewTransform {
        x: (width as f64 - display_w) * 0.5 + state.pan.0 as f64,
        y: (height as f64 - display_h) * 0.5 + state.pan.1 as f64,
        scale,
    })
}

fn draw_canvas(state: &AppState, cr: &gtk::cairo::Context, width: i32, height: i32) {
    cr.set_source_rgb(0.141, 0.141, 0.141);
    let _ = cr.paint();

    let (Some(image), Some(preview), Some(transform)) = (
        state.image.as_ref(),
        state.preview.as_ref(),
        view_transform(state, width, height),
    ) else {
        return;
    };

    let display_w = image.width() as f64 * transform.scale;
    let display_h = image.height() as f64 * transform.scale;

    cr.set_source_rgba(0.0, 0.0, 0.0, 0.35);
    cr.rectangle(
        transform.x - 8.0,
        transform.y - 8.0,
        display_w + 16.0,
        display_h + 16.0,
    );
    let _ = cr.fill();

    paint_preview(
        cr,
        preview,
        transform.x,
        transform.y,
        display_w,
        display_h,
    );

    for (index, rect) in state.boxes.iter().enumerate() {
        let points = screen_corners(*rect, transform);
        let selected = state.selected == Some(index);

        draw_polygon(
            cr,
            points,
            5.0,
            (0.0, 0.0, 0.0, if selected { 0.85 } else { 0.65 }),
        );
        draw_polygon(
            cr,
            points,
            if selected { 2.5 } else { 2.0 },
            if selected {
                (0.208, 0.518, 0.894, 1.0)
            } else {
                (0.45, 0.67, 1.0, 0.95)
            },
        );

        if selected {
            for (i, point) in points.iter().enumerate() {
                draw_handle(
                    cr,
                    *point,
                    HANDLE_RADIUS,
                    state.selected_mode
                        == Some(match i {
                            0 => DragMode::TopLeft,
                            1 => DragMode::TopRight,
                            2 => DragMode::BottomRight,
                            _ => DragMode::BottomLeft,
                        }),
                    true,
                );
            }

            let edges = edge_midpoints(points);
            for (i, point) in edges.iter().enumerate() {
                draw_handle(
                    cr,
                    *point,
                    EDGE_HANDLE_RADIUS,
                    state.selected_mode
                        == Some(match i {
                            0 => DragMode::TopEdge,
                            1 => DragMode::RightEdge,
                            2 => DragMode::BottomEdge,
                            _ => DragMode::LeftEdge,
                        }),
                    false,
                );
            }
        }
    }

    if let Some(active) = state.drag {
        if let Some(rect) = state.boxes.get(active.index) {
            if let Some(target) = editor::drag_target(*rect, active.mode) {
                draw_magnifier(
                    cr,
                    preview,
                    image.width(),
                    image.height(),
                    width,
                    height,
                    active.pointer,
                    target,
                    transform,
                );
            }
        }
    }
}

fn paint_preview(
    cr: &gtk::cairo::Context,
    pixbuf: &Pixbuf,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
) {
    let _ = cr.save();
    cr.translate(x, y);
    cr.scale(
        width / pixbuf.width().max(1) as f64,
        height / pixbuf.height().max(1) as f64,
    );
    cr.set_source_pixbuf(pixbuf, 0.0, 0.0);
    let _ = cr.paint();
    let _ = cr.restore();
}

fn draw_polygon(
    cr: &gtk::cairo::Context,
    points: [[f64; 2]; 4],
    width: f64,
    rgba: (f64, f64, f64, f64),
) {
    cr.new_path();
    cr.move_to(points[0][0], points[0][1]);
    for point in points.iter().skip(1) {
        cr.line_to(point[0], point[1]);
    }
    cr.close_path();
    cr.set_line_width(width);
    cr.set_line_join(gtk::cairo::LineJoin::Round);
    cr.set_source_rgba(rgba.0, rgba.1, rgba.2, rgba.3);
    let _ = cr.stroke();
}

fn draw_handle(
    cr: &gtk::cairo::Context,
    point: [f64; 2],
    radius: f64,
    active: bool,
    filled: bool,
) {
    cr.arc(point[0], point[1], radius, 0.0, std::f64::consts::TAU);
    if filled || active {
        cr.set_source_rgb(0.208, 0.518, 0.894);
    } else {
        cr.set_source_rgb(1.0, 1.0, 1.0);
    }
    let _ = cr.fill_preserve();
    cr.set_line_width(if active { 3.0 } else { 2.0 });
    cr.set_source_rgba(0.0, 0.0, 0.0, 0.85);
    let _ = cr.stroke();
}

fn draw_magnifier(
    cr: &gtk::cairo::Context,
    pixbuf: &Pixbuf,
    image_w: u32,
    image_h: u32,
    viewport_w: i32,
    viewport_h: i32,
    pointer: [f64; 2],
    target: [f32; 2],
    transform: ViewTransform,
) {
    let gap = 22.0;
    let mut x = pointer[0] + gap;
    let mut y = pointer[1] - MAGNIFIER_SIZE - gap;

    if x + MAGNIFIER_SIZE > viewport_w as f64 - 8.0 {
        x = pointer[0] - MAGNIFIER_SIZE - gap;
    }
    if y < 8.0 {
        y = pointer[1] + gap;
    }

    x = x.clamp(8.0, (viewport_w as f64 - MAGNIFIER_SIZE - 8.0).max(8.0));
    y = y.clamp(8.0, (viewport_h as f64 - MAGNIFIER_SIZE - 8.0).max(8.0));

    let sample_w = (MAGNIFIER_SIZE / (transform.scale * MAGNIFIER_ZOOM)).max(4.0);
    let sample_h = sample_w;
    let left = (target[0] as f64 - sample_w * 0.5)
        .clamp(0.0, (image_w as f64 - sample_w).max(0.0));
    let top = (target[1] as f64 - sample_h * 0.5)
        .clamp(0.0, (image_h as f64 - sample_h).max(0.0));

    cr.set_source_rgba(0.0, 0.0, 0.0, 0.82);
    rounded_rect(
        cr,
        x - 4.0,
        y - 4.0,
        MAGNIFIER_SIZE + 8.0,
        MAGNIFIER_SIZE + 8.0,
        12.0,
    );
    let _ = cr.fill();

    let _ = cr.save();
    rounded_rect(cr, x, y, MAGNIFIER_SIZE, MAGNIFIER_SIZE, 9.0);
    cr.clip();

    let px_per_image_x = pixbuf.width() as f64 / image_w.max(1) as f64;
    let px_per_image_y = pixbuf.height() as f64 / image_h.max(1) as f64;
    let preview_left = left * px_per_image_x;
    let preview_top = top * px_per_image_y;
    let preview_sample_w = sample_w * px_per_image_x;
    let preview_sample_h = sample_h * px_per_image_y;
    let sx = MAGNIFIER_SIZE / preview_sample_w.max(1.0);
    let sy = MAGNIFIER_SIZE / preview_sample_h.max(1.0);

    cr.translate(x - preview_left * sx, y - preview_top * sy);
    cr.scale(sx, sy);
    cr.set_source_pixbuf(pixbuf, 0.0, 0.0);
    let _ = cr.paint();
    let _ = cr.restore();

    cr.set_line_width(2.0);
    cr.set_source_rgb(1.0, 1.0, 1.0);
    rounded_rect(cr, x, y, MAGNIFIER_SIZE, MAGNIFIER_SIZE, 9.0);
    let _ = cr.stroke();

    let cx = x + MAGNIFIER_SIZE * 0.5;
    let cy = y + MAGNIFIER_SIZE * 0.5;
    cr.set_line_width(1.5);
    cr.move_to(cx - 12.0, cy);
    cr.line_to(cx + 12.0, cy);
    cr.move_to(cx, cy - 12.0);
    cr.line_to(cx, cy + 12.0);
    let _ = cr.stroke();
}

fn rounded_rect(
    cr: &gtk::cairo::Context,
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    radius: f64,
) {
    let r = radius.min(width * 0.5).min(height * 0.5);
    cr.new_sub_path();
    cr.arc(
        x + width - r,
        y + r,
        r,
        -std::f64::consts::FRAC_PI_2,
        0.0,
    );
    cr.arc(
        x + width - r,
        y + height - r,
        r,
        0.0,
        std::f64::consts::FRAC_PI_2,
    );
    cr.arc(
        x + r,
        y + height - r,
        r,
        std::f64::consts::FRAC_PI_2,
        std::f64::consts::PI,
    );
    cr.arc(
        x + r,
        y + r,
        r,
        std::f64::consts::PI,
        std::f64::consts::PI * 1.5,
    );
    cr.close_path();
}

fn screen_corners(rect: PhotoRect, transform: ViewTransform) -> [[f64; 2]; 4] {
    editor::rect_image_corners(rect).map(|[x, y]| {
        [
            transform.x + x as f64 * transform.scale,
            transform.y + y as f64 * transform.scale,
        ]
    })
}

fn edge_midpoints(points: [[f64; 2]; 4]) -> [[f64; 2]; 4] {
    [
        midpoint(points[0], points[1]),
        midpoint(points[1], points[2]),
        midpoint(points[2], points[3]),
        midpoint(points[3], points[0]),
    ]
}

fn midpoint(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn hit_test(
    state: &AppState,
    pointer: [f64; 2],
    transform: ViewTransform,
) -> Option<(usize, DragMode)> {
    for (index, rect) in state.boxes.iter().enumerate().rev() {
        let points = screen_corners(*rect, transform);

        let corners = [
            (points[0], DragMode::TopLeft),
            (points[1], DragMode::TopRight),
            (points[2], DragMode::BottomRight),
            (points[3], DragMode::BottomLeft),
        ];
        for (point, mode) in corners {
            if distance(point, pointer) <= HANDLE_RADIUS + 5.0 {
                return Some((index, mode));
            }
        }

        let edges = [
            (midpoint(points[0], points[1]), DragMode::TopEdge),
            (midpoint(points[1], points[2]), DragMode::RightEdge),
            (midpoint(points[2], points[3]), DragMode::BottomEdge),
            (midpoint(points[3], points[0]), DragMode::LeftEdge),
        ];
        for (point, mode) in edges {
            if distance(point, pointer) <= EDGE_HANDLE_RADIUS + 6.0 {
                return Some((index, mode));
            }
        }

        if editor::point_in_quad(pointer, points) {
            return Some((index, DragMode::Move));
        }
    }

    None
}
