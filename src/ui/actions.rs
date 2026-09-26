use std::{
    cell::RefCell,
    path::{Path, PathBuf},
    rc::Rc,
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

use adw::prelude::*;
use gdk_pixbuf::{Colorspace, Pixbuf};
use gtk::glib;
use image::DynamicImage;

use crate::{
    detection::detect_photos,
    editor,
    scanner::{self, ScannerDevice},
};

use super::{
    state::{AppState, Busy},
    Ui,
};

pub(super) fn connect_actions(state: &Rc<RefCell<AppState>>, ui: &Ui) {
    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.empty_open_button.clone();
        button.connect_clicked(move |_| {
            choose_and_load(state.clone(), ui.clone());
        });
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.empty_scan_button.clone();
        button.connect_clicked(move |_| {
            start_scan(state.clone(), ui.clone());
        });
    }

    {
        let state = state.clone();
        let ui = ui.clone();
        let button = ui.scan_button.clone();
        button.connect_clicked(move |_| {
            start_scan(state.clone(), ui.clone());
        });
    }

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

fn start_scan(state: Rc<RefCell<AppState>>, ui: Ui) {
    if state.borrow().busy != Busy::None {
        return;
    }

    {
        let mut st = state.borrow_mut();
        st.busy = Busy::Scanning;
        st.status = "Looking for scanners…".into();
    }
    refresh_ui(&state.borrow(), &ui);

    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(scanner::list_devices());
    });

    glib::timeout_add_local(Duration::from_millis(50), move || match rx.try_recv() {
        Ok(result) => {
            match result {
                Ok(devices) => {
                    {
                        let mut st = state.borrow_mut();
                        st.busy = Busy::None;
                        st.status = if devices.is_empty() {
                            "No SANE scanners found.".into()
                        } else {
                            format!(
                                "Found {} scanner{}.",
                                devices.len(),
                                if devices.len() == 1 { "" } else { "s" }
                            )
                        };
                    }
                    refresh_ui(&state.borrow(), &ui);

                    if devices.is_empty() {
                        ui.toast_overlay
                            .add_toast(adw::Toast::new("No scanners found"));
                    } else {
                        show_scan_dialog(state.clone(), ui.clone(), devices);
                    }
                }
                Err(error) => {
                    {
                        let mut st = state.borrow_mut();
                        st.busy = Busy::None;
                        st.status = error;
                    }
                    refresh_ui(&state.borrow(), &ui);
                    ui.toast_overlay
                        .add_toast(adw::Toast::new("Could not access SANE"));
                }
            }
            glib::ControlFlow::Break
        }
        Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
        Err(mpsc::TryRecvError::Disconnected) => {
            {
                let mut st = state.borrow_mut();
                st.busy = Busy::None;
                st.status = "Scanner discovery failed.".into();
            }
            refresh_ui(&state.borrow(), &ui);
            glib::ControlFlow::Break
        }
    });
}

fn show_scan_dialog(state: Rc<RefCell<AppState>>, ui: Ui, devices: Vec<ScannerDevice>) {
    let dialog = gtk::Dialog::builder()
        .title("Scan from Scanner")
        .transient_for(&ui.window)
        .modal(true)
        .resizable(false)
        .build();

    dialog.add_button("Cancel", gtk::ResponseType::Cancel);
    dialog.add_button("Scan", gtk::ResponseType::Accept);
    dialog.set_default_response(gtk::ResponseType::Accept);

    let content = dialog.content_area();
    content.set_spacing(12);
    content.set_margin_top(18);
    content.set_margin_bottom(18);
    content.set_margin_start(18);
    content.set_margin_end(18);

    let grid = gtk::Grid::builder()
        .column_spacing(12)
        .row_spacing(12)
        .build();

    let device_label = gtk::Label::new(Some("Scanner"));
    device_label.set_halign(gtk::Align::Start);
    let device_combo = gtk::ComboBoxText::new();
    device_combo.set_hexpand(true);
    for device in &devices {
        device_combo.append_text(&device.label);
    }
    device_combo.set_active(Some(0));

    let resolution_label = gtk::Label::new(Some("Resolution"));
    resolution_label.set_halign(gtk::Align::Start);
    let resolution_combo = gtk::ComboBoxText::new();
    resolution_combo.append_text("300 DPI");
    resolution_combo.append_text("600 DPI");
    resolution_combo.append_text("1200 DPI");
    resolution_combo.set_active(Some(1));

    grid.attach(&device_label, 0, 0, 1, 1);
    grid.attach(&device_combo, 1, 0, 1, 1);
    grid.attach(&resolution_label, 0, 1, 1, 1);
    grid.attach(&resolution_combo, 1, 1, 1, 1);

    let hint = gtk::Label::new(Some(
        "The scan is acquired through the system SANE backend and then detected automatically.",
    ));
    hint.set_wrap(true);
    hint.set_xalign(0.0);
    hint.add_css_class("dim-label");

    content.append(&grid);
    content.append(&hint);

    dialog.connect_response(move |dialog, response| {
        if response == gtk::ResponseType::Accept {
            let device_index = device_combo.active().unwrap_or(0) as usize;
            let resolution = match resolution_combo.active().unwrap_or(1) {
                0 => 300,
                2 => 1200,
                _ => 600,
            };

            if let Some(device) = devices.get(device_index).cloned() {
                dialog.close();
                perform_scan(state.clone(), ui.clone(), device, resolution);
                return;
            }
        }

        dialog.close();
    });

    dialog.present();
}

fn perform_scan(
    state: Rc<RefCell<AppState>>,
    ui: Ui,
    device: ScannerDevice,
    resolution: u32,
) {
    if state.borrow().busy != Busy::None {
        return;
    }

    {
        let mut st = state.borrow_mut();
        st.busy = Busy::Scanning;
        st.status = format!("Scanning at {resolution} DPI…");
    }
    refresh_ui(&state.borrow(), &ui);

    let device_id = device.id.clone();
    let device_label = device.label.clone();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let _ = tx.send(scanner::scan(&device_id, resolution));
    });

    glib::timeout_add_local(Duration::from_millis(80), move || match rx.try_recv() {
        Ok(result) => {
            match result {
                Ok(image) => {
                    let preview = make_preview(&image);
                    let image = Arc::new(image);
                    {
                        let mut st = state.borrow_mut();
                        st.image = Some(image);
                        st.preview = Some(preview);
                        st.image_path = None;
                        st.boxes.clear();
                        st.selected = None;
                        st.selected_mode = None;
                        st.drag = None;
                        st.zoom = 1.0;
                        st.pan = (0.0, 0.0);
                        st.undo_stack.clear();
                        st.redo_stack.clear();
                        st.busy = Busy::None;
                        st.status =
                            format!("Scanned from {device_label} at {resolution} DPI.");
                    }

                    refresh_ui(&state.borrow(), &ui);
                    ui.drawing.queue_draw();
                    ui.toast_overlay.add_toast(adw::Toast::new("Scan complete"));
                    start_detection(state.clone(), ui.clone());
                }
                Err(error) => {
                    {
                        let mut st = state.borrow_mut();
                        st.busy = Busy::None;
                        st.status = format!("Scan failed: {error}");
                    }
                    refresh_ui(&state.borrow(), &ui);
                    ui.toast_overlay.add_toast(adw::Toast::new("Scan failed"));
                }
            }
            glib::ControlFlow::Break
        }
        Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
        Err(mpsc::TryRecvError::Disconnected) => {
            {
                let mut st = state.borrow_mut();
                st.busy = Busy::None;
                st.status = "Scan failed.".into();
            }
            refresh_ui(&state.borrow(), &ui);
            glib::ControlFlow::Break
        }
    });
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

pub(super) fn refresh_ui(state: &AppState, ui: &Ui) {
    if let Some(image) = state.image.as_ref() {
        if let Some(path) = state.image_path.as_ref() {
            ui.source_row.set_title(
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("Loaded image"),
            );
        } else {
            ui.source_row.set_title("Scanned image");
        }
        ui.source_row
            .set_subtitle(&format!("{} × {} px", image.width(), image.height()));
    } else {
        ui.source_row.set_title("No image loaded");
        ui.source_row
            .set_subtitle("PNG, JPEG, TIFF or SANE scanner");
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
    ui.scan_button.set_sensitive(idle);
    ui.empty_scan_button.set_sensitive(idle);
    ui.open_button.set_sensitive(idle);
    ui.empty_open_button.set_sensitive(idle);
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
    let has_image = state.image.is_some();
    ui.empty_page.set_visible(!has_image);
    ui.drawing.set_sensitive(has_image);
}
