mod actions;
mod canvas;
mod state;

use std::{cell::RefCell, rc::Rc};

use adw::prelude::*;
use gtk::{gdk, glib};

use actions::{connect_actions, refresh_ui};
use canvas::connect_canvas;
use state::AppState;

#[derive(Clone)]
struct Ui {
    window: adw::ApplicationWindow,
    drawing: gtk::DrawingArea,
    empty_page: adw::StatusPage,
    empty_open_button: gtk::Button,
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

    let open_button = labeled_icon_button("document-open-symbolic", "Open", "Open scan");
    let detect_button = labeled_icon_button(
        "system-search-symbolic",
        "Detect Photos",
        "Detect photos in the scan",
    );
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
    empty_page.set_can_target(true);
    empty_page.add_css_class("canvas-empty-page");

    let empty_open_button = gtk::Button::with_label("Open Scan…");
    empty_open_button.add_css_class("suggested-action");
    empty_open_button.set_halign(gtk::Align::Center);
    empty_page.set_child(Some(&empty_open_button));

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
        empty_open_button,
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

fn labeled_icon_button(icon_name: &str, label: &str, tooltip: &str) -> gtk::Button {
    let content = gtk::Box::new(gtk::Orientation::Horizontal, 6);
    content.append(&gtk::Image::from_icon_name(icon_name));
    content.append(&gtk::Label::new(Some(label)));

    let button = gtk::Button::new();
    button.set_child(Some(&content));
    button.set_tooltip_text(Some(tooltip));
    button
}

fn install_css() {
    let provider = gtk::CssProvider::new();
    provider.load_from_string(
        "
        .scan-canvas {
            background: @view_bg_color;
        }

        .sidebar {
            padding: 0;
        }

        .canvas-empty-page,
        .canvas-empty-page label,
        .canvas-empty-page image {
            color: @view_fg_color;
        }

        .canvas-empty-page .dim-label {
            opacity: 0.72;
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
