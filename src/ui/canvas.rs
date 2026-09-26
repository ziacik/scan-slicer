use std::{cell::RefCell, rc::Rc};

use adw::prelude::*;
use gdk_pixbuf::Pixbuf;
use gtk::{
    gdk,
    gdk::prelude::GdkCairoContextExt,
    glib,
};

use crate::{
    detection::PhotoRect,
    editor::{self, DragMode},
};

use super::{
    actions::refresh_ui,
    state::{ActiveDrag, AppState},
    Ui,
};

const HANDLE_RADIUS: f64 = 7.0;
const EDGE_HANDLE_RADIUS: f64 = 5.0;
const MAGNIFIER_SIZE: f64 = 150.0;
const MAGNIFIER_ZOOM: f64 = 5.0;

#[derive(Clone, Copy)]
struct ViewTransform {
    x: f64,
    y: f64,
    scale: f64,
}

pub(super) fn connect_canvas(state: &Rc<RefCell<AppState>>, ui: &Ui) {
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

                let cursor = if state.drag.is_some() || state.pan_drag_start.is_some() {
                    "grabbing"
                } else if let Some(transform) =
                    view_transform(&state, drawing.width(), drawing.height())
                {
                    match hit_test(&state, [x, y], transform) {
                        Some((index, DragMode::Move)) if state.selected == Some(index) => "move",
                        Some((index, DragMode::TopEdge | DragMode::BottomEdge))
                            if state.selected == Some(index) =>
                        {
                            "ns-resize"
                        }
                        Some((index, DragMode::LeftEdge | DragMode::RightEdge))
                            if state.selected == Some(index) =>
                        {
                            "ew-resize"
                        }
                        Some((index, DragMode::TopLeft | DragMode::BottomRight))
                            if state.selected == Some(index) =>
                        {
                            "nwse-resize"
                        }
                        Some((index, DragMode::TopRight | DragMode::BottomLeft))
                            if state.selected == Some(index) =>
                        {
                            "nesw-resize"
                        }
                        _ => "grab",
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
                        if st.selected == Some(index) {
                            st.push_undo();
                            let start_rect = st.boxes[index];
                            st.selected_mode = Some(mode);
                            st.drag = Some(ActiveDrag {
                                index,
                                mode,
                                start_rect,
                                start_pointer: [x, y],
                                pointer: [x, y],
                            });
                        } else {
                            // The first interaction only focuses the frame. Keep the
                            // current gesture as a canvas pan; a subsequent drag on the
                            // focused frame edits it.
                            st.selected = Some(index);
                            st.selected_mode = None;
                            st.pan_drag_start = Some(st.pan);
                        }
                    } else {
                        st.selected = None;
                        st.selected_mode = None;
                        st.pan_drag_start = Some(st.pan);
                    }
                    gesture.set_state(gtk::EventSequenceState::Claimed);
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

                if let Some(active) = st.drag {
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
                } else if let Some((start_x, start_y)) = st.pan_drag_start {
                    st.pan = (start_x + dx as f32, start_y + dy as f32);
                } else {
                    return;
                }

                drop(st);
                refresh_ui(&state.borrow(), &ui);
                ui.drawing.queue_draw();
            });
        }

        {
            let state = state.clone();
            let ui = ui.clone();
            gesture.connect_drag_end(move |_, _, _| {
                let had_interaction = {
                    let mut st = state.borrow_mut();
                    let adjusted = st.drag.is_some();
                    let panned = st.pan_drag_start.is_some();
                    st.drag = None;
                    st.pan_drag_start = None;
                    st.selected_mode = None;
                    if adjusted {
                        st.status = "Frame adjusted.".into();
                    }
                    adjusted || panned
                };

                if had_interaction {
                    refresh_ui(&state.borrow(), &ui);
                    ui.drawing.queue_draw();
                }
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
