use image::{Rgba, RgbaImage};
use imageproc::geometric_transformations::{warp_into, Border, Interpolation, Projection};

use crate::detection::PhotoRect;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragMode {
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

pub fn apply_drag(
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
            let max_x = corners.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
            let max_y = corners.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);

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

pub fn rect_image_corners(rect: PhotoRect) -> [[f32; 2]; 4] {
    rect.corners.unwrap_or([
        [rect.x as f32, rect.y as f32],
        [(rect.x + rect.w) as f32, rect.y as f32],
        [(rect.x + rect.w) as f32, (rect.y + rect.h) as f32],
        [rect.x as f32, (rect.y + rect.h) as f32],
    ])
}

pub fn drag_target(rect: PhotoRect, mode: DragMode) -> Option<[f32; 2]> {
    let corners = rect_image_corners(rect);
    match mode {
        DragMode::Move => None,
        DragMode::TopLeft => Some(corners[0]),
        DragMode::TopRight => Some(corners[1]),
        DragMode::BottomRight => Some(corners[2]),
        DragMode::BottomLeft => Some(corners[3]),
        DragMode::TopEdge => Some(midpoint(corners[0], corners[1])),
        DragMode::RightEdge => Some(midpoint(corners[1], corners[2])),
        DragMode::BottomEdge => Some(midpoint(corners[2], corners[3])),
        DragMode::LeftEdge => Some(midpoint(corners[3], corners[0])),
    }
}

pub fn is_valid_quad(points: [[f32; 2]; 4]) -> bool {
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

pub fn point_in_quad(point: [f64; 2], points: [[f64; 2]; 4]) -> bool {
    let mut has_positive = false;
    let mut has_negative = false;

    for i in 0..4 {
        let a = points[i];
        let b = points[(i + 1) % 4];
        let cross =
            (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]);
        has_positive |= cross > 0.0;
        has_negative |= cross < 0.0;
        if has_positive && has_negative {
            return false;
        }
    }

    true
}

pub fn perspective_crop(
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

fn midpoint(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

fn set_rect_corners(rect: &mut PhotoRect, corners: [[f32; 2]; 4]) {
    let min_x = corners.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
    let min_y = corners.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
    let max_x = corners.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
    let max_y = corners.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);

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

fn point_distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
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
}
