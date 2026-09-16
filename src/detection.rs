use image::{DynamicImage, GenericImageView};
use opencv::{
    core::{self, Mat, Point, Size, Vector},
    geometry, imgproc,
    prelude::*,
};

#[derive(Clone, Copy, Debug)]
pub struct PhotoRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl PhotoRect {
    pub fn clamped(self, image_w: u32, image_h: u32) -> Self {
        let x = self.x.min(image_w);
        let y = self.y.min(image_h);
        let w = self.w.min(image_w.saturating_sub(x));
        let h = self.h.min(image_h.saturating_sub(y));
        Self { x, y, w, h }
    }
}

pub fn detect_photos(image: &DynamicImage, threshold: u8, margin: u32) -> Vec<PhotoRect> {
    detect_photos_cv(image, threshold, margin).unwrap_or_default()
}

fn detect_photos_cv(
    image: &DynamicImage,
    threshold: u8,
    margin: u32,
) -> opencv::Result<Vec<PhotoRect>> {
    let (width, height) = image.dimensions();
    if width < 20 || height < 20 {
        return Ok(vec![PhotoRect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        }]);
    }

    // Detection does not need the full scan resolution. Working around 1600 px
    // keeps it fast while preserving enough detail for photo borders.
    let scale = (width.max(height) as f32 / 1600.0).max(1.0);
    let small_w = ((width as f32 / scale).round() as u32).max(1);
    let small_h = ((height as f32 / scale).round() as u32).max(1);

    let gray_image = image
        .resize_exact(
            small_w,
            small_h,
            image::imageops::FilterType::Triangle,
        )
        .to_luma8();

    let gray = Mat::new_rows_cols_with_bytes::<u8>(
        small_h as i32,
        small_w as i32,
        gray_image.as_raw(),
    )?;

    let mut blurred = Mat::default();
    imgproc::blur(
        &gray,
        &mut blurred,
        Size::new(5, 5),
        Point::new(-1, -1),
        core::BORDER_DEFAULT,
    )?;

    let low = threshold.max(5) as f64;
    let high = (low * 3.0).max(60.0);

    let mut edges = Mat::default();
    imgproc::canny(&blurred, &mut edges, low, high, 3, true)?;

    // Join small gaps in otherwise continuous photo borders.
    let kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(5, 5),
        Point::new(-1, -1),
    )?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &edges,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &kernel,
        Point::new(-1, -1),
        2,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    let mut contours: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours(
        &closed,
        &mut contours,
        imgproc::RETR_LIST,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let scan_area = (small_w as f64) * (small_h as f64);
    let min_area = scan_area * 0.012;
    let max_area = scan_area * 0.80;

    let mut candidates = Vec::new();

    for contour in contours {
        let area = geometry::contour_area(&contour, false)?.abs();
        if area < min_area || area > max_area {
            continue;
        }

        let perimeter = geometry::arc_length(&contour, true)?;
        if perimeter <= 0.0 {
            continue;
        }

        let mut approx: Vector<Point> = Vector::new();
        geometry::approx_poly_dp(&contour, &mut approx, perimeter * 0.025, true)?;

        // A real photo border should be a roughly rectangular convex shape.
        if approx.len() != 4 || !geometry::is_contour_convex(&approx)? {
            continue;
        }

        let rotated = geometry::min_area_rect(&approx)?;
        let rw = rotated.size.width.abs() as f64;
        let rh = rotated.size.height.abs() as f64;
        let rect_area = rw * rh;

        if rw < small_w as f64 * 0.08 || rh < small_h as f64 * 0.08 {
            continue;
        }

        // Reject very non-rectangular contours and thin scan-wide artefacts.
        if rect_area <= 0.0 || area / rect_area < 0.70 {
            continue;
        }

        let bounds = geometry::bounding_rect(&approx)?;
        if bounds.width as f64 > small_w as f64 * 0.95
            || bounds.height as f64 > small_h as f64 * 0.95
        {
            continue;
        }

        let mut x = (bounds.x as f32 * scale).floor() as i64 - margin as i64;
        let mut y = (bounds.y as f32 * scale).floor() as i64 - margin as i64;
        let mut right =
            ((bounds.x + bounds.width) as f32 * scale).ceil() as i64 + margin as i64;
        let mut bottom =
            ((bounds.y + bounds.height) as f32 * scale).ceil() as i64 + margin as i64;

        x = x.clamp(0, width as i64);
        y = y.clamp(0, height as i64);
        right = right.clamp(x, width as i64);
        bottom = bottom.clamp(y, height as i64);

        let rect = PhotoRect {
            x: x as u32,
            y: y as u32,
            w: (right - x) as u32,
            h: (bottom - y) as u32,
        };

        // findContours can return both sides of the same border. Keep one.
        if candidates.iter().any(|existing| overlap_ratio(*existing, rect) > 0.85) {
            continue;
        }

        candidates.push(rect);
    }

    candidates.sort_by_key(|r| (r.y / 40, r.x));
    Ok(candidates)
}

fn overlap_ratio(a: PhotoRect, b: PhotoRect) -> f32 {
    let left = a.x.max(b.x);
    let top = a.y.max(b.y);
    let right = (a.x + a.w).min(b.x + b.w);
    let bottom = (a.y + a.h).min(b.y + b.h);

    if right <= left || bottom <= top {
        return 0.0;
    }

    let intersection = (right - left) as f32 * (bottom - top) as f32;
    let smaller = (a.w * a.h).min(b.w * b.h) as f32;

    if smaller <= 0.0 {
        0.0
    } else {
        intersection / smaller
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_rect_to_image() {
        let rect = PhotoRect {
            x: 90,
            y: 80,
            w: 40,
            h: 50,
        }
        .clamped(100, 100);

        assert_eq!(rect.x, 90);
        assert_eq!(rect.y, 80);
        assert_eq!(rect.w, 10);
        assert_eq!(rect.h, 20);
    }

    #[test]
    fn overlap_uses_smaller_rect() {
        let a = PhotoRect {
            x: 0,
            y: 0,
            w: 100,
            h: 100,
        };
        let b = PhotoRect {
            x: 10,
            y: 10,
            w: 80,
            h: 80,
        };

        assert_eq!(overlap_ratio(a, b), 1.0);
    }
}
