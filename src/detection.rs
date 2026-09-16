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
    pub corners: Option<[[f32; 2]; 4]>,
}

impl PhotoRect {
    pub fn clamped(self, image_w: u32, image_h: u32) -> Self {
        let x = self.x.min(image_w);
        let y = self.y.min(image_h);
        let w = self.w.min(image_w.saturating_sub(x));
        let h = self.h.min(image_h.saturating_sub(y));
        Self {
            x,
            y,
            w,
            h,
            corners: self.corners,
        }
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
            corners: None,
        }]);
    }

    let scale = (width.max(height) as f32 / 1600.0).max(1.0);
    let small_w = ((width as f32 / scale).round() as u32).max(1);
    let small_h = ((height as f32 / scale).round() as u32).max(1);

    let small = image
        .resize_exact(
            small_w,
            small_h,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();

    let background = estimate_background(&small);
    let cutoff = threshold.max(5) as u16 * 3;

    let mut mask_bytes = vec![0u8; (small_w * small_h) as usize];
    for y in 0..small_h {
        for x in 0..small_w {
            let p = small.get_pixel(x, y).0;
            let distance = p[0].abs_diff(background[0]) as u16
                + p[1].abs_diff(background[1]) as u16
                + p[2].abs_diff(background[2]) as u16;

            if distance >= cutoff {
                mask_bytes[(y * small_w + x) as usize] = 255;
            }
        }
    }

    let mask = Mat::new_rows_cols_with_bytes::<u8>(
        small_h as i32,
        small_w as i32,
        &mask_bytes,
    )?;

    // Fill small gaps and scratches inside a photograph, then slightly expand
    // the foreground so fragmented parts of one photo become one component.
    let close_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(11, 11),
        Point::new(-1, -1),
    )?;
    let mut closed = Mat::default();
    imgproc::morphology_ex(
        &mask,
        &mut closed,
        imgproc::MORPH_CLOSE,
        &close_kernel,
        Point::new(-1, -1),
        2,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    let dilate_kernel = imgproc::get_structuring_element(
        imgproc::MORPH_RECT,
        Size::new(5, 5),
        Point::new(-1, -1),
    )?;
    let mut connected = Mat::default();
    imgproc::dilate(
        &closed,
        &mut connected,
        &dilate_kernel,
        Point::new(-1, -1),
        1,
        core::BORDER_CONSTANT,
        imgproc::morphology_default_border_value()?,
    )?;

    let mut contours: Vector<Vector<Point>> = Vector::new();
    imgproc::find_contours(
        &connected,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let scan_area = (small_w as f64) * (small_h as f64);
    let min_area = scan_area * 0.006;
    let max_area = scan_area * 0.75;

    let mut candidates = Vec::new();

    for contour in contours {
        let area = geometry::contour_area(&contour, false)?.abs();
        if area < min_area || area > max_area {
            continue;
        }

        let rotated = geometry::min_area_rect(&contour)?;
        let rw = rotated.size.width.abs();
        let rh = rotated.size.height.abs();
        if rw <= 1.0 || rh <= 1.0 {
            continue;
        }

        if rw < small_w as f32 * 0.08 || rh < small_h as f32 * 0.08 {
            continue;
        }

        let long = rw.max(rh);
        let short = rw.min(rh);
        if short / long < 0.22 {
            continue;
        }

        let rect_area = rw as f64 * rh as f64;
        if rect_area <= 0.0 || area / rect_area < 0.18 {
            continue;
        }

        let corners_small = rotated_corners(
            rotated.center.x,
            rotated.center.y,
            rw,
            rh,
            rotated.angle,
        );

        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        let mut corners = [[0.0f32; 2]; 4];

        for (i, p) in corners_small.iter().enumerate() {
            let x = (p[0] * scale).clamp(0.0, width as f32);
            let y = (p[1] * scale).clamp(0.0, height as f32);
            corners[i] = [x, y];
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        let bbox_w = max_x - min_x;
        let bbox_h = max_y - min_y;

        // Reject scan-wide/page-edge artefacts like the old bogus "photo 2".
        if bbox_w > width as f32 * 0.92 || bbox_h > height as f32 * 0.92 {
            continue;
        }

        let margin = margin as f32;
        let x = (min_x - margin).floor().max(0.0) as u32;
        let y = (min_y - margin).floor().max(0.0) as u32;
        let right = (max_x + margin).ceil().min(width as f32) as u32;
        let bottom = (max_y + margin).ceil().min(height as f32) as u32;

        let rect = PhotoRect {
            x,
            y,
            w: right.saturating_sub(x),
            h: bottom.saturating_sub(y),
            corners: Some(corners),
        };

        if candidates
            .iter()
            .any(|existing| overlap_ratio(*existing, rect) > 0.80)
        {
            continue;
        }

        candidates.push(rect);
    }

    candidates.sort_by_key(|r| (r.y / 40, r.x));
    Ok(candidates)
}

fn estimate_background(image: &image::RgbImage) -> [u8; 3] {
    let (w, h) = image.dimensions();
    let patch = (w.min(h) / 20).clamp(3, 30);
    let corners = [
        (0, 0),
        (w.saturating_sub(patch), 0),
        (0, h.saturating_sub(patch)),
        (w.saturating_sub(patch), h.saturating_sub(patch)),
    ];

    let mut samples = Vec::with_capacity((patch * patch * 4) as usize);
    for (start_x, start_y) in corners {
        for y in start_y..(start_y + patch).min(h) {
            for x in start_x..(start_x + patch).min(w) {
                samples.push(image.get_pixel(x, y).0);
            }
        }
    }

    let median = |channel: usize| {
        let mut values = samples.iter().map(|p| p[channel]).collect::<Vec<_>>();
        values.sort_unstable();
        values[values.len() / 2]
    };

    [median(0), median(1), median(2)]
}

fn rotated_corners(cx: f32, cy: f32, w: f32, h: f32, angle_deg: f32) -> [[f32; 2]; 4] {
    let angle = angle_deg.to_radians();
    let cos = angle.cos();
    let sin = angle.sin();
    let hw = w / 2.0;
    let hh = h / 2.0;

    [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)].map(|(x, y)| {
        [
            cx + x * cos - y * sin,
            cy + x * sin + y * cos,
        ]
    })
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
            corners: None,
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
            corners: None,
        };
        let b = PhotoRect {
            x: 10,
            y: 10,
            w: 80,
            h: 80,
            corners: None,
        };

        assert_eq!(overlap_ratio(a, b), 1.0);
    }
}
