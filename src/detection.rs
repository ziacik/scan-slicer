use std::collections::VecDeque;

use image::{DynamicImage, GenericImageView};

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
    let (width, height) = image.dimensions();
    if width < 20 || height < 20 {
        return vec![PhotoRect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        }];
    }

    let scale = (width.max(height) as f32 / 1200.0).max(1.0);
    let small_w = ((width as f32 / scale).round() as u32).max(1);
    let small_h = ((height as f32 / scale).round() as u32).max(1);

    let small = image
        .resize_exact(small_w, small_h, image::imageops::FilterType::Triangle)
        .to_rgb8();

    let background = estimate_background(&small);
    let mut mask = vec![false; (small_w * small_h) as usize];

    for y in 0..small_h {
        for x in 0..small_w {
            let p = small.get_pixel(x, y).0;
            let distance = color_distance(p, background);
            mask[(y * small_w + x) as usize] = distance >= threshold as u16 * 3;
        }
    }

    // Join nearby foreground areas so a bright patch inside a photo does not split it.
    for _ in 0..2 {
        mask = dilate(&mask, small_w, small_h, 2);
    }

    let min_area = ((small_w * small_h) as f32 * 0.008) as u32;
    let mut components = connected_components(&mask, small_w, small_h)
        .into_iter()
        .filter(|r| r.w * r.h >= min_area)
        .filter(|r| r.w > small_w / 12 && r.h > small_h / 12)
        .collect::<Vec<_>>();

    components.sort_by_key(|r| (r.y / 20, r.x));

    components
        .into_iter()
        .map(|r| {
            let mut x = (r.x as f32 * scale).floor() as i64 - margin as i64;
            let mut y = (r.y as f32 * scale).floor() as i64 - margin as i64;
            let mut right = ((r.x + r.w) as f32 * scale).ceil() as i64 + margin as i64;
            let mut bottom = ((r.y + r.h) as f32 * scale).ceil() as i64 + margin as i64;

            x = x.clamp(0, width as i64);
            y = y.clamp(0, height as i64);
            right = right.clamp(x, width as i64);
            bottom = bottom.clamp(y, height as i64);

            PhotoRect {
                x: x as u32,
                y: y as u32,
                w: (right - x) as u32,
                h: (bottom - y) as u32,
            }
        })
        .collect()
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

fn color_distance(a: [u8; 3], b: [u8; 3]) -> u16 {
    (a[0].abs_diff(b[0]) as u16)
        + (a[1].abs_diff(b[1]) as u16)
        + (a[2].abs_diff(b[2]) as u16)
}

fn dilate(mask: &[bool], w: u32, h: u32, radius: i32) -> Vec<bool> {
    let mut out = vec![false; mask.len()];

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let index = (y as u32 * w + x as u32) as usize;
            if !mask[index] {
                continue;
            }

            for yy in (y - radius).max(0)..=(y + radius).min(h as i32 - 1) {
                for xx in (x - radius).max(0)..=(x + radius).min(w as i32 - 1) {
                    out[(yy as u32 * w + xx as u32) as usize] = true;
                }
            }
        }
    }

    out
}

fn connected_components(mask: &[bool], w: u32, h: u32) -> Vec<PhotoRect> {
    let mut visited = vec![false; mask.len()];
    let mut result = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let start = (y * w + x) as usize;
            if visited[start] || !mask[start] {
                continue;
            }

            let mut queue = VecDeque::new();
            queue.push_back((x, y));
            visited[start] = true;

            let mut min_x = x;
            let mut max_x = x;
            let mut min_y = y;
            let mut max_y = y;

            while let Some((cx, cy)) = queue.pop_front() {
                min_x = min_x.min(cx);
                max_x = max_x.max(cx);
                min_y = min_y.min(cy);
                max_y = max_y.max(cy);

                let x0 = cx.saturating_sub(1);
                let x1 = (cx + 1).min(w - 1);
                let y0 = cy.saturating_sub(1);
                let y1 = (cy + 1).min(h - 1);

                for ny in y0..=y1 {
                    for nx in x0..=x1 {
                        let i = (ny * w + nx) as usize;
                        if !visited[i] && mask[i] {
                            visited[i] = true;
                            queue.push_back((nx, ny));
                        }
                    }
                }
            }

            result.push(PhotoRect {
                x: min_x,
                y: min_y,
                w: max_x - min_x + 1,
                h: max_y - min_y + 1,
            });
        }
    }

    result
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
}
