use std::f32::consts::{FRAC_PI_2, PI};

use image::{DynamicImage, GenericImageView, RgbImage};
use opencv::{
    core::{Mat, Vec4i, Vector},
    imgproc,
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

#[derive(Clone, Copy, Debug)]
struct BackgroundModel {
    mean: [f32; 3],
    sigma: [f32; 3],
}

#[derive(Clone, Copy, Debug)]
struct DetectedLine {
    p: [f32; 2],
    d: [f32; 2],
    angle: f32,
    length: f32,
}

#[derive(Clone, Copy, Debug)]
struct LinePair {
    first: usize,
    second: usize,
    angle: f32,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    rect: PhotoRect,
    score: f32,
    area: f32,
}

pub fn detect_photos(image: &DynamicImage, threshold: u8, margin: u32) -> Vec<PhotoRect> {
    let classic = detect_photos_cv(image, threshold, margin).unwrap_or_default();
    if classic.is_empty() {
        return classic;
    }

    match crate::ai_detection::refine_photos_ai(image, &classic, margin) {
        Ok(photos) if !photos.is_empty() => photos,
        Ok(_) => classic,
        Err(error) => {
            eprintln!("MobileSAM refinement failed, using OpenCV candidates: {error:#}");
            classic
        }
    }
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

    // The paper deliberately works from a low-resolution preview. That makes
    // long object sides easier to group and keeps the hypothesis search small.
    let scale = (width.max(height) as f32 / 1200.0).max(1.0);
    let small_w = ((width as f32 / scale).round() as u32).max(1);
    let small_h = ((height as f32 / scale).round() as u32).max(1);
    let preview = image
        .resize_exact(
            small_w,
            small_h,
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();

    let background = estimate_background(&preview);
    let bg_mask = classify_background(&preview, background, threshold);
    let boundary = background_boundary(&preview, &bg_mask);

    let boundary_mat =
        Mat::new_rows_cols_with_bytes::<u8>(small_h as i32, small_w as i32, &boundary)?;

    let min_dim = small_w.min(small_h) as f64;
    let mut raw_lines: Vector<Vec4i> = Vector::new();
    imgproc::hough_lines_p(
        &boundary_mat,
        &mut raw_lines,
        1.0,
        std::f64::consts::PI / 360.0,
        18,
        (min_dim * 0.075).max(18.0),
        (min_dim * 0.035).max(6.0),
    )?;

    let mut lines = raw_lines
        .iter()
        .filter_map(|v| make_line(v[0], v[1], v[2], v[3]))
        .collect::<Vec<_>>();
    lines.sort_by(|a, b| b.length.total_cmp(&a.length));
    lines = dedupe_lines(lines);

    // Keep the combinatorics bounded on noisy scans. Long boundary segments
    // carry the most useful rectangle evidence.
    lines.truncate(48);

    let pairs = make_parallel_pairs(&lines, small_w, small_h);
    let mut candidates = make_rectangle_hypotheses(
        &lines,
        &pairs,
        &boundary,
        &bg_mask,
        small_w,
        small_h,
        scale,
        width,
        height,
        margin,
    );

    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

    // Same spirit as the paper: accept the strongest hypotheses first and
    // suppress hypotheses that substantially cover an already accepted one.
    let mut accepted: Vec<Candidate> = Vec::new();
    for candidate in candidates {
        if accepted.iter().any(|other| {
            overlap_area_ratio(candidate.rect, other.rect) > 0.30
                || overlap_area_ratio(other.rect, candidate.rect) > 0.30
        }) {
            continue;
        }
        accepted.push(candidate);
    }

    let mut result = accepted.into_iter().map(|c| c.rect).collect::<Vec<_>>();
    result.sort_by_key(|r| (r.y / 40, r.x));
    Ok(result)
}

fn estimate_background(image: &RgbImage) -> BackgroundModel {
    const BINS: usize = 16;
    const BIN_COUNT: usize = BINS * BINS * BINS;

    let (w, h) = image.dimensions();
    let mut votes = vec![0u32; BIN_COUNT];

    // Guerzhoy/Zhou exploit the fact that background colour appears in long,
    // contiguous, nearly uniform row/column segments. Do the same here.
    for y in 0..h {
        vote_uniform_run(
            (0..w).map(|x| image.get_pixel(x, y).0),
            w,
            &mut votes,
        );
    }
    for x in 0..w {
        vote_uniform_run(
            (0..h).map(|y| image.get_pixel(x, y).0),
            h,
            &mut votes,
        );
    }

    let best = votes
        .iter()
        .enumerate()
        .max_by_key(|(_, count)| *count)
        .map(|(index, _)| index)
        .unwrap_or(BIN_COUNT - 1);

    let br = ((best / (BINS * BINS)) % BINS) as i32 * 16 + 8;
    let bg = ((best / BINS) % BINS) as i32 * 16 + 8;
    let bb = (best % BINS) as i32 * 16 + 8;

    let mut count = 0f32;
    let mut sum = [0f32; 3];
    let mut sum_sq = [0f32; 3];

    // Refine the quantized hypothesis using pixels around that colour.
    for pixel in image.pixels() {
        let p = pixel.0;
        if (p[0] as i32 - br).abs() <= 28
            && (p[1] as i32 - bg).abs() <= 28
            && (p[2] as i32 - bb).abs() <= 28
        {
            count += 1.0;
            for c in 0..3 {
                let value = p[c] as f32;
                sum[c] += value;
                sum_sq[c] += value * value;
            }
        }
    }

    if count < 16.0 {
        return BackgroundModel {
            mean: [br as f32, bg as f32, bb as f32],
            sigma: [4.0; 3],
        };
    }

    let mut mean = [0f32; 3];
    let mut sigma = [0f32; 3];
    for c in 0..3 {
        mean[c] = sum[c] / count;
        let variance = (sum_sq[c] / count - mean[c] * mean[c]).max(0.0);
        sigma[c] = variance.sqrt().clamp(2.0, 20.0);
    }

    BackgroundModel { mean, sigma }
}

fn vote_uniform_run<I>(pixels: I, line_len: u32, votes: &mut [u32])
where
    I: Iterator<Item = [u8; 3]>,
{
    let min_run = (line_len / 14).max(6) as usize;
    let mut run_len = 0usize;
    let mut mean = [0f32; 3];

    let finish_run = |run_len: usize, mean: [f32; 3], votes: &mut [u32]| {
        if run_len < min_run {
            return;
        }
        let r = ((mean[0].round() as usize).min(255)) / 16;
        let g = ((mean[1].round() as usize).min(255)) / 16;
        let b = ((mean[2].round() as usize).min(255)) / 16;
        let index = r * 16 * 16 + g * 16 + b;
        // Longer homogeneous runs are stronger evidence for background.
        votes[index] = votes[index].saturating_add(run_len as u32);
    };

    for p in pixels {
        if run_len == 0 {
            mean = [p[0] as f32, p[1] as f32, p[2] as f32];
            run_len = 1;
            continue;
        }

        let max_diff = (0..3)
            .map(|c| (p[c] as f32 - mean[c]).abs())
            .fold(0.0f32, f32::max);

        if max_diff <= 10.0 {
            run_len += 1;
            let n = run_len as f32;
            for c in 0..3 {
                mean[c] += (p[c] as f32 - mean[c]) / n;
            }
        } else {
            finish_run(run_len, mean, votes);
            mean = [p[0] as f32, p[1] as f32, p[2] as f32];
            run_len = 1;
        }
    }

    finish_run(run_len, mean, votes);
}

fn classify_background(
    image: &RgbImage,
    model: BackgroundModel,
    threshold: u8,
) -> Vec<bool> {
    let (w, h) = image.dimensions();
    let sensitivity = (threshold as f32 / 22.0).clamp(0.45, 3.5);
    let mut result = vec![false; (w * h) as usize];

    for y in 0..h {
        for x in 0..w {
            let p = image.get_pixel(x, y).0;
            let mut normalized = 0f32;
            for c in 0..3 {
                // A noise floor prevents perfectly flat scanner backgrounds
                // from becoming absurdly sensitive to tiny RGB differences.
                let tolerance = (model.sigma[c] * 3.0 + 7.0) * sensitivity;
                normalized = normalized.max((p[c] as f32 - model.mean[c]).abs() / tolerance);
            }
            result[(y * w + x) as usize] = normalized <= 1.0;
        }
    }

    result
}

fn background_boundary(image: &RgbImage, bg: &[bool]) -> Vec<u8> {
    let (w, h) = image.dimensions();
    let mut boundary = vec![0u8; (w * h) as usize];

    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let here = bg[i];

            let mut transition = false;
            if x + 1 < w {
                transition |= here != bg[(y * w + x + 1) as usize];
            }
            if y + 1 < h {
                transition |= here != bg[((y + 1) * w + x) as usize];
            }

            if transition {
                boundary[i] = 255;
                if x + 1 < w {
                    boundary[(y * w + x + 1) as usize] = 255;
                }
                if y + 1 < h {
                    boundary[((y + 1) * w + x) as usize] = 255;
                }
            }
        }
    }

    boundary
}

fn make_line(x1: i32, y1: i32, x2: i32, y2: i32) -> Option<DetectedLine> {
    let dx = (x2 - x1) as f32;
    let dy = (y2 - y1) as f32;
    let length = (dx * dx + dy * dy).sqrt();
    if length < 1.0 {
        return None;
    }

    let mut angle = dy.atan2(dx);
    if angle < 0.0 {
        angle += PI;
    }
    if angle >= PI {
        angle -= PI;
    }

    Some(DetectedLine {
        p: [x1 as f32, y1 as f32],
        d: [dx / length, dy / length],
        angle,
        length,
    })
}

fn dedupe_lines(lines: Vec<DetectedLine>) -> Vec<DetectedLine> {
    let mut kept: Vec<DetectedLine> = Vec::new();

    'candidate: for line in lines {
        for other in &kept {
            if angle_distance(line.angle, other.angle) < 3.0_f32.to_radians()
                && point_line_distance(line_midpoint(line), *other) < 5.0
            {
                continue 'candidate;
            }
        }
        kept.push(line);
    }

    kept
}

fn make_parallel_pairs(lines: &[DetectedLine], w: u32, h: u32) -> Vec<LinePair> {
    let min_sep = w.min(h) as f32 * 0.07;
    let max_sep = (w * w + h * h) as f32;
    let max_sep = max_sep.sqrt() * 0.80;
    let mut pairs = Vec::new();

    for i in 0..lines.len() {
        for j in (i + 1)..lines.len() {
            if angle_distance(lines[i].angle, lines[j].angle) > 7.0_f32.to_radians() {
                continue;
            }

            let separation = point_line_distance(line_midpoint(lines[j]), lines[i]);
            if separation < min_sep || separation > max_sep {
                continue;
            }

            pairs.push(LinePair {
                first: i,
                second: j,
                angle: average_line_angle(lines[i].angle, lines[j].angle),
            });
        }
    }

    pairs
}

#[allow(clippy::too_many_arguments)]
fn make_rectangle_hypotheses(
    lines: &[DetectedLine],
    pairs: &[LinePair],
    boundary: &[u8],
    bg_mask: &[bool],
    small_w: u32,
    small_h: u32,
    scale: f32,
    width: u32,
    height: u32,
    margin: u32,
) -> Vec<Candidate> {
    let scan_area = (small_w * small_h) as f32;
    let mut result = Vec::new();

    for a_index in 0..pairs.len() {
        for b_index in (a_index + 1)..pairs.len() {
            let a = pairs[a_index];
            let b = pairs[b_index];

            if a.first == b.first
                || a.first == b.second
                || a.second == b.first
                || a.second == b.second
            {
                continue;
            }

            let angle = angle_distance(a.angle, b.angle);
            if (angle - FRAC_PI_2).abs() > 13.0_f32.to_radians() {
                continue;
            }

            let Some(p00) = line_intersection(lines[a.first], lines[b.first]) else {
                continue;
            };
            let Some(p10) = line_intersection(lines[a.second], lines[b.first]) else {
                continue;
            };
            let Some(p11) = line_intersection(lines[a.second], lines[b.second]) else {
                continue;
            };
            let Some(p01) = line_intersection(lines[a.first], lines[b.second]) else {
                continue;
            };
            let corners = [p00, p10, p11, p01];

            if corners.iter().any(|p| {
                p[0] < -(small_w as f32) * 0.03
                    || p[1] < -(small_h as f32) * 0.03
                    || p[0] > small_w as f32 * 1.03
                    || p[1] > small_h as f32 * 1.03
            }) {
                continue;
            }

            let side_a = distance(p00, p10);
            let side_b = distance(p10, p11);
            let min_side = small_w.min(small_h) as f32 * 0.075;
            if side_a < min_side || side_b < min_side {
                continue;
            }

            // Reject implausibly thin rectangle hypotheses. This mainly removes
            // accidental combinations of scanner/page edges with one photo edge.
            let aspect = side_a.min(side_b) / side_a.max(side_b);
            if aspect < 0.28 {
                continue;
            }

            let area = side_a * side_b;
            if area < scan_area * 0.006 || area > scan_area * 0.72 {
                continue;
            }

            // A rectangle hugging three scanner boundaries is almost certainly
            // the scan/page frame rather than one of several photos.
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
            let border = small_w.min(small_h) as f32 * 0.025;
            let touched_edges = usize::from(min_x <= border)
                + usize::from(min_y <= border)
                + usize::from(max_x >= small_w as f32 - border)
                + usize::from(max_y >= small_h as f32 - border);
            if touched_edges >= 3 {
                continue;
            }

            let supports = [
                side_support(boundary, small_w, small_h, p00, p10),
                side_support(boundary, small_w, small_h, p10, p11),
                side_support(boundary, small_w, small_h, p11, p01),
                side_support(boundary, small_w, small_h, p01, p00),
            ];
            let average_support = supports.iter().sum::<f32>() / 4.0;
            let strong_sides = supports.iter().filter(|&&s| s >= 0.34).count();

            // Allow one damaged/faded side, but not arbitrary interior geometry.
            if average_support < 0.36 || strong_sides < 3 {
                continue;
            }

            let interior = interior_non_background_ratio(
                bg_mask,
                small_w,
                small_h,
                corners,
            );
            if interior < 0.18 {
                continue;
            }

            let score = average_support * 0.78 + interior.min(1.0) * 0.22;
            let original_corners = corners.map(|p| {
                [
                    (p[0] * scale).clamp(0.0, width as f32),
                    (p[1] * scale).clamp(0.0, height as f32),
                ]
            });

            let mut min_x = f32::INFINITY;
            let mut min_y = f32::INFINITY;
            let mut max_x = f32::NEG_INFINITY;
            let mut max_y = f32::NEG_INFINITY;
            for p in original_corners {
                min_x = min_x.min(p[0]);
                min_y = min_y.min(p[1]);
                max_x = max_x.max(p[0]);
                max_y = max_y.max(p[1]);
            }

            let margin = margin as f32;
            let x = (min_x - margin).floor().max(0.0) as u32;
            let y = (min_y - margin).floor().max(0.0) as u32;
            let right = (max_x + margin).ceil().min(width as f32) as u32;
            let bottom = (max_y + margin).ceil().min(height as f32) as u32;

            result.push(Candidate {
                rect: PhotoRect {
                    x,
                    y,
                    w: right.saturating_sub(x),
                    h: bottom.saturating_sub(y),
                    corners: Some(original_corners),
                },
                score,
                area: area * scale * scale,
            });
        }
    }

    // Exact/near duplicate hypotheses are common when Hough returns adjacent
    // lines for the same physical edge. Keep the strongest representative.
    result.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut deduped: Vec<Candidate> = Vec::new();
    for candidate in result {
        if deduped
            .iter()
            .any(|other| overlap_smaller_ratio(candidate.rect, other.rect) > 0.82)
        {
            continue;
        }
        deduped.push(candidate);
    }

    deduped
}

fn side_support(
    boundary: &[u8],
    w: u32,
    h: u32,
    a: [f32; 2],
    b: [f32; 2],
) -> f32 {
    let length = distance(a, b);
    let samples = ((length / 5.0).round() as usize).clamp(16, 100);
    let mut hits = 0usize;

    for i in 0..samples {
        let t = (i as f32 + 0.5) / samples as f32;
        let x = a[0] + (b[0] - a[0]) * t;
        let y = a[1] + (b[1] - a[1]) * t;

        let mut found = false;
        'search: for yy in -3..=3 {
            for xx in -3..=3 {
                let px = x.round() as i32 + xx;
                let py = y.round() as i32 + yy;
                if px >= 0
                    && py >= 0
                    && px < w as i32
                    && py < h as i32
                    && boundary[(py as u32 * w + px as u32) as usize] != 0
                {
                    found = true;
                    break 'search;
                }
            }
        }
        hits += usize::from(found);
    }

    hits as f32 / samples as f32
}

fn interior_non_background_ratio(
    bg: &[bool],
    w: u32,
    h: u32,
    corners: [[f32; 2]; 4],
) -> f32 {
    let mut foreground = 0usize;
    let mut total = 0usize;

    // Avoid the border itself; sample the interior only.
    for iy in 1..=8 {
        let v = iy as f32 / 9.0;
        let left = lerp_point(corners[0], corners[3], v);
        let right = lerp_point(corners[1], corners[2], v);

        for ix in 1..=8 {
            let u = ix as f32 / 9.0;
            let p = lerp_point(left, right, u);
            let x = p[0].round() as i32;
            let y = p[1].round() as i32;
            if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
                total += 1;
                if !bg[(y as u32 * w + x as u32) as usize] {
                    foreground += 1;
                }
            }
        }
    }

    if total == 0 {
        0.0
    } else {
        foreground as f32 / total as f32
    }
}

fn line_intersection(a: DetectedLine, b: DetectedLine) -> Option<[f32; 2]> {
    let denominator = cross(a.d, b.d);
    if denominator.abs() < 1e-4 {
        return None;
    }

    let delta = [b.p[0] - a.p[0], b.p[1] - a.p[1]];
    let t = cross(delta, b.d) / denominator;
    Some([a.p[0] + a.d[0] * t, a.p[1] + a.d[1] * t])
}

fn line_midpoint(line: DetectedLine) -> [f32; 2] {
    [
        line.p[0] + line.d[0] * line.length * 0.5,
        line.p[1] + line.d[1] * line.length * 0.5,
    ]
}

fn point_line_distance(point: [f32; 2], line: DetectedLine) -> f32 {
    let delta = [point[0] - line.p[0], point[1] - line.p[1]];
    cross(delta, line.d).abs()
}

fn angle_distance(a: f32, b: f32) -> f32 {
    let diff = (a - b).abs();
    diff.min(PI - diff)
}

fn average_line_angle(a: f32, b: f32) -> f32 {
    // Double-angle averaging handles the equivalence of angle and angle + PI.
    let x = (2.0 * a).cos() + (2.0 * b).cos();
    let y = (2.0 * a).sin() + (2.0 * b).sin();
    let mut angle = 0.5 * y.atan2(x);
    if angle < 0.0 {
        angle += PI;
    }
    angle
}

fn cross(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[1] - a[1] * b[0]
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn lerp_point(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

fn overlap_area_ratio(a: PhotoRect, b: PhotoRect) -> f32 {
    let left = a.x.max(b.x);
    let top = a.y.max(b.y);
    let right = (a.x + a.w).min(b.x + b.w);
    let bottom = (a.y + a.h).min(b.y + b.h);
    if right <= left || bottom <= top {
        return 0.0;
    }

    let intersection = (right - left) as f32 * (bottom - top) as f32;
    let area = (a.w * a.h) as f32;
    if area <= 0.0 {
        0.0
    } else {
        intersection / area
    }
}

fn overlap_smaller_ratio(a: PhotoRect, b: PhotoRect) -> f32 {
    overlap_area_ratio(a, b).max(overlap_area_ratio(b, a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

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
    fn detects_dark_and_faded_rotated_photos() {
        let background = Rgb([238, 235, 228]);
        let mut image = RgbImage::from_pixel(800, 600, background);

        draw_rotated_rect(&mut image, 220.0, 155.0, 285.0, 185.0, -4.0, Rgb([45, 48, 50]));
        draw_rotated_rect(&mut image, 195.0, 430.0, 210.0, 255.0, 3.0, Rgb([92, 88, 84]));
        draw_rotated_rect(&mut image, 590.0, 405.0, 235.0, 270.0, -2.0, Rgb([207, 202, 194]));

        let found = detect_photos_cv(&DynamicImage::ImageRgb8(image), 22, 0).unwrap();
        assert_eq!(found.len(), 3, "detected: {found:?}");
    }

    #[test]
    fn detects_photos_on_coloured_background() {
        let background = Rgb([118, 92, 72]);
        let mut image = RgbImage::from_pixel(720, 520, background);

        draw_rotated_rect(&mut image, 220.0, 250.0, 270.0, 180.0, 11.0, Rgb([210, 205, 195]));
        draw_rotated_rect(&mut image, 525.0, 270.0, 220.0, 260.0, -7.0, Rgb([50, 55, 65]));

        let found = detect_photos_cv(&DynamicImage::ImageRgb8(image), 22, 0).unwrap();
        assert_eq!(found.len(), 2, "detected: {found:?}");
    }

    fn draw_rotated_rect(
        image: &mut RgbImage,
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
        angle_deg: f32,
        colour: Rgb<u8>,
    ) {
        let angle = angle_deg.to_radians();
        let cos = angle.cos();
        let sin = angle.sin();
        let half_w = w / 2.0;
        let half_h = h / 2.0;

        for y in 0..image.height() {
            for x in 0..image.width() {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let local_x = dx * cos + dy * sin;
                let local_y = -dx * sin + dy * cos;
                if local_x.abs() <= half_w && local_y.abs() <= half_h {
                    image.put_pixel(x, y, colour);
                }
            }
        }
    }
}
