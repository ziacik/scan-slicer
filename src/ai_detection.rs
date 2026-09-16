use std::{cell::RefCell, env};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam::{self, Sam};
use hf_hub::HFClientSync;
use image::{DynamicImage, GenericImageView};
use opencv::{
    core::{Mat, Point, Vector},
    geometry, imgproc,
};

use crate::detection::PhotoRect;

const MODEL_REPO_OWNER: &str = "lmz";
const MODEL_REPO_NAME: &str = "candle-sam";
const MODEL_FILE: &str = "mobile_sam-tiny-vitt.safetensors";
const MAX_CANDIDATES: usize = 8;

thread_local! {
    static DETECTOR: RefCell<Option<MobileSamDetector>> = const { RefCell::new(None) };
}

struct MobileSamDetector {
    model: Sam,
    device: Device,
}

pub fn refine_photos_ai(
    image: &DynamicImage,
    candidates: &[PhotoRect],
    margin: u32,
) -> Result<Vec<PhotoRect>> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }

    DETECTOR.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(MobileSamDetector::load()?);
        }

        slot.as_ref()
            .expect("detector initialized")
            .refine(image, candidates, margin)
    })
}

impl MobileSamDetector {
    fn load() -> Result<Self> {
        let model_path = if let Ok(path) = env::var("SCAN_SLICER_MOBILESAM_MODEL") {
            path.into()
        } else {
            HFClientSync::new()
                .context("could not initialize Hugging Face client")?
                .model(MODEL_REPO_OWNER, MODEL_REPO_NAME)
                .download_file()
                .filename(MODEL_FILE)
                .send()
                .context("could not download/cache MobileSAM model")?
        };

        let device = Device::Cpu;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)
                .context("could not load MobileSAM weights")?
        };
        let model = Sam::new_tiny(vb).context("could not initialize MobileSAM")?;

        Ok(Self { model, device })
    }

    fn refine(
        &self,
        image: &DynamicImage,
        candidates: &[PhotoRect],
        margin: u32,
    ) -> Result<Vec<PhotoRect>> {
        let (width, height) = image.dimensions();
        if width < 20 || height < 20 {
            return Ok(candidates.to_vec());
        }

        let resize_scale = (sam::IMAGE_SIZE as f32 / width.max(height) as f32).min(1.0);
        let preview_w = ((width as f32 * resize_scale).round() as u32).max(1);
        let preview_h = ((height as f32 * resize_scale).round() as u32).max(1);
        let preview = image
            .resize_exact(
                preview_w,
                preview_h,
                image::imageops::FilterType::Triangle,
            )
            .to_rgb8();

        let tensor = Tensor::from_vec(
            preview.as_raw().clone(),
            (preview_h as usize, preview_w as usize, 3),
            &self.device,
        )?
        .permute((2, 0, 1))?;

        // The expensive TinyViT image encoder runs exactly once. Each OpenCV
        // candidate below only needs the much cheaper prompt + mask decoder.
        let embeddings = self.model.embeddings(&tensor)?;

        let mut result = Vec::new();
        for candidate in candidates.iter().take(MAX_CANDIDATES) {
            if let Some(rect) = self.refine_candidate(
                *candidate,
                &embeddings,
                preview_w,
                preview_h,
                resize_scale,
                width,
                height,
                margin,
            )? {
                result.push(rect);
            }
        }

        result.sort_by_key(|r| (r.y / 40, r.x));
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_candidate(
        &self,
        candidate: PhotoRect,
        embeddings: &Tensor,
        preview_w: u32,
        preview_h: u32,
        resize_scale: f32,
        width: u32,
        height: u32,
        margin: u32,
    ) -> Result<Option<PhotoRect>> {
        let corners = photo_corners(candidate);
        let center = average_point(&corners);

        // Positive prompts well inside all four corners plus the centre make
        // SAM favour the physical print rather than a face/person inside it.
        let positive = [
            bilerp(corners, 0.22, 0.22),
            bilerp(corners, 0.78, 0.22),
            bilerp(corners, 0.78, 0.78),
            bilerp(corners, 0.22, 0.78),
            center,
        ];

        // Negative prompts just outside every side tell SAM that the scanner
        // bed/paper around the print belongs to the background.
        let min_side = candidate.w.min(candidate.h) as f32;
        let outside = (min_side * 0.06).clamp(8.0, 40.0);
        let mut points = Vec::with_capacity(9);

        for p in positive {
            points.push(normalize_point(
                p,
                resize_scale,
                preview_w,
                preview_h,
                true,
            ));
        }

        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            let midpoint = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
            let dx = midpoint[0] - center[0];
            let dy = midpoint[1] - center[1];
            let len = (dx * dx + dy * dy).sqrt().max(1.0);
            let p = [
                (midpoint[0] + dx / len * outside).clamp(0.0, width as f32 - 1.0),
                (midpoint[1] + dy / len * outside).clamp(0.0, height as f32 - 1.0),
            ];
            points.push(normalize_point(
                p,
                resize_scale,
                preview_w,
                preview_h,
                false,
            ));
        }

        let (low_res_mask, _iou) = self.model.forward_for_embeddings(
            embeddings,
            preview_h as usize,
            preview_w as usize,
            &points,
            false,
        )?;

        let mask = low_res_mask
            .upsample_nearest2d(sam::IMAGE_SIZE, sam::IMAGE_SIZE)?
            .get(0)?
            .get(0)?
            .narrow(0, 0, preview_h as usize)?
            .narrow(1, 0, preview_w as usize)?
            .ge(0.0)?
            .to_dtype(DType::U8)?;

        let values = mask.flatten_all()?.to_vec1::<u8>()?;
        let mask_mat = Mat::new_rows_cols_with_bytes::<u8>(
            preview_h as i32,
            preview_w as i32,
            &values,
        )?;

        let mut contours: Vector<Vector<Point>> = Vector::new();
        imgproc::find_contours(
            &mask_mat,
            &mut contours,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_SIMPLE,
            Point::new(0, 0),
        )?;

        let candidate_preview_center = [
            center[0] * resize_scale,
            center[1] * resize_scale,
        ];

        // Prefer the contour that actually contains the candidate centre.
        let mut chosen: Option<(Vector<Point>, f64)> = None;
        for contour in contours {
            let area = geometry::contour_area(&contour, false)?.abs();
            if area <= 1.0 {
                continue;
            }

            let contains_center = imgproc::point_polygon_test(
                &contour,
                opencv::core::Point2f::new(
                    candidate_preview_center[0],
                    candidate_preview_center[1],
                ),
                false,
            )? >= 0.0;

            match &chosen {
                None => chosen = Some((contour, area)),
                Some((_, best_area)) if contains_center && area > *best_area => {
                    chosen = Some((contour, area));
                }
                _ => {}
            }
        }

        let Some((contour, contour_area)) = chosen else {
            return Ok(None);
        };

        let rotated = geometry::min_area_rect(&contour)?;
        let rw = rotated.size.width.abs();
        let rh = rotated.size.height.abs();
        if rw <= 1.0 || rh <= 1.0 {
            return Ok(None);
        }

        let rect_area = rw as f64 * rh as f64;
        if rect_area <= 1.0 {
            return Ok(None);
        }

        let rectangularity = (contour_area / rect_area) as f32;
        if rectangularity < 0.70 {
            return Ok(None);
        }

        let aspect = rw.min(rh) / rw.max(rh);
        if aspect < 0.20 {
            return Ok(None);
        }

        let preview_corners = rotated_corners(
            rotated.center.x,
            rotated.center.y,
            rw,
            rh,
            rotated.angle,
        );
        let refined_corners = preview_corners.map(|p| {
            [
                (p[0] / resize_scale).clamp(0.0, width as f32),
                (p[1] / resize_scale).clamp(0.0, height as f32),
            ]
        });

        let refined = rect_from_corners(refined_corners, width, height, margin);

        // A person/tree mask inside a photo is much smaller than the original
        // OpenCV hypothesis. A scanner-background mask is much larger. Accept
        // SAM only when it describes roughly the same physical rectangle.
        let overlap = overlap_over_candidate(candidate, refined);
        let candidate_area = (candidate.w * candidate.h).max(1) as f32;
        let refined_area = (refined.w * refined.h).max(1) as f32;
        let size_ratio = refined_area / candidate_area;

        if overlap < 0.58 || !(0.55..=1.55).contains(&size_ratio) {
            return Ok(None);
        }

        let edge_eps = width.min(height) as f32 * 0.02;
        let min_x = refined_corners
            .iter()
            .map(|p| p[0])
            .fold(f32::INFINITY, f32::min);
        let min_y = refined_corners
            .iter()
            .map(|p| p[1])
            .fold(f32::INFINITY, f32::min);
        let max_x = refined_corners
            .iter()
            .map(|p| p[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let max_y = refined_corners
            .iter()
            .map(|p| p[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let touched_edges = usize::from(min_x <= edge_eps)
            + usize::from(min_y <= edge_eps)
            + usize::from(max_x >= width as f32 - edge_eps)
            + usize::from(max_y >= height as f32 - edge_eps);
        if touched_edges >= 3 {
            return Ok(None);
        }

        Ok(Some(refined))
    }
}

fn normalize_point(
    p: [f32; 2],
    resize_scale: f32,
    preview_w: u32,
    preview_h: u32,
    positive: bool,
) -> (f64, f64, bool) {
    let x = ((p[0] * resize_scale) / preview_w as f32).clamp(0.0, 1.0);
    let y = ((p[1] * resize_scale) / preview_h as f32).clamp(0.0, 1.0);
    (x as f64, y as f64, positive)
}

fn photo_corners(rect: PhotoRect) -> [[f32; 2]; 4] {
    rect.corners.unwrap_or([
        [rect.x as f32, rect.y as f32],
        [(rect.x + rect.w) as f32, rect.y as f32],
        [(rect.x + rect.w) as f32, (rect.y + rect.h) as f32],
        [rect.x as f32, (rect.y + rect.h) as f32],
    ])
}

fn average_point(points: &[[f32; 2]; 4]) -> [f32; 2] {
    [
        points.iter().map(|p| p[0]).sum::<f32>() / 4.0,
        points.iter().map(|p| p[1]).sum::<f32>() / 4.0,
    ]
}

fn bilerp(corners: [[f32; 2]; 4], u: f32, v: f32) -> [f32; 2] {
    let top = lerp(corners[0], corners[1], u);
    let bottom = lerp(corners[3], corners[2], u);
    lerp(top, bottom, v)
}

fn lerp(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
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

fn rect_from_corners(
    corners: [[f32; 2]; 4],
    width: u32,
    height: u32,
    margin: u32,
) -> PhotoRect {
    let min_x = corners
        .iter()
        .map(|p| p[0])
        .fold(f32::INFINITY, f32::min);
    let min_y = corners
        .iter()
        .map(|p| p[1])
        .fold(f32::INFINITY, f32::min);
    let max_x = corners
        .iter()
        .map(|p| p[0])
        .fold(f32::NEG_INFINITY, f32::max);
    let max_y = corners
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max);

    let margin = margin as f32;
    let x = (min_x - margin).floor().max(0.0) as u32;
    let y = (min_y - margin).floor().max(0.0) as u32;
    let right = (max_x + margin).ceil().min(width as f32) as u32;
    let bottom = (max_y + margin).ceil().min(height as f32) as u32;

    PhotoRect {
        x,
        y,
        w: right.saturating_sub(x),
        h: bottom.saturating_sub(y),
        corners: Some(corners),
    }
}

fn overlap_over_candidate(candidate: PhotoRect, refined: PhotoRect) -> f32 {
    let left = candidate.x.max(refined.x);
    let top = candidate.y.max(refined.y);
    let right = (candidate.x + candidate.w).min(refined.x + refined.w);
    let bottom = (candidate.y + candidate.h).min(refined.y + refined.h);

    if right <= left || bottom <= top {
        return 0.0;
    }

    let intersection = (right - left) as f32 * (bottom - top) as f32;
    let candidate_area = (candidate.w * candidate.h).max(1) as f32;
    intersection / candidate_area
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilerp_center_is_rectangle_center() {
        let corners = [[0.0, 0.0], [10.0, 0.0], [10.0, 20.0], [0.0, 20.0]];
        assert_eq!(bilerp(corners, 0.5, 0.5), [5.0, 10.0]);
    }

    #[test]
    fn overlap_is_relative_to_candidate() {
        let candidate = PhotoRect {
            x: 10,
            y: 10,
            w: 100,
            h: 100,
            corners: None,
        };
        let refined = PhotoRect {
            x: 20,
            y: 20,
            w: 80,
            h: 80,
            corners: None,
        };

        assert!((overlap_over_candidate(candidate, refined) - 0.64).abs() < 0.001);
    }
}
