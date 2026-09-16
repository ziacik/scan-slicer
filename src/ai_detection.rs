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
const POINTS_PER_SIDE: usize = 16;

thread_local! {
    static DETECTOR: RefCell<Option<MobileSamDetector>> = const { RefCell::new(None) };
}

struct MobileSamDetector {
    model: Sam,
    device: Device,
}

#[derive(Clone, Copy)]
struct Candidate {
    rect: PhotoRect,
    score: f32,
}

pub fn detect_photos_ai(image: &DynamicImage, margin: u32) -> Result<Vec<PhotoRect>> {
    DETECTOR.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(MobileSamDetector::load()?);
        }

        slot.as_ref()
            .expect("detector initialized")
            .detect(image, margin)
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

    fn detect(&self, image: &DynamicImage, margin: u32) -> Result<Vec<PhotoRect>> {
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

        let masks = self.model.generate_masks(
            &tensor,
            POINTS_PER_SIDE,
            0,
            512.0 / 1500.0,
            1,
        )?;

        let mut candidates = Vec::new();

        for mask in masks {
            let (mask_h, mask_w) = mask.data.dims2()?;
            let values = mask.data.flatten_all()?.to_vec1::<u32>()?;

            let sam_px_x = sam::IMAGE_SIZE as f32 / mask_w as f32;
            let sam_px_y = sam::IMAGE_SIZE as f32 / mask_h as f32;
            let valid_w = ((preview_w as f32 / sam_px_x).ceil() as usize).min(mask_w);
            let valid_h = ((preview_h as f32 / sam_px_y).ceil() as usize).min(mask_h);

            if valid_w < 2 || valid_h < 2 {
                continue;
            }

            let mut bytes = vec![0u8; mask_w * mask_h];
            for y in 0..valid_h {
                for x in 0..valid_w {
                    if values[y * mask_w + x] != 0 {
                        bytes[y * mask_w + x] = 255;
                    }
                }
            }

            let mask_mat =
                Mat::new_rows_cols_with_bytes::<u8>(mask_h as i32, mask_w as i32, &bytes)?;
            let mut contours: Vector<Vector<Point>> = Vector::new();
            imgproc::find_contours(
                &mask_mat,
                &mut contours,
                imgproc::RETR_EXTERNAL,
                imgproc::CHAIN_APPROX_SIMPLE,
                Point::new(0, 0),
            )?;

            let mut best_contour: Option<Vector<Point>> = None;
            let mut best_area = 0.0f64;
            for contour in contours {
                let area = geometry::contour_area(&contour, false)?.abs();
                if area > best_area {
                    best_area = area;
                    best_contour = Some(contour);
                }
            }

            let Some(contour) = best_contour else {
                continue;
            };

            let rotated = geometry::min_area_rect(&contour)?;
            let rw = rotated.size.width.abs();
            let rh = rotated.size.height.abs();
            if rw <= 1.0 || rh <= 1.0 {
                continue;
            }

            let rect_area = rw as f64 * rh as f64;
            let image_area = (valid_w * valid_h) as f64;
            let area_fraction = rect_area / image_area;
            if !(0.008..=0.72).contains(&area_fraction) {
                continue;
            }

            let rectangularity = (best_area / rect_area) as f32;
            if rectangularity < 0.68 {
                continue;
            }

            let aspect = rw.min(rh) / rw.max(rh);
            if aspect < 0.20 {
                continue;
            }

            let mask_corners = rotated_corners(
                rotated.center.x,
                rotated.center.y,
                rw,
                rh,
                rotated.angle,
            );

            let mut corners = [[0.0f32; 2]; 4];
            for (i, point) in mask_corners.iter().enumerate() {
                let preview_x = point[0] * sam_px_x;
                let preview_y = point[1] * sam_px_y;
                corners[i] = [
                    (preview_x / resize_scale).clamp(0.0, width as f32),
                    (preview_y / resize_scale).clamp(0.0, height as f32),
                ];
            }

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

            let edge_eps = width.min(height) as f32 * 0.02;
            let touched_edges = usize::from(min_x <= edge_eps)
                + usize::from(min_y <= edge_eps)
                + usize::from(max_x >= width as f32 - edge_eps)
                + usize::from(max_y >= height as f32 - edge_eps);
            if touched_edges >= 3 {
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

            let score = mask.confidence * 0.60 + rectangularity.min(1.0) * 0.40;
            candidates.push(Candidate { rect, score });
        }

        candidates.sort_by(|a, b| b.score.total_cmp(&a.score));

        let mut accepted: Vec<Candidate> = Vec::new();
        for candidate in candidates {
            if accepted
                .iter()
                .any(|other| overlap_smaller_ratio(candidate.rect, other.rect) > 0.72)
            {
                continue;
            }
            accepted.push(candidate);
        }

        let mut result = accepted.into_iter().map(|c| c.rect).collect::<Vec<_>>();
        result.sort_by_key(|r| (r.y / 40, r.x));
        Ok(result)
    }
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

fn overlap_smaller_ratio(a: PhotoRect, b: PhotoRect) -> f32 {
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
    fn duplicate_overlap_uses_smaller_box() {
        let a = PhotoRect {
            x: 10,
            y: 10,
            w: 100,
            h: 100,
            corners: None,
        };
        let b = PhotoRect {
            x: 20,
            y: 20,
            w: 60,
            h: 60,
            corners: None,
        };

        assert_eq!(overlap_smaller_ratio(a, b), 1.0);
    }
}
