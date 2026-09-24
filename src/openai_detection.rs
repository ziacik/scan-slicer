use std::{env, time::Duration};

use anyhow::{anyhow, bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as BASE64_STANDARD, Engine as _};
use image::{codecs::jpeg::JpegEncoder, DynamicImage, GenericImageView};
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::detection::PhotoRect;

const DEFAULT_MODEL: &str = "gpt-5.6-terra";
const MAX_PREVIEW_DIM: u32 = 1800;

#[derive(Debug, Deserialize)]
struct Layout {
    photos: Vec<DetectedPhoto>,
}

#[derive(Debug, Deserialize)]
struct DetectedPhoto {
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
}

#[derive(Debug, Deserialize)]
struct Point {
    x: f32,
    y: f32,
}

pub fn detect_photos_openai(image: &DynamicImage, margin: u32) -> Result<Vec<PhotoRect>> {
    let api_key = env::var("OPENAI_API_KEY").context("OPENAI_API_KEY is not set")?;
    let model = env::var("SCAN_SLICER_OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());

    let preview = image.thumbnail(MAX_PREVIEW_DIM, MAX_PREVIEW_DIM).to_rgb8();
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, 88)
        .encode_image(&DynamicImage::ImageRgb8(preview))
        .context("could not encode scan preview for OpenAI")?;

    let image_url = format!(
        "data:image/jpeg;base64,{}",
        BASE64_STANDARD.encode(jpeg)
    );

    let schema = json!({
        "type": "object",
        "properties": {
            "photos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "p1": point_schema(),
                        "p2": point_schema(),
                        "p3": point_schema(),
                        "p4": point_schema()
                    },
                    "required": ["p1", "p2", "p3", "p4"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["photos"],
        "additionalProperties": false
    });

    let body = json!({
        "model": model,
        "store": false,
        "max_output_tokens": 1500,
        "input": [{
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": concat!(
                        "Detect only the separate physical photographic prints lying on the scanned surface. ",
                        "Do not detect people, faces, objects, frames, buildings, trees, or other content inside a photograph. ",
                        "Do not return the scanner/page boundary. Include faded, low-contrast, black-and-white, damaged, ",
                        "and slightly rotated prints. For every physical print return its four outer corners. ",
                        "Coordinates are normalized from 0 to 1000 relative to the exact submitted image: x=0 is left, ",
                        "x=1000 is right, y=0 is top, y=1000 is bottom. Order the points clockwise. ",
                        "p1 must be the corner visually closest to the top-left of the scan, then p2, p3, p4 clockwise. ",
                        "Return no object unless it is itself a physical photo print."
                    )
                },
                {
                    "type": "input_image",
                    "image_url": image_url,
                    "detail": "high"
                }
            ]
        }],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "photo_print_layout",
                "strict": true,
                "schema": schema
            }
        }
    });

    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(45))
        .build()
        .context("could not create OpenAI HTTP client")?;

    let response = client
        .post("https://api.openai.com/v1/responses")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .context("OpenAI request failed")?;

    let status = response.status();
    let raw = response.text().context("could not read OpenAI response")?;
    if !status.is_success() {
        let message = serde_json::from_str::<Value>(&raw)
            .ok()
            .and_then(|v| {
                v.pointer("/error/message")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| raw.chars().take(300).collect());
        bail!("OpenAI API returned {status}: {message}");
    }

    let response_json: Value =
        serde_json::from_str(&raw).context("OpenAI returned invalid JSON")?;
    let output_text = extract_output_text(&response_json)
        .ok_or_else(|| anyhow!("OpenAI response did not contain output_text"))?;
    let layout: Layout =
        serde_json::from_str(output_text).context("OpenAI returned invalid photo layout JSON")?;

    let (width, height) = image.dimensions();
    let mut photos = layout
        .photos
        .into_iter()
        .filter_map(|photo| photo_to_rect(photo, width, height, margin))
        .collect::<Vec<_>>();

    photos.sort_by_key(|r| (r.y / 40, r.x));
    dedupe(&mut photos);
    Ok(photos)
}

fn point_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "x": { "type": "number", "minimum": 0, "maximum": 1000 },
            "y": { "type": "number", "minimum": 0, "maximum": 1000 }
        },
        "required": ["x", "y"],
        "additionalProperties": false
    })
}

fn extract_output_text(response: &Value) -> Option<&str> {
    if let Some(text) = response.get("output_text").and_then(Value::as_str) {
        return Some(text);
    }

    response
        .get("output")?
        .as_array()?
        .iter()
        .flat_map(|item| {
            item.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .find_map(|content| {
            if content.get("type").and_then(Value::as_str) == Some("output_text") {
                content.get("text").and_then(Value::as_str)
            } else {
                None
            }
        })
}

fn photo_to_rect(
    photo: DetectedPhoto,
    width: u32,
    height: u32,
    margin: u32,
) -> Option<PhotoRect> {
    let points = [photo.p1, photo.p2, photo.p3, photo.p4];
    let corners = points.map(|p| {
        [
            (p.x.clamp(0.0, 1000.0) / 1000.0) * width as f32,
            (p.y.clamp(0.0, 1000.0) / 1000.0) * height as f32,
        ]
    });

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

    let bbox_w = (max_x - min_x).max(0.0);
    let bbox_h = (max_y - min_y).max(0.0);
    let area_fraction = (bbox_w * bbox_h) / ((width as f32 * height as f32).max(1.0));

    if bbox_w < width.min(height) as f32 * 0.04
        || bbox_h < width.min(height) as f32 * 0.04
        || !(0.003..=0.80).contains(&area_fraction)
    {
        return None;
    }

    let edge_eps = width.min(height) as f32 * 0.02;
    let touched_edges = usize::from(min_x <= edge_eps)
        + usize::from(min_y <= edge_eps)
        + usize::from(max_x >= width as f32 - edge_eps)
        + usize::from(max_y >= height as f32 - edge_eps);
    if touched_edges >= 3 {
        return None;
    }

    let margin = margin as f32;
    let x = (min_x - margin).floor().max(0.0) as u32;
    let y = (min_y - margin).floor().max(0.0) as u32;
    let right = (max_x + margin).ceil().min(width as f32) as u32;
    let bottom = (max_y + margin).ceil().min(height as f32) as u32;

    Some(PhotoRect {
        x,
        y,
        w: right.saturating_sub(x),
        h: bottom.saturating_sub(y),
        corners: Some(corners),
    })
}

fn dedupe(rects: &mut Vec<PhotoRect>) {
    let mut kept = Vec::new();
    for rect in rects.drain(..) {
        if kept
            .iter()
            .copied()
            .any(|other| overlap_smaller_ratio(rect, other) > 0.85)
        {
            continue;
        }
        kept.push(rect);
    }
    *rects = kept;
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
    let smaller = (a.w * a.h).min(b.w * b.h).max(1) as f32;
    intersection / smaller
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_normalized_points_to_image_space() {
        let photo = DetectedPhoto {
            p1: Point { x: 100.0, y: 200.0 },
            p2: Point { x: 600.0, y: 200.0 },
            p3: Point { x: 600.0, y: 700.0 },
            p4: Point { x: 100.0, y: 700.0 },
        };

        let rect = photo_to_rect(photo, 2000, 1000, 0).unwrap();
        assert_eq!(rect.x, 200);
        assert_eq!(rect.y, 200);
        assert_eq!(rect.w, 1000);
        assert_eq!(rect.h, 500);
    }
}
