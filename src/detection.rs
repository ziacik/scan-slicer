use image::DynamicImage;

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

pub struct DetectionOutput {
    pub boxes: Vec<PhotoRect>,
    pub engine: &'static str,
    pub warning: Option<String>,
}

pub fn detect_photos(image: &DynamicImage, margin: u32) -> DetectionOutput {
    match crate::openai_detection::detect_photos_openai(image, margin) {
        Ok(boxes) => DetectionOutput {
            boxes,
            engine: "OpenAI vision",
            warning: None,
        },
        Err(error) => DetectionOutput {
            boxes: Vec::new(),
            engine: "OpenAI vision",
            warning: Some(format!("Detection failed: {error}")),
        },
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
}
