use opencv::{
    core,
    imgproc,
    prelude::*,
    types,
};
use anyhow::Result;

pub enum AnonymizationMethod {
    Blur { kernel_size: i32 },
    Pixelate { block_size: i32 },
    BlackOut,
    Emoji { emoji_path: String },
}

pub struct Anonymizer {
    method: AnonymizationMethod,
}

impl Anonymizer {
    pub fn new(method: AnonymizationMethod) -> Self {
        Self { method }
    }

    pub fn anonymize(&self, image: &Mat, face_rect: core::Rect) -> Result<Mat> {
        let mut output = image.clone();
        let roi = Mat::roi(&output, face_rect)?;

        match &self.method {
            AnonymizationMethod::Blur { kernel_size } => {
                let mut blurred = Mat::default();
                imgproc::gaussian_blur(
                    &roi,
                    &mut blurred,
                    core::Size::new(*kernel_size, *kernel_size),
                    0.0,
                    0.0,
                    core::BORDER_DEFAULT,
                )?;
                blurred.copy_to(&mut roi)?;
            }
            AnonymizationMethod::Pixelate { block_size } => {
                let scale = 1.0 / *block_size as f64;
                let mut small = Mat::default();
                let mut pixelated = Mat::default();

                // Resize down
                imgproc::resize(
                    &roi,
                    &mut small,
                    core::Size::new(0, 0),
                    scale,
                    scale,
                    imgproc::INTER_LINEAR,
                )?;

                // Resize up
                imgproc::resize(
                    &small,
                    &mut pixelated,
                    roi.size()?,
                    0.0,
                    0.0,
                    imgproc::INTER_NEAREST,
                )?;

                pixelated.copy_to(&mut roi)?;
            }
            AnonymizationMethod::BlackOut => {
                let color = core::Scalar::new(0.0, 0.0, 0.0, 255.0);
                imgproc::rectangle(
                    &mut output,
                    face_rect,
                    color,
                    -1,
                    imgproc::LINE_8,
                    0,
                )?;
            }
            AnonymizationMethod::Emoji { emoji_path } => {
                let emoji = imgcodecs::imread(emoji_path, imgcodecs::IMREAD_UNCHANGED)?;
                if emoji.empty() {
                    return Err(anyhow::anyhow!("Failed to load emoji image"));
                }

                // Resize emoji to fit face region
                let mut resized_emoji = Mat::default();
                imgproc::resize(
                    &emoji,
                    &mut resized_emoji,
                    face_rect.size(),
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )?;

                // Handle alpha channel if present
                if resized_emoji.channels() == 4 {
                    let mut channels = types::VectorOfMat::new();
                    core::split(&resized_emoji, &mut channels)?;

                    let alpha = channels.get(3)?;
                    let mut rgb_channels = types::VectorOfMat::new();
                    for i in 0..3 {
                        rgb_channels.push(channels.get(i)?);
                    }

                    let mut rgb = Mat::default();
                    core::merge(&rgb_channels, &mut rgb)?;

                    // Apply alpha blending
                    let mut alpha_norm = Mat::default();
                    let mut inv_alpha = Mat::default();
                    core::normalize(&alpha, &mut alpha_norm, 0.0, 1.0, core::NORM_MINMAX, -1, &core::no_array())?;
                    core::subtract(&core::Scalar::new(1.0, 1.0, 1.0, 1.0), &alpha_norm, &mut inv_alpha, &core::no_array(), -1)?;

                    let mut face_float = Mat::default();
                    roi.convert_to(&mut face_float, core::CV_32F, 1.0/255.0, 0.0)?;

                    let mut emoji_float = Mat::default();
                    rgb.convert_to(&mut emoji_float, core::CV_32F, 1.0/255.0, 0.0)?;

                    let mut blended = Mat::default();
                    core::add_weighted(&face_float, 1.0, &emoji_float, -1.0, 0.0, &mut blended, -1)?;
                    core::multiply(&blended, &inv_alpha, &mut blended, 1.0, -1)?;
                    core::add_weighted(&emoji_float, 1.0, &blended, 1.0, 0.0, &mut blended, -1)?;

                    let mut result = Mat::default();
                    blended.convert_to(&mut result, core::CV_8U, 255.0, 0.0)?;
                    result.copy_to(&mut roi)?;
                } else {
                    resized_emoji.copy_to(&mut roi)?;
                }
            }
        }

        Ok(output)
    }

    pub fn batch_anonymize(
        &self,
        image: &Mat,
        face_rects: &[core::Rect],
    ) -> Result<Mat> {
        let mut output = image.clone();
        for rect in face_rects {
            output = self.anonymize(&output, *rect)?;
        }
        Ok(output)
    }
} 