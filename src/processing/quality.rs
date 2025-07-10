use opencv::{
    core,
    imgproc,
    prelude::*,
};
use serde::Serialize;
use anyhow::Result;

#[derive(Debug, Clone, Serialize)]
pub struct QualityMetrics {
    pub brightness: f32,      // 0.0 to 1.0
    pub contrast: f32,        // 0.0 to 1.0
    pub sharpness: f32,      // 0.0 to 1.0
    pub blur_score: f32,     // 0.0 to 1.0 (higher means less blurry)
    pub face_size: f32,      // Relative to image size (0.0 to 1.0)
    pub face_angle: f32,     // Deviation from frontal pose in degrees
    pub occlusion: f32,      // Estimated face occlusion (0.0 to 1.0)
    pub symmetry: f32,       // Face symmetry score (0.0 to 1.0)
    pub overall_score: f32,  // Combined quality score (0.0 to 1.0)
}

impl QualityMetrics {
    pub fn get_quality_description(&self) -> String {
        let mut issues = Vec::new();

        if self.brightness < 0.3 {
            issues.push("too dark");
        } else if self.brightness > 0.8 {
            issues.push("too bright");
        }

        if self.contrast < 0.3 {
            issues.push("low contrast");
        }

        if self.blur_score < 0.5 {
            issues.push("blurry");
        }

        if self.face_size < 0.1 {
            issues.push("face too small");
        }

        if self.face_angle > 30.0 {
            issues.push("face not frontal");
        }

        if self.occlusion > 0.3 {
            issues.push("face partially occluded");
        }

        if self.symmetry < 0.7 {
            issues.push("asymmetric face pose");
        }

        if issues.is_empty() {
            format!("Good quality image (score: {:.0}%)", self.overall_score * 100.0)
        } else {
            format!("Image quality issues: {} (score: {:.0}%)", 
                issues.join(", "), 
                self.overall_score * 100.0)
        }
    }
}

pub struct QualityAssessor {
    min_face_size: f32,
    max_angle: f32,
}

impl Default for QualityAssessor {
    fn default() -> Self {
        Self {
            min_face_size: 0.1,
            max_angle: 30.0,
        }
    }
}

impl QualityAssessor {
    pub fn assess_quality(&self, face_mat: &Mat, face_rect: &core::Rect) -> Result<QualityMetrics> {
        let brightness = self.calculate_brightness(face_mat)?;
        let contrast = self.calculate_contrast(face_mat)?;
        let sharpness = self.calculate_sharpness(face_mat)?;
        let blur_score = self.calculate_blur_score(face_mat)?;
        
        let face_size = self.calculate_relative_face_size(face_rect, face_mat)?;
        let face_angle = self.estimate_face_angle(face_mat)?;
        let occlusion = self.estimate_occlusion(face_mat)?;
        let symmetry = self.calculate_symmetry(face_mat)?;

        let overall_score = self.calculate_overall_score(
            &[
                brightness,
                contrast,
                sharpness,
                blur_score,
                face_size,
                1.0 - (face_angle / 90.0),
                1.0 - occlusion,
                symmetry,
            ]
        );

        Ok(QualityMetrics {
            brightness,
            contrast,
            sharpness,
            blur_score,
            face_size,
            face_angle,
            occlusion,
            symmetry,
            overall_score,
        })
    }

    fn calculate_brightness(&self, image: &Mat) -> Result<f32> {
        let mut mean = core::Scalar::default();
        let mut _stddev = core::Scalar::default();
        core::mean_std_dev(image, &mut mean, &mut _stddev, &core::no_array())?;
        Ok((mean[0] / 255.0) as f32)
    }

    fn calculate_contrast(&self, image: &Mat) -> Result<f32> {
        let mut mean = core::Scalar::default();
        let mut stddev = core::Scalar::default();
        core::mean_std_dev(image, &mut mean, &mut stddev, &core::no_array())?;
        Ok((stddev[0] / 128.0).min(1.0) as f32)
    }

    fn calculate_sharpness(&self, image: &Mat) -> Result<f32> {
        let mut gray = Mat::default();
        if image.channels() > 1 {
            imgproc::cvt_color(image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        } else {
            gray = image.clone();
        }

        let mut gradient_x = Mat::default();
        let mut gradient_y = Mat::default();
        imgproc::sobel(&gray, &mut gradient_x, core::CV_64F, 1, 0, 3, 1.0, 0.0, core::BORDER_DEFAULT)?;
        imgproc::sobel(&gray, &mut gradient_y, core::CV_64F, 0, 1, 3, 1.0, 0.0, core::BORDER_DEFAULT)?;

        let mut magnitude = Mat::default();
        core::magnitude(&gradient_x, &gradient_y, &mut magnitude)?;

        let mut mean = core::Scalar::default();
        let mut _stddev = core::Scalar::default();
        core::mean_std_dev(&magnitude, &mut mean, &mut _stddev, &core::no_array())?;

        Ok((mean[0] / 128.0).min(1.0) as f32)
    }

    fn calculate_blur_score(&self, image: &Mat) -> Result<f32> {
        let mut gray = Mat::default();
        if image.channels() > 1 {
            imgproc::cvt_color(image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        } else {
            gray = image.clone();
        }

        let mut laplacian = Mat::default();
        imgproc::laplacian(&gray, &mut laplacian, core::CV_64F, 3, 1.0, 0.0, core::BORDER_DEFAULT)?;

        let mut std_dev = core::Scalar::default();
        let mut _mean = core::Scalar::default();
        core::mean_std_dev(&laplacian, &mut _mean, &mut std_dev, &core::no_array())?;

        let variance = std_dev[0] * std_dev[0];
        Ok((variance / 1000.0).min(1.0) as f32)
    }

    fn calculate_relative_face_size(&self, face_rect: &core::Rect, image: &Mat) -> Result<f32> {
        let face_area = (face_rect.width * face_rect.height) as f32;
        let image_area = (image.cols() * image.rows()) as f32;
        Ok((face_area / image_area).min(1.0))
    }

    fn estimate_face_angle(&self, _image: &Mat) -> Result<f32> {
        Ok(0.0)
    }

    fn estimate_occlusion(&self, _image: &Mat) -> Result<f32> {
        Ok(0.0)
    }

    fn calculate_symmetry(&self, image: &Mat) -> Result<f32> {
        let mut flipped = Mat::default();
        core::flip(image, &mut flipped, 1)?; // Flip horizontally

        let mut diff = Mat::default();
        core::absdiff(image, &flipped, &mut diff)?;

        let mut mean = core::Scalar::default();
        let mut _stddev = core::Scalar::default();
        core::mean_std_dev(&diff, &mut mean, &mut _stddev, &core::no_array())?;

        Ok((1.0 - (mean[0] / 255.0)) as f32)
    }

    fn calculate_overall_score(&self, metrics: &[f32]) -> f32 {
        let weights = [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1];
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (metric, &weight) in metrics.iter().zip(weights.iter()) {
            weighted_sum += metric * weight;
            weight_sum += weight;
        }

        (weighted_sum / weight_sum).min(1.0)
    }
} 