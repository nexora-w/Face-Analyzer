use opencv::{
    core,
    imgproc,
    prelude::*,
};
use serde::Serialize;
use anyhow::Result;

#[derive(Debug, Clone, Serialize)]
pub struct PreprocessingConfig {
    pub brightness: f64,      // -1.0 to 1.0
    pub contrast: f64,        // 0.0 to 3.0
    pub blur_size: i32,       // Gaussian blur kernel size (odd number)
    pub sharpen: bool,        // Whether to apply sharpening
    pub equalize: bool,       // Whether to apply histogram equalization
    pub denoise: bool,        // Whether to apply denoising
    pub normalize: bool,      // Whether to normalize pixel values
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            blur_size: 3,
            sharpen: false,
            equalize: true,
            denoise: true,
            normalize: true,
        }
    }
}

pub struct ImagePreprocessor {
    config: PreprocessingConfig,
}

impl ImagePreprocessor {
    pub fn new(config: PreprocessingConfig) -> Self {
        Self { config }
    }

    pub fn process(&self, image: &Mat) -> Result<Mat> {
        let mut processed = image.clone();

        let mut float_img = Mat::default();
        image.convert_to(&mut float_img, core::CV_32F, 1.0, 0.0)?;

        if self.config.brightness != 0.0 || self.config.contrast != 1.0 {
            self.adjust_brightness_contrast(&mut float_img)?;
        }

        float_img.convert_to(&mut processed, core::CV_8U, 255.0, 0.0)?;

        if self.config.blur_size > 1 {
            let mut blurred = Mat::default();
            imgproc::gaussian_blur(
                &processed,
                &mut blurred,
                core::Size::new(self.config.blur_size, self.config.blur_size),
                0.0,
                0.0,
                core::BORDER_DEFAULT,
            )?;
            processed = blurred;
        }

        if self.config.sharpen {
            processed = self.apply_sharpening(&processed)?;
        }

        if self.config.equalize {
            processed = self.apply_equalization(&processed)?;
        }

        if self.config.denoise {
            let mut denoised = Mat::default();
            core::fast_nl_means_denoising(&processed, &mut denoised, 3.0, 7, 21)?;
            processed = denoised;
        }

        if self.config.normalize {
            let mut normalized = Mat::default();
            core::normalize(&processed, &mut normalized, 0.0, 255.0, core::NORM_MINMAX, core::CV_8U, &core::no_array())?;
            processed = normalized;
        }

        Ok(processed)
    }

    fn adjust_brightness_contrast(&self, image: &mut Mat) -> Result<()> {
        let alpha = self.config.contrast;
        let beta = self.config.brightness * 127.0;

        core::add_weighted(
            image,
            alpha,
            &Mat::zeros(image.size()?, core::CV_32F)?,
            0.0,
            beta,
            image,
            -1,
        )?;

        Ok(())
    }

    fn apply_sharpening(&self, image: &Mat) -> Result<Mat> {
        let kernel = Mat::from_slice_2d(&[
            [-1.0f32, -1.0, -1.0],
            [-1.0, 9.0, -1.0],
            [-1.0, -1.0, -1.0],
        ])?;

        let mut sharpened = Mat::default();
        imgproc::filter_2d(
            image,
            &mut sharpened,
            -1,
            &kernel,
            core::Point::new(-1, -1),
            0.0,
            core::BORDER_DEFAULT,
        )?;

        Ok(sharpened)
    }

    fn apply_equalization(&self, image: &Mat) -> Result<Mat> {
        let mut equalized = Mat::default();

        if image.channels() == 1 {
            imgproc::equalize_hist(image, &mut equalized)?;
        } else {
            let mut lab = Mat::default();
            imgproc::cvt_color(image, &mut lab, imgproc::COLOR_BGR2Lab, 0)?;

            let mut lab_channels = core::Vector::<Mat>::new();
            core::split(&lab, &mut lab_channels)?;

            imgproc::equalize_hist(&lab_channels.get(0)?, &mut lab_channels.get_mut(0)?)?;

            core::merge(&lab_channels, &mut lab)?;
            imgproc::cvt_color(&lab, &mut equalized, imgproc::COLOR_Lab2BGR, 0)?;
        }

        Ok(equalized)
    }

    pub fn auto_adjust(&mut self, image: &Mat) -> Result<()> {
        let mut mean = core::Scalar::default();
        let mut stddev = core::Scalar::default();
        core::mean_std_dev(image, &mut mean, &mut stddev, &core::no_array())?;
        
        let target_mean = 127.0;
        self.config.brightness = (target_mean - mean[0]) / 255.0;

        let target_stddev = 64.0;
        self.config.contrast = target_stddev / stddev[0];
        self.config.contrast = self.config.contrast.clamp(0.5, 2.0);

        self.config.denoise = stddev[0] < 30.0;
        self.config.sharpen = mean[0] > 100.0;
        self.config.equalize = stddev[0] < 50.0;

        Ok(())
    }
} 