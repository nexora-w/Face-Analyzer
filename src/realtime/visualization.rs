use opencv::{
    core,
    highgui,
    imgproc,
    prelude::*,
    types::VectorOfPoint,
};
use crate::face::FaceAttributes;
use crate::attributes::{
    landmarks::FacialLandmarks,
    pose::HeadPose,
};
use anyhow::Result;

pub struct VisualizationConfig {
    pub show_bounding_box: bool,
    pub show_landmarks: bool,
    pub show_pose: bool,
    pub show_attributes: bool,
    pub font_scale: f64,
    pub line_thickness: i32,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            show_bounding_box: true,
            show_landmarks: true,
            show_pose: true,
            show_attributes: true,
            font_scale: 0.5,
            line_thickness: 2,
        }
    }
}

pub struct Visualizer {
    config: VisualizationConfig,
    window_name: String,
}

impl Visualizer {
    pub fn new(window_name: &str, config: VisualizationConfig) -> Self {
        highgui::named_window(window_name, highgui::WINDOW_AUTOSIZE).unwrap();
        Self {
            config,
            window_name: window_name.to_string(),
        }
    }

    pub fn display_frame(&self, frame: &Mat, faces: &[(core::Rect, FaceAttributes)]) -> Result<()> {
        let mut display = frame.clone();

        for (bbox, attributes) in faces {
            if self.config.show_bounding_box {
                self.draw_bounding_box(&mut display, bbox)?;
            }

            if self.config.show_landmarks {
                if let Some(landmarks) = &attributes.landmarks {
                    self.draw_landmarks(&mut display, landmarks)?;
                }
            }

            if self.config.show_pose {
                if let Some(pose_est) = &attributes.pose {
                    self.draw_head_pose(&mut display, bbox, &pose_est.head_pose)?;
                }
            }

            if self.config.show_attributes {
                self.draw_attributes(&mut display, bbox, attributes)?;
            }
        }

        highgui::imshow(&self.window_name, &display)?;
        Ok(())
    }

    fn draw_bounding_box(&self, image: &mut Mat, bbox: &core::Rect) -> Result<()> {
        imgproc::rectangle(
            image,
            *bbox,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            self.config.line_thickness,
            imgproc::LINE_8,
            0,
        )?;
        Ok(())
    }

    fn draw_landmarks(&self, image: &mut Mat, landmarks: &FacialLandmarks) -> Result<()> {
        // Draw face outline
        let jaw_points: Vec<core::Point> = landmarks.jaw_line.iter()
            .map(|p| core::Point::new(p.x as i32, p.y as i32))
            .collect();
        let jaw_line = VectorOfPoint::from_iter(jaw_points);
        imgproc::polylines(
            image,
            &jaw_line,
            false,
            core::Scalar::new(255.0, 0.0, 0.0, 0.0),
            self.config.line_thickness,
            imgproc::LINE_8,
            0,
        )?;

        // Draw eyes
        for eye in [&landmarks.left_eye, &landmarks.right_eye] {
            let eye_points: Vec<core::Point> = eye.iter()
                .map(|p| core::Point::new(p.x as i32, p.y as i32))
                .collect();
            let eye_line = VectorOfPoint::from_iter(eye_points);
            imgproc::polylines(
                image,
                &eye_line,
                true,
                core::Scalar::new(0.0, 255.0, 255.0, 0.0),
                self.config.line_thickness,
                imgproc::LINE_8,
                0,
            )?;
        }

        // Draw nose
        let nose_points: Vec<core::Point> = landmarks.nose_bridge.iter()
            .map(|p| core::Point::new(p.x as i32, p.y as i32))
            .collect();
        let nose_line = VectorOfPoint::from_iter(nose_points);
        imgproc::polylines(
            image,
            &nose_line,
            false,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            self.config.line_thickness,
            imgproc::LINE_8,
            0,
        )?;

        // Draw mouth
        let mouth_points: Vec<core::Point> = landmarks.outer_lips.iter()
            .map(|p| core::Point::new(p.x as i32, p.y as i32))
            .collect();
        let mouth_line = VectorOfPoint::from_iter(mouth_points);
        imgproc::polylines(
            image,
            &mouth_line,
            true,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
            self.config.line_thickness,
            imgproc::LINE_8,
            0,
        )?;

        Ok(())
    }

    fn draw_head_pose(&self, image: &mut Mat, bbox: &core::Rect, pose: &HeadPose) -> Result<()> {
        let center = core::Point::new(
            bbox.x + bbox.width / 2,
            bbox.y + bbox.height / 2,
        );

        // Draw axes
        let axis_length = bbox.width as f32 * 0.5;
        let (sin_y, cos_y) = (pose.yaw.to_radians().sin(), pose.yaw.to_radians().cos());
        let (sin_p, cos_p) = (pose.pitch.to_radians().sin(), pose.pitch.to_radians().cos());
        
        // X-axis (red)
        let x_end = core::Point::new(
            (center.x as f32 + axis_length * cos_y) as i32,
            (center.y as f32 + axis_length * sin_y) as i32,
        );
        imgproc::line(
            image,
            center,
            x_end,
            core::Scalar::new(0.0, 0.0, 255.0, 0.0),
            self.config.line_thickness,
            imgproc::LINE_8,
            0,
        )?;

        // Y-axis (green)
        let y_end = core::Point::new(
            (center.x as f32 - axis_length * sin_p) as i32,
            (center.y as f32 + axis_length * cos_p) as i32,
        );
        imgproc::line(
            image,
            center,
            y_end,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            self.config.line_thickness,
            imgproc::LINE_8,
            0,
        )?;

        Ok(())
    }

    fn draw_attributes(&self, image: &mut Mat, bbox: &core::Rect, attrs: &FaceAttributes) -> Result<()> {
        let mut y_offset = 0;
        let line_height = 20;
        let text_color = core::Scalar::new(255.0, 255.0, 255.0, 0.0);
        let bg_color = core::Scalar::new(0.0, 0.0, 0.0, 0.0);

        // Helper function to draw text with background
        let mut draw_text = |text: &str, y_pos: i32| -> Result<()> {
            let origin = core::Point::new(bbox.x, bbox.y + y_pos);
            
            // Get text size
            let font = imgproc::FONT_HERSHEY_SIMPLEX;
            let thickness = 1;
            let baseline = 0;
            let size = imgproc::get_text_size(
                text,
                font,
                self.config.font_scale,
                thickness,
                &mut baseline.clone(),
            )?;

            // Draw background rectangle
            imgproc::rectangle(
                image,
                core::Rect::new(origin.x, origin.y - size.height, size.width, size.height + baseline),
                bg_color,
                -1,
                imgproc::LINE_8,
                0,
            )?;

            // Draw text
            imgproc::put_text(
                image,
                text,
                origin,
                font,
                self.config.font_scale,
                text_color,
                thickness,
                imgproc::LINE_8,
                false,
            )?;

            Ok(())
        };

        // Age and gender
        draw_text(&format!("Age: {:.1}", attrs.age), y_offset)?;
        y_offset += line_height;
        draw_text(&format!("Gender: {}", attrs.gender), y_offset)?;
        y_offset += line_height;

        // Emotion
        if let Some(emotion) = &attrs.emotion {
            draw_text(
                &format!("Emotion: {:?} ({:.0}%)", 
                    emotion.emotion,
                    emotion.confidence * 100.0
                ),
                y_offset
            )?;
            y_offset += line_height;
        }

        // Ethnicity
        if let Some(ethnicity) = &attrs.ethnicity {
            draw_text(
                &format!("Ethnicity: {:?} ({:.0}%)",
                    ethnicity.primary_ethnicity,
                    ethnicity.confidence * 100.0
                ),
                y_offset
            )?;
        }

        Ok(())
    }

    pub fn handle_key_events(&mut self) -> Result<bool> {
        let key = highgui::wait_key(1)?;
        match key as u8 as char {
            'q' => Ok(false),
            'b' => {
                self.config.show_bounding_box = !self.config.show_bounding_box;
                Ok(true)
            }
            'l' => {
                self.config.show_landmarks = !self.config.show_landmarks;
                Ok(true)
            }
            'p' => {
                self.config.show_pose = !self.config.show_pose;
                Ok(true)
            }
            'a' => {
                self.config.show_attributes = !self.config.show_attributes;
                Ok(true)
            }
            _ => Ok(true)
        }
    }

    pub fn cleanup(&self) {
        highgui::destroy_window(&self.window_name).ok();
    }
} 