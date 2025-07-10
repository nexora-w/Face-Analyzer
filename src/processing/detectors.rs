use opencv::{
    core,
    dnn,
    prelude::*,
    types::VectorOfMat,
};
use serde::Serialize;
use anyhow::Result;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum DetectorType {
    Haar,
    DNN,
    MTCNN,
    RetinaFace,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetectionResult {
    pub bbox: core::Rect,
    pub confidence: f32,
    pub landmarks: Option<Vec<core::Point2f>>,
}

pub struct FaceDetector {
    detector_type: DetectorType,
    confidence_threshold: f32,
    min_face_size: core::Size,
    scale_factor: f32,
}

impl FaceDetector {
    pub fn new(
        detector_type: DetectorType,
        confidence_threshold: f32,
        min_face_size: core::Size,
        scale_factor: f32,
    ) -> Self {
        Self {
            detector_type,
            confidence_threshold,
            min_face_size,
            scale_factor,
        }
    }

    pub fn detect(&self, image: &Mat) -> Result<Vec<DetectionResult>> {
        match self.detector_type {
            DetectorType::Haar => self.detect_haar(image),
            DetectorType::DNN => self.detect_dnn(image),
            DetectorType::MTCNN => self.detect_mtcnn(image),
            DetectorType::RetinaFace => self.detect_retinaface(image),
        }
    }

    fn detect_haar(&self, image: &Mat) -> Result<Vec<DetectionResult>> {
        let cascade = opencv::objdetect::CascadeClassifier::new(
            "haarcascades/haarcascade_frontalface_default.xml"
        )?;

        let mut gray = Mat::default();
        opencv::imgproc::cvt_color(image, &mut gray, opencv::imgproc::COLOR_BGR2GRAY, 0)?;

        let mut faces = opencv::types::VectorOfRect::new();
        cascade.detect_multi_scale(
            &gray,
            &mut faces,
            self.scale_factor,
            3,
            0,
            self.min_face_size,
            core::Size::new(0, 0),
        )?;

        Ok(faces.iter().map(|rect| DetectionResult {
            bbox: rect,
            confidence: 1.0,
            landmarks: None,
        }).collect())
    }

    fn detect_dnn(&self, image: &Mat) -> Result<Vec<DetectionResult>> {
        let model_path = "models/res10_300x300_ssd_iter_140000.caffemodel";
        let config_path = "models/deploy.prototxt";

        let net = dnn::read_net_from_caffe(config_path, model_path)?;
        
        let blob = dnn::blob_from_image(
            image,
            1.0,
            core::Size::new(300, 300),
            core::Scalar::new(104.0, 177.0, 123.0, 0.0),
            false,
            false,
        )?;

        net.set_input(&blob, "", 1.0, core::Scalar::default())?;
        let detections = net.forward("detection_out", &mut VectorOfMat::new())?;

        let mut results = Vec::new();
        let detection_mat = detections.try_as_mat()?;
        let num_detections = detection_mat.rows();

        for i in 0..num_detections {
            let confidence = detection_mat.at_row::<f32>(i)?[2];
            if confidence > self.confidence_threshold {
                let x1 = (detection_mat.at_row::<f32>(i)?[3] * image.cols() as f32) as i32;
                let y1 = (detection_mat.at_row::<f32>(i)?[4] * image.rows() as f32) as i32;
                let x2 = (detection_mat.at_row::<f32>(i)?[5] * image.cols() as f32) as i32;
                let y2 = (detection_mat.at_row::<f32>(i)?[6] * image.rows() as f32) as i32;

                let rect = core::Rect::new(
                    x1,
                    y1,
                    (x2 - x1).max(0),
                    (y2 - y1).max(0),
                );

                results.push(DetectionResult {
                    bbox: rect,
                    confidence,
                    landmarks: None,
                });
            }
        }

        Ok(results)
    }

    fn detect_mtcnn(&self, _image: &Mat) -> Result<Vec<DetectionResult>> {
        unimplemented!("MTCNN detection not yet implemented")
    }

    fn detect_retinaface(&self, _image: &Mat) -> Result<Vec<DetectionResult>> {
        unimplemented!("RetinaFace detection not yet implemented")
    }
}

pub struct DetectorFactory;

impl DetectorFactory {
    pub fn create_detector(
        detector_type: DetectorType,
        confidence_threshold: Option<f32>,
        min_face_size: Option<core::Size>,
        scale_factor: Option<f32>,
    ) -> Result<FaceDetector> {
        match detector_type {
            DetectorType::Haar => {
                let cascade_path = Path::new("haarcascades/haarcascade_frontalface_default.xml");
                if !cascade_path.exists() {
                    return Err(anyhow::anyhow!("Haar cascade file not found"));
                }
            }
            DetectorType::DNN => {
                let model_path = Path::new("models/res10_300x300_ssd_iter_140000.caffemodel");
                let config_path = Path::new("models/deploy.prototxt");
                if !model_path.exists() || !config_path.exists() {
                    return Err(anyhow::anyhow!("DNN model files not found"));
                }
            }
            DetectorType::MTCNN => {
            }
            DetectorType::RetinaFace => {
            }
        }

        Ok(FaceDetector::new(
            detector_type,
            confidence_threshold.unwrap_or(0.5),
            min_face_size.unwrap_or(core::Size::new(30, 30)),
            scale_factor.unwrap_or(1.1),
        ))
    }
} 