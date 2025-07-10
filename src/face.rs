use opencv::{core, imgproc, prelude::*};
use ort::{Session, Value};
use serde::Serialize;
use crate::attributes::{
    emotion::{Emotion, EmotionPrediction},
    landmarks::FacialLandmarks,
    pose::PoseEstimation,
    ethnicity::EthnicityPrediction,
};

#[derive(Debug, Serialize)]
pub struct FaceAttributes {
    pub age: f32,
    pub gender: String,
    pub emotion: Option<EmotionPrediction>,
    pub landmarks: Option<FacialLandmarks>,
    pub pose: Option<PoseEstimation>,
    pub ethnicity: Option<EthnicityPrediction>,
}

pub fn analyze_face(face_roi: &Mat, session: &Session) -> Option<FaceAttributes> {
    let mut resized = Mat::default();
    imgproc::resize(
        face_roi,
        &mut resized,
        core::Size { width: 62, height: 62 },
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    ).ok()?;
    let mut bgr = Mat::default();
    if resized.channels() == 1 {
        imgproc::cvt_color(&resized, &mut bgr, imgproc::COLOR_GRAY2BGR, 0).ok()?;
    } else {
        bgr = resized;
    }
    let mut bgr_f32 = Mat::default();
    bgr.convert_to(&mut bgr_f32, core::CV_32F, 1.0 / 255.0, 0.0).ok()?;
    let mut chw = vec![0f32; 3 * 62 * 62];
    for c in 0..3 {
        for y in 0..62 {
            for x in 0..62 {
                let val = *bgr_f32.at_2d::<core::Vec3f>(y, x).ok()?;
                chw[c * 62 * 62 + y * 62 + x] = val[c];
            }
        }
    }
    let input_tensor = ort::Tensor::from_array(
        ndarray::Array4::from_shape_vec((1, 3, 62, 62), chw).ok()?
    );
    let outputs = session.run(vec![input_tensor]).ok()?;
    if outputs.len() != 2 {
        return None;
    }
    let age = if let Value::Tensor(age_tensor) = &outputs[0] {
        let age_val: f32 = *age_tensor.data::<f32>().ok()?.get(0)?;
        age_val * 100.0
    } else {
        return None;
    };
    let gender = if let Value::Tensor(prob_tensor) = &outputs[1] {
        let probs = prob_tensor.data::<f32>().ok()?;
        if probs[0] > probs[1] {
            "male"
        } else {
            "female"
        }.to_string()
    } else {
        return None;
    };

    // TODO: Initialize and use the new attribute detectors
    let emotion = None; // Will be implemented with EmotionDetector
    let landmarks = None; // Will be implemented with LandmarkDetector
    let pose = None; // Will be implemented with PoseEstimator
    let ethnicity = None; // Will be implemented with EthnicityEstimator

    Some(FaceAttributes {
        age,
        gender,
        emotion,
        landmarks,
        pose,
        ethnicity,
    })
} 