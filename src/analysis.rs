use opencv::{core, imgcodecs, imgproc, objdetect, prelude::*, types};
use ort::{Environment, SessionBuilder};
use serde::Serialize;
use crate::face::{analyze_face, FaceAttributes};

#[derive(Serialize)]
pub struct FaceResult {
    pub bbox: (i32, i32, i32, i32),
    pub attributes: Option<FaceAttributes>,
}

#[derive(Serialize)]
pub struct AnalysisResult {
    pub faces: Vec<FaceResult>,
}

pub fn analyze_image(image_path: &str) -> opencv::Result<(Mat, AnalysisResult)> {
    let mut img = imgcodecs::imread(image_path, imgcodecs::IMREAD_COLOR)?;
    if img.empty() {
        eprintln!("Could not load image: {}", image_path);
        std::process::exit(1);
    }
    let face_cascade = objdetect::CascadeClassifier::new(
        "haarcascades/haarcascade_frontalface_default.xml",
    )?;
    let mut gray = Mat::default();
    imgproc::cvt_color(&img, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    let mut faces = types::VectorOfRect::new();
    face_cascade.detect_multi_scale(
        &gray,
        &mut faces,
        1.1,
        3,
        0,
        core::Size { width: 30, height: 30 },
        core::Size { width: 0, height: 0 },
    )?;
    let environment = Environment::builder().with_name("face_attr").build().unwrap();
    let session = SessionBuilder::new(&environment)
        .unwrap()
        .with_model_from_file("models/face_attributes.onnx")
        .unwrap();
    let mut results = Vec::new();
    for face in faces.iter() {
        imgproc::rectangle(
            &mut img,
            face,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
        let face_roi = Mat::roi(&img, face)?;
        let attributes = analyze_face(&face_roi, &session);
        results.push(FaceResult {
            bbox: (face.x, face.y, face.width, face.height),
            attributes,
        });
    }
    Ok((img, AnalysisResult { faces: results }))
} 