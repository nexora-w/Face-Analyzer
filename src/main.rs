use opencv::{core, imgcodecs, imgproc, objdetect, prelude::*, types};
use serde::Serialize;
use std::env;
use std::fs::File;
use std::path::Path;

#[derive(Serialize)]
struct FaceResult {
    bbox: (i32, i32, i32, i32),
}

#[derive(Serialize)]
struct AnalysisResult {
    faces: Vec<FaceResult>,
}

fn main() -> opencv::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image_path>", args[0]);
        std::process::exit(1);
    }
    let image_path = &args[1];

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

        let face_roi = Mat::roi(&gray, face)?;

        results.push(FaceResult {
            bbox: (face.x, face.y, face.width, face.height),
        });
    }

    imgcodecs::imwrite("images/output.jpg", &img, &opencv::types::VectorOfint::new())?;

    let analysis = AnalysisResult { faces: results };
    let json = serde_json::to_string_pretty(&analysis).unwrap();
    let mut file = File::create("output.json").unwrap();
    use std::io::Write;
    file.write_all(json.as_bytes()).unwrap();

    println!("Analysis complete. Results saved to images/output.jpg and output.json");
    Ok(())
}