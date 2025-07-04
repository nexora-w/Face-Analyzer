use opencv::{core, imgcodecs, imgproc, objdetect, prelude::*, types};
use serde::Serialize;
use std::env;
use std::fs::File;
use std::path::Path;
use std::fs;

use ort::{Environment, SessionBuilder, Value};

mod face;
mod analysis;
use crate::face::{analyze_face, FaceAttributes};
use crate::analysis::{analyze_image, AnalysisResult, FaceResult};
use std::io::Write;

fn print_usage(program: &str) {
    println!("Usage: {} <image_path> [output_image_path] [output_json_path]", program);
    println!("\nArguments:");
    println!("  <image_path>           Path to the input image (required)");
    println!("  [output_image_path]    Path to save the annotated image (default: images/output.jpg)");
    println!("  [output_json_path]     Path to save the JSON results (default: output.json)");
    println!("\nOptions:");
    println!("  -h, --help             Show this help message and exit");
}

fn main() -> opencv::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        print_usage(&args[0]);
        std::process::exit(0);
    }

    // Batch mode
    if args[1] == "--batch" && args.len() >= 3 {
        let input_dir = &args[2];
        let annotated_dir = Path::new("batch_output/annotated");
        let json_dir = Path::new("batch_output/json");
        let faces_dir = Path::new("batch_output/faces");
        fs::create_dir_all(annotated_dir).ok();
        fs::create_dir_all(json_dir).ok();
        fs::create_dir_all(faces_dir).ok();
        let entries = match fs::read_dir(input_dir) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Failed to read input directory: {}", e);
                std::process::exit(1);
            }
        };
        let mut image_files = vec![];
        for entry in entries {
            if let Ok(entry) = entry {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    let ext = ext.to_string_lossy().to_lowercase();
                    if ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" {
                        image_files.push(path);
                    }
                }
            }
        }
        let total = image_files.len();
        for (i, path) in image_files.iter().enumerate() {
            let fname = path.file_stem().unwrap().to_string_lossy();
            let annotated_path = annotated_dir.join(format!("{}_annotated.jpg", fname));
            let json_path = json_dir.join(format!("{}.json", fname));
            println!("Processing {}/{}: {}", i + 1, total, path.display());
            let (img, analysis) = match analyze_image(path.to_str().unwrap()) {
                Ok(res) => res,
                Err(e) => {
                    eprintln!("  Failed to analyze {}: {}", path.display(), e);
                    continue;
                }
            };
            if let Err(e) = imgcodecs::imwrite(annotated_path.to_str().unwrap(), &img, &types::VectorOfint::new()) {
                eprintln!("  Failed to write annotated image: {}", e);
                continue;
            }
            let json = match serde_json::to_string_pretty(&analysis) {
                Ok(j) => j,
                Err(e) => {
                    eprintln!("  Failed to serialize JSON: {}", e);
                    continue;
                }
            };
            if let Err(e) = File::create(&json_path).and_then(|mut file| file.write_all(json.as_bytes())) {
                eprintln!("  Failed to write JSON: {}", e);
                continue;
            }
            let orig_img = imgcodecs::imread(path.to_str().unwrap(), imgcodecs::IMREAD_COLOR).unwrap_or_default();
            for (face_idx, face) in analysis.faces.iter().enumerate() {
                let (x, y, w, h) = face.bbox;
                let rect = core::Rect { x, y, width: w, height: h };
                if x >= 0 && y >= 0 && w > 0 && h > 0 && x + w <= orig_img.cols() && y + h <= orig_img.rows() {
                    if let Ok(face_roi) = Mat::roi(&orig_img, rect) {
                        let face_path = faces_dir.join(format!("{}_face{}.jpg", fname, face_idx + 1));
                        if let Err(e) = imgcodecs::imwrite(face_path.to_str().unwrap(), &face_roi, &types::VectorOfint::new()) {
                            eprintln!("  Failed to write face image: {}", e);
                        }
                    }
                }
            }
            println!("  Saved: {} and {} ({} faces)", annotated_path.display(), json_path.display(), analysis.faces.len());
        }
        println!("Batch processing complete. Results in batch_output/.");
        return Ok(());
    }

    // Single image mode (default)
    let image_path = &args[1];
    let output_image_path = args.get(2).map(|s| s.as_str()).unwrap_or("images/output.jpg");
    let output_json_path = args.get(3).map(|s| s.as_str()).unwrap_or("output.json");

    let model_path = "models/face_attributes.onnx";
    let cascade_path = "haarcascades/haarcascade_frontalface_default.xml";
    if !Path::new(model_path).exists() {
        eprintln!("Required model file not found: {}", model_path);
        std::process::exit(1);
    }
    if !Path::new(cascade_path).exists() {
        eprintln!("Required cascade file not found: {}", cascade_path);
        std::process::exit(1);
    }

    if let Some(parent) = Path::new(output_image_path).parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).map_err(|e| {
                eprintln!("Failed to create output directory: {}", e);
                opencv::Error::new(0, format!("Failed to create output directory: {}", e))
            })?;
        }
    }

    let (img, analysis) = match analyze_image(image_path) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Failed to analyze image: {}", e);
            std::process::exit(1);
        }
    };
    if let Err(e) = opencv::imgcodecs::imwrite(output_image_path, &img, &opencv::types::VectorOfint::new()) {
        eprintln!("Failed to write output image: {}", e);
        std::process::exit(1);
    }
    let json = match serde_json::to_string_pretty(&analysis) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Failed to serialize analysis result: {}", e);
            std::process::exit(1);
        }
    };
    if let Err(e) = File::create(output_json_path).and_then(|mut file| file.write_all(json.as_bytes())) {
        eprintln!("Failed to write output JSON: {}", e);
        std::process::exit(1);
    }
    println!("Analysis complete. Results saved to {} and {}", output_image_path, output_json_path);
    Ok(())
}