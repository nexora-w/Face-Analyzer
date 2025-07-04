use opencv::{core, imgcodecs, imgproc, objdetect, prelude::*, types};
use serde::Serialize;
use std::env;
use std::fs::File;
use std::path::Path;

use ort::{Environment, SessionBuilder, Value};

mod face;
mod analysis;
use crate::face::{analyze_face, FaceAttributes};
use crate::analysis::{analyze_image, AnalysisResult, FaceResult};
use std::io::Write;

fn main() -> opencv::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image_path> [output_image_path] [output_json_path]", args[0]);
        std::process::exit(1);
    }
    let image_path = &args[1];
    let output_image_path = args.get(2).map(|s| s.as_str()).unwrap_or("images/output.jpg");
    let output_json_path = args.get(3).map(|s| s.as_str()).unwrap_or("output.json");

    // Ensure output directory exists
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