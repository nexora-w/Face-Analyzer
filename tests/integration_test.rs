use image_analyze::analysis::analyze_image;
use std::fs;

#[test]
fn test_analyze_image_runs() {
    let result = analyze_image("images/test.jpg");
    assert!(result.is_ok());
}

#[test]
fn test_missing_image_file() {
    let result = analyze_image("images/does_not_exist.jpg");
    assert!(result.is_err());
}

#[test]
fn test_invalid_image_file() {
    let path = "images/invalid.jpg";
    fs::write(path, b"not an image").unwrap();
    let result = analyze_image(path);
    assert!(result.is_err());
    let _ = fs::remove_file(path);
} 