use image_analyze::analysis::analyze_image;

#[test]
fn test_analyze_image_runs() {
    // This test checks that analyze_image runs without panicking on a sample image.
    // You should provide a valid test image at images/test.jpg for this test to pass.
    let result = analyze_image("images/test.jpg");
    assert!(result.is_ok());
} 