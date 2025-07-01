# Face Analyzer (Rust)

A Rust-based app that loads an image, detects faces using OpenCV, analyzes facial attributes with a machine learning model, and outputs results as annotated images and structured JSON.

## Features
- Face detection using OpenCV Haar cascades
- Facial attribute analysis using ONNX models
- Annotated image output
- Structured JSON output

## Requirements
- Rust (edition 2021 or later)
- OpenCV installed on your system
- ONNX Runtime (for `ort` crate)
- Download required models:
  - Haar cascade: `haarcascades/haarcascade_frontalface_default.xml`
  - ONNX model: `models/face_attributes.onnx` (see below)

## Setup
1. **Clone this repository**
2. **Install dependencies:**
   ```sh
   cargo build
   ```
3. **Download models:**
   - [Haar Cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
   - [Example ONNX Model](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/age-gender-recognition-retail-0013/description/age-gender-recognition-retail-0013.md)
   - Place them in `haarcascades/` and `models/` respectively.
4. **Add input images to `images/`**

## Usage
```sh
cargo run -- images/input.jpg
```
- Annotated image will be saved as `images/output.jpg`
- JSON results will be saved as `output.json`

## Notes
- The ONNX model and preprocessing may need adjustment depending on the model you use.
- For more advanced face detection, consider using OpenCV DNN or other models. 