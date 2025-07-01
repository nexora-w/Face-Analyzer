# Image Analyze

A Rust-based application for face detection and attribute analysis in images.  
It uses OpenCV for face detection and ONNX Runtime for age and gender prediction.

---

## Features

- Detects faces in images using Haar cascades (OpenCV).
- Predicts age and gender for each detected face using an ONNX model.
- Outputs an annotated image and a structured JSON file with results.
- Modular, testable, and easy to extend.

---

## Project Structure

```
.
├── src/              # Rust source code
│   ├── main.rs       # Entry point, minimal orchestration
│   ├── lib.rs        # Library root, exposes modules
│   ├── face.rs       # Face attribute analysis logic
│   └── analysis.rs   # Image analysis and result struct
├── tests/            # Integration tests
├── images/           # Input and output images
├── models/           # Model files (e.g., ONNX)
├── haarcascades/     # Haar cascade files for face detection
├── scripts/          # Utility scripts (e.g., data download)
├── Cargo.toml        # Rust package manifest
├── README.md         # Project documentation
```

---

## Requirements

- Rust (edition 2021 or later)
- OpenCV (with Rust bindings)
- ONNX Runtime (via the `ort` crate)
- Download required models:
  - Haar cascade: `haarcascades/haarcascade_frontalface_default.xml`
  - ONNX model: `models/face_attributes.onnx`

---

## Setup

1. **Clone this repository:**
   ```sh
   git clone https://github.com/nexora-w/Face-Analyzer
   cd Face-Analyzer
   ```

2. **Install dependencies:**
   ```sh
   cargo build --release
   ```

3. **Download models:**
   - [Haar Cascade](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
   - [Example ONNX Model](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/age-gender-recognition-retail-0013/description/age-gender-recognition-retail-0013.md)
   - Place them in `haarcascades/` and `models/` respectively.

4. **Add input images to `images/`**

---

## Usage

Run the application with:

```sh
cargo run --release -- <image_path>
```

- Annotated image will be saved as `images/output.jpg`
- JSON results will be saved as `output.json`

---

## Example

```sh
cargo run --release -- images/sample.jpg
```

**Output:**
- `images/output.jpg` (with rectangles around faces)
- `output.json` (with age/gender predictions)

---

## Testing

Run all tests with:

```sh
cargo test
```

---

## Troubleshooting

- **OpenCV errors:** Ensure OpenCV is installed and accessible to the Rust bindings.
- **Model not found:** Make sure the required model files are in the correct directories.
- **No faces detected:** Try with a clearer image or adjust the cascade parameters in the code.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Rust](https://www.rust-lang.org/)

For more details, see comments in the code and each module. 