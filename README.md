# Face Analyzer

A Rust application for face detection and attribute analysis (age, gender) using OpenCV and ONNX Runtime.

## Features
- Detects faces in images
- Predicts age and gender for each detected face
- Outputs annotated image and JSON results
- Robust error handling and configurable output paths

## Usage

```
cargo run --release -- <image_path> [output_image_path] [output_json_path]
```
- `<image_path>`: Path to the input image (required)
- `[output_image_path]`: Path to save the annotated image (default: `images/output.jpg`)
- `[output_json_path]`: Path to save the JSON results (default: `output.json`)

Example:
```
cargo run --release -- input.jpg results/annotated.jpg results/analysis.json
```

## Dependencies
- Rust
- OpenCV (Rust crate and system library)
- ort (ONNX Runtime)
- serde, serde_json, ndarray

## Output
- Annotated image with detected faces
- JSON file with bounding boxes and attributes for each face

## Error Handling
- The program prints user-friendly error messages for file, directory, and analysis errors.
- Output directories are created automatically if they do not exist.

## License
See LICENSE file.

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

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Rust](https://www.rust-lang.org/)

For more details, see comments in the code and each module. 