# Image Analyze

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

- `src/`: All Rust source code, now modularized for clarity and maintainability.
- `tests/`: Integration tests for the project.
- `images/`: Store input images and output results (e.g., output.jpg).
- `models/`: Store ONNX or other model files.
- `haarcascades/`: Haar cascade XML files for face detection.
- `scripts/`: Place for utility scripts (currently a placeholder).

## Usage

Build and run the project:

```sh
cargo run --release -- <image_path>
```

Results will be saved to `images/output.jpg` and `output.json`.

## Testing

Run tests with:

```sh
cargo test
```

---

For more details, see comments in the code and each module. 