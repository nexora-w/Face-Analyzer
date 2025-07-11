# Face Analyzer

A comprehensive Rust-based face analysis system with advanced features including real-time processing, database integration, and web interface. Built with OpenCV, ONNX Runtime, and modern web technologies.

## Core Features

### Face Analysis
- Advanced face detection with multiple algorithms (Haar Cascade, MTCNN, RetinaFace)
- Facial attribute analysis:
  - Basic: Age, gender
  - Enhanced: Emotions, facial landmarks, pose estimation, ethnicity
- Face quality assessment
- Image preprocessing (brightness, contrast adjustment)
- Face anonymization options (blur, pixelate, blackout, emoji)

### Real-time Processing
- Webcam support via OpenCV VideoCapture
- Video file processing
- Real-time visualization
- Progress tracking for batch operations

### Database Integration
- Face embedding generation for recognition
- Face similarity comparison
- PostgreSQL integration for face data storage
- Efficient metadata management and querying

### Output and Reporting
- HTML report generation with customizable templates
- CSV export functionality
- Base64 image encoding
- Responsive grid layout for face displays
- Progress tracking for batch operations

### API and Integration
- RESTful API using actix-web
- WebSocket support for real-time updates
- File upload handling with multipart
- CORS support
- Docker containerization

### Web Interface
- Modern UI built with Yew framework
- Responsive design with CSS Grid
- Real-time updates via WebSocket
- Configuration interface
- Result visualization dashboard

### Security Features
- Face anonymization options
- AES-GCM encryption for sensitive data
- Secure storage with salt and key derivation
- Access control for API endpoints

### Performance Optimizations
- GPU acceleration support
- Multi-threaded batch processing
- Model optimization (quantization, TensorRT)
- LRU caching for results

## Dependencies
- Rust (edition 2021 or later)
- OpenCV (with Rust bindings)
- ONNX Runtime
- PostgreSQL
- Additional dependencies in Cargo.toml

## Quick Start

1. **Clone and Setup:**
   ```sh
   git clone https://github.com/nexora-w/Face-Analyzer
   cd Face-Analyzer
   cargo build --release
   ```

2. **Configure Database:**
   ```sh
   # Set environment variables
   export DATABASE_URL="postgresql://user:password@localhost/face_analyzer"
   
   # Run migrations
   cargo run --bin migrations
   ```

3. **Start Services:**
   ```sh
   # Start the API server
   cargo run --bin api-server
   
   # In another terminal, start the web interface
   cargo run --bin web-ui
   ```

4. **Access the Application:**
   - Web Interface: http://localhost:8080
   - API Documentation: http://localhost:3000/docs

## Usage Modes

### Command Line Interface
```sh
cargo run --release -- [OPTIONS] <COMMAND>

Commands:
  analyze    Analyze single image
  batch      Process multiple images
  video      Process video file
  webcam     Real-time webcam analysis
  server     Start API server
```

### Web Interface
Navigate to http://localhost:8080 to access the web dashboard featuring:
- Live webcam processing
- Batch upload interface
- Analysis results viewer
- Configuration panel
- Real-time updates

### API Integration
```sh
# Example: Analyze an image via API
curl -X POST http://localhost:3000/api/v1/analyze \
  -F "image=@path/to/image.jpg" \
  -H "Authorization: Bearer <token>"
```

## Project Structure

```
.
├── src/
│   ├── main.rs           # CLI entry point
│   ├── lib.rs            # Library root
│   ├── face.rs           # Face analysis core
│   ├── analysis.rs       # Image processing
│   ├── api/              # REST API implementation
│   ├── db/               # Database operations
│   ├── web/              # Web interface (Yew)
│   └── security/         # Security features
├── migrations/           # Database migrations
├── tests/               # Test suite
├── frontend/            # Web UI assets
├── scripts/            # Utility scripts
└── docker/             # Docker configuration
```

## Configuration

The system can be configured via:
- Environment variables
- Configuration file (config.toml)
- Web interface settings
- API endpoints

Key configuration options include:
- Database connection
- API security settings
- Processing parameters
- Model selection
- Output formatting

## Testing

```sh
# Run all tests
cargo test

# Run specific test suites
cargo test --test integration
cargo test --test api
cargo test --test web
```

## Security Considerations

- API authentication required
- Data encryption at rest
- Secure WebSocket connections
- Face data privacy options
- Access control levels

## Performance Tuning

- GPU acceleration available
- Batch processing optimization
- Caching configuration
- Database indexing
- Model quantization options

## Contributing

Contributions welcome! Please check our contributing guidelines and open issues or pull requests.

## License

See LICENSE file.

## Acknowledgements

- OpenCV
- ONNX Runtime
- Rust
- Yew Framework
- PostgreSQL
- Actix Web