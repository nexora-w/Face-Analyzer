use opencv::{prelude::*, videoio, Result};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use anyhow::Context;

pub struct WebcamConfig {
    pub device_id: i32,
    pub width: i32,
    pub height: i32,
    pub fps: f64,
}

impl Default for WebcamConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            width: 640,
            height: 480,
            fps: 30.0,
        }
    }
}

pub struct WebcamCapture {
    camera: videoio::VideoCapture,
    config: WebcamConfig,
    frame_time: Duration,
    last_frame: Instant,
}

impl WebcamCapture {
    pub fn new(config: WebcamConfig) -> Result<Self> {
        let mut camera = videoio::VideoCapture::new(config.device_id, videoio::CAP_ANY)?;
        
        // Configure camera
        camera.set(videoio::CAP_PROP_FRAME_WIDTH, config.width as f64)?;
        camera.set(videoio::CAP_PROP_FRAME_HEIGHT, config.height as f64)?;
        camera.set(videoio::CAP_PROP_FPS, config.fps)?;

        if !camera.is_opened()? {
            return Err(opencv::Error::new(0, format!("Failed to open camera device {}", config.device_id)));
        }

        Ok(Self {
            camera,
            config,
            frame_time: Duration::from_secs_f64(1.0 / config.fps),
            last_frame: Instant::now(),
        })
    }

    pub fn start_capture(
        mut self,
        tx: mpsc::Sender<Mat>,
        running: Arc<Mutex<bool>>,
    ) -> anyhow::Result<()> {
        println!("Starting webcam capture...");
        
        while *running.lock().unwrap() {
            // Maintain frame rate
            let elapsed = self.last_frame.elapsed();
            if elapsed < self.frame_time {
                std::thread::sleep(self.frame_time - elapsed);
            }
            self.last_frame = Instant::now();

            // Capture frame
            let mut frame = Mat::default();
            if !self.camera.read(&mut frame)? {
                println!("Failed to read frame from camera");
                continue;
            }

            if frame.empty() {
                println!("Empty frame received from camera");
                continue;
            }

            // Send frame through channel
            if tx.try_send(frame).is_err() {
                println!("Frame processing is too slow, dropping frame");
            }
        }

        println!("Stopping webcam capture...");
        Ok(())
    }

    pub fn get_camera_info(&self) -> anyhow::Result<String> {
        let actual_width = self.camera.get(videoio::CAP_PROP_FRAME_WIDTH)?;
        let actual_height = self.camera.get(videoio::CAP_PROP_FRAME_HEIGHT)?;
        let actual_fps = self.camera.get(videoio::CAP_PROP_FPS)?;
        
        Ok(format!(
            "Camera Info:\n  Resolution: {:.0}x{:.0}\n  FPS: {:.1}\n  Device ID: {}",
            actual_width,
            actual_height,
            actual_fps,
            self.config.device_id
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webcam_config_default() {
        let config = WebcamConfig::default();
        assert_eq!(config.device_id, 0);
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.fps, 30.0);
    }

    #[test]
    fn test_webcam_creation() {
        let config = WebcamConfig::default();
        let result = WebcamCapture::new(config);
        // Note: This test might fail if no webcam is available
        assert!(result.is_ok());
    }
} 