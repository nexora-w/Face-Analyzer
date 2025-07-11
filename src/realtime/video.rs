use opencv::{prelude::*, videoio, Result};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use indicatif::{ProgressBar, ProgressStyle};

pub struct VideoConfig {
    pub target_fps: Option<f64>,
    pub start_time: Option<f64>,  // Start time in seconds
    pub end_time: Option<f64>,    // End time in seconds
    pub resize_width: Option<i32>,
    pub resize_height: Option<i32>,
}

impl Default for VideoConfig {
    fn default() -> Self {
        Self {
            target_fps: None,
            start_time: None,
            end_time: None,
            resize_width: None,
            resize_height: None,
        }
    }
}

pub struct VideoInfo {
    pub width: i32,
    pub height: i32,
    pub fps: f64,
    pub frame_count: i64,
    pub duration: f64,  // Duration in seconds
}

pub struct VideoProcessor {
    capture: videoio::VideoCapture,
    config: VideoConfig,
    info: VideoInfo,
    frame_time: Option<Duration>,
}

impl VideoProcessor {
    pub fn new<P: AsRef<Path>>(video_path: P, config: VideoConfig) -> Result<Self> {
        let mut capture = videoio::VideoCapture::from_file(
            video_path.as_ref().to_str().unwrap(),
            videoio::CAP_ANY
        )?;

        if !capture.is_opened()? {
            return Err(opencv::Error::new(0, "Failed to open video file".to_string()));
        }

        let width = capture.get(videoio::CAP_PROP_FRAME_WIDTH)? as i32;
        let height = capture.get(videoio::CAP_PROP_FRAME_HEIGHT)? as i32;
        let fps = capture.get(videoio::CAP_PROP_FPS)?;
        let frame_count = capture.get(videoio::CAP_PROP_FRAME_COUNT)? as i64;
        let duration = frame_count as f64 / fps;

        let frame_time = config.target_fps.map(|target_fps| {
            Duration::from_secs_f64(1.0 / target_fps)
        });

        // Set start position if specified
        if let Some(start_time) = config.start_time {
            capture.set(videoio::CAP_PROP_POS_MSEC, start_time * 1000.0)?;
        }

        Ok(Self {
            capture,
            config,
            info: VideoInfo {
                width,
                height,
                fps,
                frame_count,
                duration,
            },
            frame_time,
        })
    }

    pub fn process_video(
        mut self,
        tx: mpsc::Sender<Mat>,
        running: Arc<Mutex<bool>>,
    ) -> anyhow::Result<()> {
        println!("Starting video processing...");

        let start_frame = (self.config.start_time.unwrap_or(0.0) * self.info.fps) as i64;
        let end_frame = self.config.end_time
            .map(|t| (t * self.info.fps) as i64)
            .unwrap_or(self.info.frame_count);
        let total_frames = end_frame - start_frame;

        let progress = ProgressBar::new(total_frames as u64);
        progress.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} frames ({percent}%) {msg}")
            .unwrap()
            .progress_chars("##-"));

        let mut last_frame_time = Instant::now();
        let mut frame_count = 0;

        while *running.lock().unwrap() && frame_count < total_frames {
            // Maintain target frame rate if specified
            if let Some(target_time) = self.frame_time {
                let elapsed = last_frame_time.elapsed();
                if elapsed < target_time {
                    std::thread::sleep(target_time - elapsed);
                }
                last_frame_time = Instant::now();
            }

            // Read frame
            let mut frame = Mat::default();
            if !self.capture.read(&mut frame)? {
                break;
            }

            if frame.empty() {
                continue;
            }

            // Resize if needed
            if let (Some(width), Some(height)) = (self.config.resize_width, self.config.resize_height) {
                let mut resized = Mat::default();
                opencv::imgproc::resize(
                    &frame,
                    &mut resized,
                    opencv::core::Size::new(width, height),
                    0.0,
                    0.0,
                    opencv::imgproc::INTER_LINEAR,
                )?;
                frame = resized;
            }

            // Send frame through channel
            if tx.try_send(frame).is_err() {
                println!("Frame processing is too slow, dropping frame");
            }

            frame_count += 1;
            progress.inc(1);
        }

        progress.finish_with_message("Video processing complete");
        Ok(())
    }

    pub fn get_video_info(&self) -> String {
        format!(
            "Video Info:\n  Resolution: {}x{}\n  FPS: {:.2}\n  Duration: {:.2}s\n  Total Frames: {}",
            self.info.width,
            self.info.height,
            self.info.fps,
            self.info.duration,
            self.info.frame_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;

    fn create_dummy_video() -> std::path::PathBuf {
        let path = std::env::temp_dir().join("test_video.mp4");
        let mut file = File::create(&path).unwrap();
        file.write_all(&[0; 1024]).unwrap(); // Write dummy data
        path
    }

    #[test]
    fn test_video_config_default() {
        let config = VideoConfig::default();
        assert!(config.target_fps.is_none());
        assert!(config.start_time.is_none());
        assert!(config.end_time.is_none());
        assert!(config.resize_width.is_none());
        assert!(config.resize_height.is_none());
    }

    #[test]
    fn test_video_processor_creation() {
        let video_path = create_dummy_video();
        let config = VideoConfig::default();
        let result = VideoProcessor::new(&video_path, config);
        std::fs::remove_file(video_path).unwrap();
        // This will fail because it's not a real video file
        assert!(result.is_err());
    }
} 