pub mod face;
pub mod analysis;

// Enhanced face attributes modules
pub mod attributes {
    pub mod emotion;
    pub mod landmarks;
    pub mod pose;
    pub mod ethnicity;
}

// Real-time processing modules
pub mod realtime {
    pub mod webcam;
    pub mod video;
    pub mod visualization;
}

// Image processing modules
pub mod processing {
    pub mod preprocessing;
    pub mod quality;
    pub mod detectors;
}

// Database modules
pub mod database {
    pub mod embeddings;
    pub mod similarity;
    pub mod storage;
}

// Output modules
pub mod output {
    pub mod html;
    pub mod csv;
    pub mod progress;
}

// API modules
pub mod api {
    pub mod rest;
    pub mod websocket;
    pub mod docker;
}

// UI modules
pub mod ui {
    pub mod web;
    pub mod config;
    pub mod dashboard;
}

// Security modules
pub mod security {
    pub mod anonymization;
    pub mod encryption;
    pub mod auth;
}

// Performance modules
pub mod performance {
    pub mod gpu;
    pub mod threading;
    pub mod optimization;
}

// Common utilities and types
pub mod common {
    pub mod error;
    pub mod types;
    pub mod config;
    pub mod logging;
} 