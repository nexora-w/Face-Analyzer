pub mod face;
pub mod analysis;

pub mod attributes {
    pub mod emotion;
    pub mod landmarks;
    pub mod pose;
    pub mod ethnicity;
}

pub mod realtime {
    pub mod webcam;
    pub mod video;
    pub mod visualization;
}

pub mod processing {
    pub mod preprocessing;
    pub mod quality;
    pub mod detectors;
}

pub mod database {
    pub mod embeddings;
    pub mod similarity;
    pub mod storage;
}

pub mod output {
    pub mod html;
    pub mod csv;
    pub mod progress;
}

pub mod api {
    pub mod rest;
    pub mod websocket;
    pub mod docker;
}

pub mod ui {
    pub mod web;
    pub mod config;
    pub mod dashboard;
}

pub mod security {
    pub mod anonymization;
    pub mod encryption;
    pub mod auth;
}

pub mod performance {
    pub mod gpu;
    pub mod threading;
    pub mod optimization;
}

pub mod common {
    pub mod error;
    pub mod types;
    pub mod config;
    pub mod logging;
} 