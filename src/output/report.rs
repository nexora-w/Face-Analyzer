use crate::database::embeddings::{FaceEmbedding, FaceMetadata};
use anyhow::Result;
use askama::Template;
use csv::Writer;
use std::path::Path;
use tokio::fs;
use base64;
use image;

#[derive(Template)]
#[template(path = "face_report.html")]
struct FaceReportTemplate<'a> {
    title: &'a str,
    faces: &'a [FaceReportEntry],
    generated_at: chrono::DateTime<chrono::Utc>,
}

struct FaceReportEntry {
    face_id: String,
    name: Option<String>,
    tags: Vec<String>,
    timestamp: chrono::DateTime<chrono::Utc>,
    confidence: f32,
    image_data: String, // Base64 encoded image
}

pub struct ReportGenerator {
    output_dir: String,
}

impl ReportGenerator {
    pub fn new(output_dir: String) -> Self {
        Self { output_dir }
    }

    pub async fn generate_html_report(
        &self,
        faces: &[FaceEmbedding],
        title: &str,
    ) -> Result<String> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(&self.output_dir).await?;

        // Convert faces to report entries with base64 encoded images
        let mut report_entries = Vec::new();
        for face in faces {
            let image_data = Self::load_image_as_base64(&face.metadata.source_image)?;
            report_entries.push(FaceReportEntry {
                face_id: face.face_id.clone(),
                name: face.metadata.name.clone(),
                tags: face.metadata.tags.clone(),
                timestamp: face.metadata.timestamp,
                confidence: face.metadata.confidence,
                image_data,
            });
        }

        // Generate HTML using template
        let template = FaceReportTemplate {
            title,
            faces: &report_entries,
            generated_at: chrono::Utc::now(),
        };

        let html = template.render()?;

        // Write HTML to file
        let file_name = format!(
            "face_report_{}.html",
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        );
        let file_path = Path::new(&self.output_dir).join(&file_name);
        fs::write(&file_path, html).await?;

        Ok(file_path.to_string_lossy().into_owned())
    }

    pub async fn export_csv(
        &self,
        faces: &[FaceEmbedding],
        include_embeddings: bool,
    ) -> Result<String> {
        // Create output directory if it doesn't exist
        fs::create_dir_all(&self.output_dir).await?;

        // Create CSV file
        let file_name = format!(
            "face_export_{}.csv",
            chrono::Utc::now().format("%Y%m%d_%H%M%S")
        );
        let file_path = Path::new(&self.output_dir).join(&file_name);
        
        let mut writer = Writer::from_path(&file_path)?;

        // Write header
        let mut headers = vec![
            "face_id",
            "name",
            "tags",
            "timestamp",
            "confidence",
            "source_image",
        ];
        if include_embeddings {
            headers.push("embedding");
        }
        writer.write_record(headers)?;

        // Write data
        for face in faces {
            let mut record = vec![
                face.face_id.clone(),
                face.metadata.name.clone().unwrap_or_default(),
                face.metadata.tags.join(","),
                face.metadata.timestamp.to_rfc3339(),
                face.metadata.confidence.to_string(),
                face.metadata.source_image.clone(),
            ];

            if include_embeddings {
                record.push(
                    face.embedding
                        .iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("|"),
                );
            }

            writer.write_record(record)?;
        }

        writer.flush()?;
        Ok(file_path.to_string_lossy().into_owned())
    }

    fn load_image_as_base64(image_path: &str) -> Result<String> {
        let img = image::open(image_path)?;
        let mut buffer = Vec::new();
        img.write_to(&mut buffer, image::ImageFormat::Jpeg)?;
        Ok(format!(
            "data:image/jpeg;base64,{}",
            base64::encode(&buffer)
        ))
    }
}

// Template for the HTML report
const REPORT_TEMPLATE: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            margin-bottom: 20px;
        }
        .face-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .face-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: white;
        }
        .face-image {
            width: 100%;
            height: 200px;
            object-fit: cover;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .face-info {
            font-size: 14px;
        }
        .tag {
            display: inline-block;
            background-color: #e9ecef;
            padding: 2px 8px;
            border-radius: 12px;
            margin: 2px;
            font-size: 12px;
        }
        .confidence {
            color: #28a745;
            font-weight: bold;
        }
        .timestamp {
            color: #666;
            font-size: 12px;
        }
        .footer {
            margin-top: 20px;
            text-align: center;
            color: #666;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{ title }}</h1>
        <div class="face-grid">
            {% for face in faces %}
            <div class="face-card">
                <img src="{{ face.image_data }}" alt="Face {{ face.face_id }}" class="face-image">
                <div class="face-info">
                    <div>ID: {{ face.face_id }}</div>
                    {% if face.name %}
                    <div>Name: {{ face.name }}</div>
                    {% endif %}
                    <div>
                        {% for tag in face.tags %}
                        <span class="tag">{{ tag }}</span>
                        {% endfor %}
                    </div>
                    <div class="confidence">Confidence: {{ face.confidence }}%</div>
                    <div class="timestamp">{{ face.timestamp }}</div>
                </div>
            </div>
            {% endfor %}
        </div>
        <div class="footer">
            Generated at {{ generated_at }}
        </div>
    </div>
</body>
</html>
"#; 