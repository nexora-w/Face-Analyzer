use sqlx::{Pool, Postgres, postgres::PgPoolOptions};
use anyhow::Result;
use serde_json::Value as JsonValue;
use uuid::Uuid;
use super::embeddings::{FaceEmbedding, FaceMetadata};
use std::path::Path;
use tokio::fs;

pub struct DatabaseConfig {
    pub connection_string: String,
    pub max_connections: u32,
    pub image_storage_path: String,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            connection_string: "postgres://localhost/face_analyzer".to_string(),
            max_connections: 5,
            image_storage_path: "data/faces".to_string(),
        }
    }
}

pub struct Database {
    pool: Pool<Postgres>,
    config: DatabaseConfig,
}

impl Database {
    pub async fn new(config: DatabaseConfig) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(config.max_connections)
            .connect(&config.connection_string)
            .await?;

        // Ensure the database schema exists
        Self::initialize_schema(&pool).await?;

        // Ensure image storage directory exists
        fs::create_dir_all(&config.image_storage_path).await?;

        Ok(Self { pool, config })
    }

    async fn initialize_schema(pool: &Pool<Postgres>) -> Result<()> {
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS faces (
                id UUID PRIMARY KEY,
                embedding FLOAT[] NOT NULL,
                name TEXT,
                tags TEXT[],
                timestamp TIMESTAMPTZ NOT NULL,
                source_image TEXT NOT NULL,
                confidence FLOAT NOT NULL,
                metadata JSONB
            );

            CREATE INDEX IF NOT EXISTS faces_name_idx ON faces(name);
            CREATE INDEX IF NOT EXISTS faces_timestamp_idx ON faces(timestamp);
            CREATE INDEX IF NOT EXISTS faces_tags_idx ON faces USING GIN(tags);
        "#).execute(pool).await?;

        Ok(())
    }

    pub async fn store_face(&self, face: FaceEmbedding) -> Result<()> {
        // Copy the source image to storage
        let image_path = Path::new(&face.metadata.source_image);
        let file_name = format!("{}.jpg", face.face_id);
        let storage_path = Path::new(&self.config.image_storage_path).join(&file_name);
        
        fs::copy(image_path, &storage_path).await?;

        // Store face data in database
        sqlx::query!(
            r#"
            INSERT INTO faces (
                id, embedding, name, tags, timestamp, source_image,
                confidence, metadata
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8
            )
            "#,
            Uuid::parse_str(&face.face_id)?,
            &face.embedding as &[f32],
            face.metadata.name,
            &face.metadata.tags as &[String],
            face.metadata.timestamp,
            storage_path.to_str().unwrap(),
            face.metadata.confidence,
            JsonValue::Null,
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn get_face(&self, face_id: &str) -> Result<Option<FaceEmbedding>> {
        let record = sqlx::query!(
            r#"
            SELECT * FROM faces WHERE id = $1
            "#,
            Uuid::parse_str(face_id)?
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(record.map(|r| FaceEmbedding {
            face_id: r.id.to_string(),
            embedding: r.embedding,
            metadata: FaceMetadata {
                name: r.name,
                tags: r.tags,
                timestamp: r.timestamp,
                source_image: r.source_image,
                confidence: r.confidence,
            },
        }))
    }

    pub async fn search_faces(&self, query: &SearchQuery) -> Result<Vec<FaceEmbedding>> {
        let mut sql = String::from("SELECT * FROM faces WHERE 1=1");
        let mut params = vec![];

        if let Some(name) = &query.name {
            sql.push_str(" AND name ILIKE $1");
            params.push(format!("%{}%", name));
        }

        if let Some(tags) = &query.tags {
            sql.push_str(" AND tags && $2");
            params.push(tags.join(","));
        }

        if let Some(start_date) = query.start_date {
            sql.push_str(" AND timestamp >= $3");
            params.push(start_date.to_string());
        }

        if let Some(end_date) = query.end_date {
            sql.push_str(" AND timestamp <= $4");
            params.push(end_date.to_string());
        }

        if let Some(min_confidence) = query.min_confidence {
            sql.push_str(" AND confidence >= $5");
            params.push(min_confidence.to_string());
        }

        sql.push_str(" ORDER BY timestamp DESC");

        let records = sqlx::query(&sql)
            .bind(params.get(0).unwrap_or(&String::new()))
            .bind(params.get(1).unwrap_or(&String::new()))
            .bind(params.get(2).unwrap_or(&String::new()))
            .bind(params.get(3).unwrap_or(&String::new()))
            .bind(params.get(4).unwrap_or(&String::new()))
            .fetch_all(&self.pool)
            .await?;

        let faces = records.into_iter().map(|r| FaceEmbedding {
            face_id: r.get::<Uuid, _>("id").to_string(),
            embedding: r.get::<Vec<f32>, _>("embedding"),
            metadata: FaceMetadata {
                name: r.get("name"),
                tags: r.get("tags"),
                timestamp: r.get("timestamp"),
                source_image: r.get("source_image"),
                confidence: r.get("confidence"),
            },
        }).collect();

        Ok(faces)
    }

    pub async fn update_face(&self, face_id: &str, updates: FaceUpdates) -> Result<()> {
        let mut sql = String::from("UPDATE faces SET");
        let mut params = vec![];

        if let Some(name) = updates.name {
            sql.push_str(" name = $1,");
            params.push(name);
        }

        if let Some(tags) = updates.tags {
            sql.push_str(" tags = $2,");
            params.push(tags.join(","));
        }

        if let Some(confidence) = updates.confidence {
            sql.push_str(" confidence = $3,");
            params.push(confidence.to_string());
        }

        // Remove trailing comma
        sql.pop();
        sql.push_str(" WHERE id = $4");

        sqlx::query(&sql)
            .bind(params.get(0).unwrap_or(&String::new()))
            .bind(params.get(1).unwrap_or(&String::new()))
            .bind(params.get(2).unwrap_or(&String::new()))
            .bind(Uuid::parse_str(face_id)?)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

    pub async fn delete_face(&self, face_id: &str) -> Result<()> {
        // Get the face record to find the image path
        let record = sqlx::query!(
            r#"
            SELECT source_image FROM faces WHERE id = $1
            "#,
            Uuid::parse_str(face_id)?
        )
        .fetch_optional(&self.pool)
        .await?;

        if let Some(record) = record {
            // Delete the image file
            if let Err(e) = fs::remove_file(&record.source_image).await {
                eprintln!("Failed to delete image file: {}", e);
            }
        }

        // Delete the database record
        sqlx::query!(
            r#"
            DELETE FROM faces WHERE id = $1
            "#,
            Uuid::parse_str(face_id)?
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn cleanup_old_faces(&self, days: i64) -> Result<u64> {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(days);
        
        let records = sqlx::query!(
            r#"
            DELETE FROM faces 
            WHERE timestamp < $1
            RETURNING source_image
            "#,
            cutoff,
        )
        .fetch_all(&self.pool)
        .await?;

        // Delete associated image files
        for record in &records {
            if let Err(e) = fs::remove_file(&record.source_image).await {
                eprintln!("Failed to delete image file: {}", e);
            }
        }

        Ok(records.len() as u64)
    }
}

pub struct SearchQuery {
    pub name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub start_date: Option<chrono::DateTime<chrono::Utc>>,
    pub end_date: Option<chrono::DateTime<chrono::Utc>>,
    pub min_confidence: Option<f32>,
}

pub struct FaceUpdates {
    pub name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub confidence: Option<f32>,
} 