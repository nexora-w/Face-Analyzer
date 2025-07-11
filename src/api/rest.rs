use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use actix_multipart::Multipart;
use actix_cors::Cors;
use serde::{Deserialize, Serialize};
use futures::{StreamExt, TryStreamExt};
use uuid::Uuid;
use std::path::Path;
use tokio::fs;
use anyhow::Result;

use crate::database::{
    storage::Database,
    embeddings::{FaceEmbedding, FaceMetadata, EmbeddingGenerator},
};
use crate::output::report::ReportGenerator;

#[derive(Deserialize)]
pub struct AnalyzeQuery {
    min_confidence: Option<f32>,
    include_embeddings: Option<bool>,
}

#[derive(Serialize)]
pub struct AnalyzeResponse {
    face_id: String,
    name: Option<String>,
    tags: Vec<String>,
    confidence: f32,
    embedding: Option<Vec<f32>>,
}

pub struct ApiConfig {
    pub host: String,
    pub port: u16,
    pub upload_dir: String,
    pub cors_origins: Vec<String>,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            upload_dir: "uploads".to_string(),
            cors_origins: vec!["http://localhost:3000".to_string()],
        }
    }
}

pub struct ApiServer {
    config: ApiConfig,
    database: Database,
    embedding_generator: EmbeddingGenerator,
    report_generator: ReportGenerator,
}

impl ApiServer {
    pub fn new(
        config: ApiConfig,
        database: Database,
        embedding_generator: EmbeddingGenerator,
        report_generator: ReportGenerator,
    ) -> Self {
        Self {
            config,
            database,
            embedding_generator,
            report_generator,
        }
    }

    pub async fn run(&self) -> Result<()> {
        fs::create_dir_all(&self.config.upload_dir).await?;

        let database = web::Data::new(self.database.clone());
        let embedding_generator = web::Data::new(self.embedding_generator.clone());
        let report_generator = web::Data::new(self.report_generator.clone());
        let upload_dir = self.config.upload_dir.clone();

        HttpServer::new(move || {
            let cors = Cors::default()
                .allowed_origin_fn(|origin, _req_head| {
                    true
                })
                .allowed_methods(vec!["GET", "POST", "PUT", "DELETE"])
                .allowed_headers(vec!["Authorization", "Content-Type"])
                .max_age(3600);

            App::new()
                .wrap(cors)
                .app_data(database.clone())
                .app_data(embedding_generator.clone())
                .app_data(report_generator.clone())
                .app_data(web::Data::new(upload_dir.clone()))
                .service(
                    web::scope("/api/v1")
                        .route("/analyze", web::post().to(analyze_image))
                        .route("/faces", web::get().to(list_faces))
                        .route("/faces/{id}", web::get().to(get_face))
                        .route("/faces/{id}", web::put().to(update_face))
                        .route("/faces/{id}", web::delete().to(delete_face))
                        .route("/report/html", web::get().to(generate_html_report))
                        .route("/report/csv", web::get().to(export_csv))
                )
        })
        .bind((self.config.host.clone(), self.config.port))?
        .run()
        .await?;

        Ok(())
    }
}

async fn analyze_image(
    mut payload: Multipart,
    query: web::Query<AnalyzeQuery>,
    database: web::Data<Database>,
    embedding_generator: web::Data<EmbeddingGenerator>,
    upload_dir: web::Data<String>,
) -> impl Responder {
    if let Ok(Some(mut field)) = payload.try_next().await {
        let content_type = field.content_disposition().unwrap();
        let filename = content_type.get_filename().unwrap();
        let file_id = Uuid::new_v4();
        let file_path = Path::new(&**upload_dir).join(file_id.to_string());

        let mut f = web::block(|| std::fs::File::create(file_path.clone())).await.unwrap();
        while let Some(chunk) = field.next().await {
            let data = chunk.unwrap();
            f = web::block(move || f.write_all(&data).map(|_| f)).await.unwrap();
        }

        let embedding = match embedding_generator.generate(&file_path.to_string_lossy()) {
            Ok(emb) => emb,
            Err(e) => return HttpResponse::BadRequest().json(format!("Failed to generate embedding: {}", e)),
        };

        let face = FaceEmbedding {
            face_id: file_id.to_string(),
            embedding,
            metadata: FaceMetadata {
                name: None,
                tags: vec![],
                timestamp: chrono::Utc::now(),
                source_image: file_path.to_string_lossy().into_owned(),
                confidence: 1.0,
            },
        };

        if let Err(e) = database.store_face(face.clone()).await {
            return HttpResponse::InternalServerError().json(format!("Failed to store face: {}", e));
        }

        let response = AnalyzeResponse {
            face_id: face.face_id,
            name: face.metadata.name,
            tags: face.metadata.tags,
            confidence: face.metadata.confidence,
            embedding: query.include_embeddings.unwrap_or(false).then(|| face.embedding),
        };

        HttpResponse::Ok().json(response)
    } else {
        HttpResponse::BadRequest().body("Invalid multipart form data")
    }
}

async fn list_faces(
    database: web::Data<Database>,
    query: web::Query<AnalyzeQuery>,
) -> impl Responder {
    let faces = match database.search_faces(&Default::default()).await {
        Ok(faces) => faces,
        Err(e) => return HttpResponse::InternalServerError().json(format!("Failed to list faces: {}", e)),
    };

    let responses: Vec<AnalyzeResponse> = faces
        .into_iter()
        .filter(|face| {
            query.min_confidence
                .map(|min| face.metadata.confidence >= min)
                .unwrap_or(true)
        })
        .map(|face| AnalyzeResponse {
            face_id: face.face_id,
            name: face.metadata.name,
            tags: face.metadata.tags,
            confidence: face.metadata.confidence,
            embedding: query.include_embeddings.unwrap_or(false).then(|| face.embedding),
        })
        .collect();

    HttpResponse::Ok().json(responses)
}

async fn get_face(
    id: web::Path<String>,
    query: web::Query<AnalyzeQuery>,
    database: web::Data<Database>,
) -> impl Responder {
    match database.get_face(&id).await {
        Ok(Some(face)) => {
            let response = AnalyzeResponse {
                face_id: face.face_id,
                name: face.metadata.name,
                tags: face.metadata.tags,
                confidence: face.metadata.confidence,
                embedding: query.include_embeddings.unwrap_or(false).then(|| face.embedding),
            };
            HttpResponse::Ok().json(response)
        }
        Ok(None) => HttpResponse::NotFound().body("Face not found"),
        Err(e) => HttpResponse::InternalServerError().json(format!("Failed to get face: {}", e)),
    }
}

#[derive(Deserialize)]
struct FaceUpdate {
    name: Option<String>,
    tags: Option<Vec<String>>,
}

async fn update_face(
    id: web::Path<String>,
    update: web::Json<FaceUpdate>,
    database: web::Data<Database>,
) -> impl Responder {
    let updates = crate::database::storage::FaceUpdates {
        name: update.name.clone(),
        tags: update.tags.clone(),
        confidence: None,
    };

    match database.update_face(&id, updates).await {
        Ok(()) => HttpResponse::Ok().finish(),
        Err(e) => HttpResponse::InternalServerError().json(format!("Failed to update face: {}", e)),
    }
}

async fn delete_face(
    id: web::Path<String>,
    database: web::Data<Database>,
) -> impl Responder {
    match database.delete_face(&id).await {
        Ok(()) => HttpResponse::Ok().finish(),
        Err(e) => HttpResponse::InternalServerError().json(format!("Failed to delete face: {}", e)),
    }
}

async fn generate_html_report(
    database: web::Data<Database>,
    report_generator: web::Data<ReportGenerator>,
) -> impl Responder {
    let faces = match database.search_faces(&Default::default()).await {
        Ok(faces) => faces,
        Err(e) => return HttpResponse::InternalServerError().json(format!("Failed to get faces: {}", e)),
    };

    match report_generator.generate_html_report(&faces, "Face Analysis Report").await {
        Ok(path) => HttpResponse::Ok().json(path),
        Err(e) => HttpResponse::InternalServerError().json(format!("Failed to generate report: {}", e)),
    }
}

async fn export_csv(
    query: web::Query<AnalyzeQuery>,
    database: web::Data<Database>,
    report_generator: web::Data<ReportGenerator>,
) -> impl Responder {
    let faces = match database.search_faces(&Default::default()).await {
        Ok(faces) => faces,
        Err(e) => return HttpResponse::InternalServerError().json(format!("Failed to get faces: {}", e)),
    };

    match report_generator
        .export_csv(&faces, query.include_embeddings.unwrap_or(false))
        .await
    {
        Ok(path) => HttpResponse::Ok().json(path),
        Err(e) => HttpResponse::InternalServerError().json(format!("Failed to export CSV: {}", e)),
    }
} 