use opencv::{core, prelude::*};
use ort::{Session, Value};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceEmbedding {
    pub embedding: Vec<f32>,
    pub face_id: String,
    pub metadata: FaceMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceMetadata {
    pub name: Option<String>,
    pub tags: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source_image: String,
    pub confidence: f32,
}

pub struct EmbeddingGenerator {
    session: Session,
    embedding_size: usize,
}

impl EmbeddingGenerator {
    pub fn new(model_path: &str) -> Result<Self> {
        let environment = ort::Environment::builder()
            .with_name("face_embedding")
            .build()?;
        
        let session = ort::SessionBuilder::new(&environment)?
            .with_model_from_file(model_path)?;

        Ok(Self {
            session,
            embedding_size: 512, // Typical size for face embeddings
        })
    }

    pub fn generate(&self, face_mat: &Mat) -> Result<Vec<f32>> {
        // Preprocess image
        let processed_tensor = self.preprocess_image(face_mat)?;
        
        // Run inference
        let outputs = self.session.run(vec![processed_tensor])?;
        
        // Post-process results
        self.postprocess_output(&outputs)
    }

    fn preprocess_image(&self, face_mat: &Mat) -> Result<ort::Tensor<f32>> {
        // Resize to required dimensions (typically 112x112 for face recognition)
        let mut resized = Mat::default();
        opencv::imgproc::resize(
            face_mat,
            &mut resized,
            core::Size::new(112, 112),
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;

        // Convert to float32 and normalize
        let mut float_mat = Mat::default();
        resized.convert_to(&mut float_mat, core::CV_32F, 1.0/255.0, 0.0)?;

        // Convert to NCHW format
        let mut tensor_data = vec![0f32; 1 * 3 * 112 * 112];
        for y in 0..112 {
            for x in 0..112 {
                let pixel = float_mat.at_2d::<core::Vec3f>(y, x)?;
                for c in 0..3 {
                    tensor_data[c * 112 * 112 + y * 112 + x] = pixel[c];
                }
            }
        }

        Ok(ort::Tensor::from_array(
            ndarray::Array4::from_shape_vec((1, 3, 112, 112), tensor_data)?
        ))
    }

    fn postprocess_output(&self, outputs: &[Value]) -> Result<Vec<f32>> {
        if let Value::Tensor(tensor) = &outputs[0] {
            let embedding = tensor.data::<f32>()?;
            if embedding.len() != self.embedding_size {
                return Err(anyhow::anyhow!("Unexpected embedding size"));
            }
            
            // L2 normalize the embedding
            let mut sum_squares = 0.0;
            for &x in embedding.iter() {
                sum_squares += x * x;
            }
            let norm = sum_squares.sqrt();
            
            let normalized = embedding.iter()
                .map(|&x| x / norm)
                .collect();
            
            Ok(normalized)
        } else {
            Err(anyhow::anyhow!("Invalid output type"))
        }
    }
}

pub struct EmbeddingComparator;

impl EmbeddingComparator {
    pub fn cosine_similarity(emb1: &[f32], emb2: &[f32]) -> f32 {
        let mut dot_product = 0.0;
        let mut norm1 = 0.0;
        let mut norm2 = 0.0;
        
        for (x1, x2) in emb1.iter().zip(emb2.iter()) {
            dot_product += x1 * x2;
            norm1 += x1 * x1;
            norm2 += x2 * x2;
        }
        
        dot_product / (norm1.sqrt() * norm2.sqrt())
    }

    pub fn euclidean_distance(emb1: &[f32], emb2: &[f32]) -> f32 {
        let mut sum_squares = 0.0;
        for (x1, x2) in emb1.iter().zip(emb2.iter()) {
            let diff = x1 - x2;
            sum_squares += diff * diff;
        }
        sum_squares.sqrt()
    }

    pub fn find_matches(
        query_embedding: &[f32],
        database_embeddings: &[FaceEmbedding],
        threshold: f32,
    ) -> Vec<(String, f32)> {
        let mut matches = Vec::new();
        
        for db_face in database_embeddings {
            let similarity = Self::cosine_similarity(query_embedding, &db_face.embedding);
            if similarity > threshold {
                matches.push((db_face.face_id.clone(), similarity));
            }
        }
        
        // Sort by similarity score in descending order
        matches.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        matches
    }

    pub fn cluster_embeddings(
        embeddings: &[FaceEmbedding],
        threshold: f32,
    ) -> Vec<Vec<String>> {
        let mut clusters = Vec::new();
        let mut assigned = vec![false; embeddings.len()];
        
        for i in 0..embeddings.len() {
            if assigned[i] {
                continue;
            }
            
            let mut cluster = vec![embeddings[i].face_id.clone()];
            assigned[i] = true;
            
            for j in (i + 1)..embeddings.len() {
                if assigned[j] {
                    continue;
                }
                
                let similarity = Self::cosine_similarity(
                    &embeddings[i].embedding,
                    &embeddings[j].embedding,
                );
                
                if similarity > threshold {
                    cluster.push(embeddings[j].face_id.clone());
                    assigned[j] = true;
                }
            }
            
            clusters.push(cluster);
        }
        
        clusters
    }
} 