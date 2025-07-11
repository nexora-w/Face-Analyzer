use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use anyhow::Result;
use rand::{rngs::OsRng, RngCore};
use sha2::{Sha256, Digest};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::{Serialize, Deserialize};
use std::path::Path;
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    pub ciphertext: String, // Base64 encoded
    pub nonce: String,      // Base64 encoded
    pub salt: String,       // Base64 encoded
}

pub struct Encryptor {
    key: Vec<u8>,
}

impl Encryptor {
    pub fn new(password: &str) -> Result<Self> {
        let mut salt = [0u8; 32];
        OsRng.fill_bytes(&mut salt);
        let key = Self::derive_key(password, &salt)?;
        Ok(Self { key })
    }

    pub fn from_key(key: Vec<u8>) -> Result<Self> {
        if key.len() != 32 {
            return Err(anyhow::anyhow!("Invalid key length"));
        }
        Ok(Self { key })
    }

    fn derive_key(password: &str, salt: &[u8]) -> Result<Vec<u8>> {
        let mut hasher = Sha256::new();
        hasher.update(password.as_bytes());
        hasher.update(salt);
        Ok(hasher.finalize().to_vec())
    }

    pub fn encrypt(&self, data: &[u8]) -> Result<EncryptedData> {
        let cipher = Aes256Gcm::new_from_slice(&self.key)?;
        
        // Generate random nonce
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Generate random salt for key derivation
        let mut salt = [0u8; 32];
        OsRng.fill_bytes(&mut salt);

        // Encrypt data
        let ciphertext = cipher
            .encrypt(nonce, data)
            .map_err(|e| anyhow::anyhow!("Encryption failed: {}", e))?;

        Ok(EncryptedData {
            ciphertext: BASE64.encode(ciphertext),
            nonce: BASE64.encode(nonce),
            salt: BASE64.encode(salt),
        })
    }

    pub fn decrypt(&self, encrypted: &EncryptedData) -> Result<Vec<u8>> {
        let cipher = Aes256Gcm::new_from_slice(&self.key)?;
        
        let ciphertext = BASE64.decode(&encrypted.ciphertext)?;
        let nonce = BASE64.decode(&encrypted.nonce)?;
        let nonce = Nonce::from_slice(&nonce);

        cipher
            .decrypt(nonce, ciphertext.as_ref())
            .map_err(|e| anyhow::anyhow!("Decryption failed: {}", e))
    }

    pub async fn encrypt_file(&self, input_path: &Path, output_path: &Path) -> Result<()> {
        // Read file
        let data = fs::read(input_path).await?;
        
        // Encrypt data
        let encrypted = self.encrypt(&data)?;
        
        // Serialize and write encrypted data
        let json = serde_json::to_string(&encrypted)?;
        fs::write(output_path, json).await?;
        
        Ok(())
    }

    pub async fn decrypt_file(&self, input_path: &Path, output_path: &Path) -> Result<()> {
        // Read encrypted file
        let json = fs::read_to_string(input_path).await?;
        let encrypted: EncryptedData = serde_json::from_str(&json)?;
        
        // Decrypt data
        let decrypted = self.decrypt(&encrypted)?;
        
        // Write decrypted data
        fs::write(output_path, decrypted).await?;
        
        Ok(())
    }
}

pub struct SecureStorage {
    encryptor: Encryptor,
    storage_dir: String,
}

impl SecureStorage {
    pub fn new(password: &str, storage_dir: String) -> Result<Self> {
        let encryptor = Encryptor::new(password)?;
        Ok(Self {
            encryptor,
            storage_dir,
        })
    }

    pub async fn store(&self, key: &str, data: &[u8]) -> Result<()> {
        let encrypted = self.encryptor.encrypt(data)?;
        let path = Path::new(&self.storage_dir).join(format!("{}.enc", key));
        let json = serde_json::to_string(&encrypted)?;
        fs::write(path, json).await?;
        Ok(())
    }

    pub async fn retrieve(&self, key: &str) -> Result<Vec<u8>> {
        let path = Path::new(&self.storage_dir).join(format!("{}.enc", key));
        let json = fs::read_to_string(path).await?;
        let encrypted: EncryptedData = serde_json::from_str(&json)?;
        self.encryptor.decrypt(&encrypted)
    }

    pub async fn delete(&self, key: &str) -> Result<()> {
        let path = Path::new(&self.storage_dir).join(format!("{}.enc", key));
        fs::remove_file(path).await?;
        Ok(())
    }

    pub async fn list_keys(&self) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        let mut entries = fs::read_dir(&self.storage_dir).await?;
        
        while let Some(entry) = entries.next_entry().await? {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".enc") {
                    keys.push(name[..name.len()-4].to_string());
                }
            }
        }
        
        Ok(keys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_encryption_decryption() {
        let password = "test_password";
        let encryptor = Encryptor::new(password).unwrap();
        
        let data = b"Hello, World!";
        let encrypted = encryptor.encrypt(data).unwrap();
        let decrypted = encryptor.decrypt(&encrypted).unwrap();
        
        assert_eq!(data.as_ref(), decrypted.as_slice());
    }

    #[tokio::test]
    async fn test_secure_storage() {
        let dir = tempdir().unwrap();
        let storage = SecureStorage::new(
            "test_password",
            dir.path().to_str().unwrap().to_string(),
        ).unwrap();
        
        let key = "test_key";
        let data = b"Hello, World!";
        
        // Store data
        storage.store(key, data).await.unwrap();
        
        // Retrieve data
        let retrieved = storage.retrieve(key).await.unwrap();
        assert_eq!(data.as_ref(), retrieved.as_slice());
        
        // List keys
        let keys = storage.list_keys().await.unwrap();
        assert_eq!(keys, vec!["test_key"]);
        
        // Delete data
        storage.delete(key).await.unwrap();
        assert!(storage.retrieve(key).await.is_err());
    }
} 