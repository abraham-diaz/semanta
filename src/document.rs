use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{Connection, Result};

use crate::chunk::chunk_text;

pub trait DocumentExtractor {
    fn extract(&self, content: &str) -> String;
}

pub struct MarkdownExtractor;

impl DocumentExtractor for MarkdownExtractor {
    fn extract(&self, content: &str) -> String {
        content.to_string()
    }
}

const DEFAULT_CHUNK_SIZE_TOKENS: usize = 1000;
const DEFAULT_CHARS_PER_TOKEN: f64 = 3.5;
const DEFAULT_OVERLAP_RATIO: f64 = 0.125;

pub fn add_document(
    db: &Connection,
    name: &str,
    content: &str,
    metadata: Option<&str>,
    tags: Option<&str>,
) -> Result<i64> {
    let extracted = MarkdownExtractor.extract(content);
    let hash = content_hash(&extracted);
    let created_at = unix_timestamp();

    db.execute(
        "INSERT INTO documents (name, hash, created_at, metadata, tags) VALUES (?1, ?2, ?3, ?4, ?5)",
        (name, &hash, created_at, metadata, tags),
    )?;
    let document_id = db.last_insert_rowid();

    let chunks = chunk_text(
        &extracted,
        DEFAULT_CHUNK_SIZE_TOKENS,
        DEFAULT_CHARS_PER_TOKEN,
        DEFAULT_OVERLAP_RATIO,
    );

    for (position, chunk) in chunks.iter().enumerate() {
        db.execute(
            "INSERT INTO chunks (document_id, text, position) VALUES (?1, ?2, ?3)",
            (document_id, chunk, position as i64),
        )?;
    }

    Ok(document_id)
}

fn content_hash(content: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

fn unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs() as i64
}
