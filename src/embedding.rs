use rusqlite::{Connection, Error, OptionalExtension, Result};

use crate::ann;
use crate::util::unix_timestamp;

pub fn store_embedding(
    db: &Connection,
    chunk_id: i64,
    vector: &[u8],
    model_name: &str,
    dimension: i64,
) -> Result<i64> {
    let expected_bytes = dimension as usize * std::mem::size_of::<f32>();
    if vector.len() != expected_bytes {
        return Err(Error::UserFunctionError(
            format!(
                "vector de {} bytes no coincide con dimension {} (se esperaban {} bytes, 4 por componente f32)",
                vector.len(),
                dimension,
                expected_bytes
            )
            .into(),
        ));
    }

    let embedding_model_id = resolve_embedding_model(db, model_name, dimension)?;
    let segment_id = ann::insert(db, chunk_id, &bytes_to_f32(vector))?;

    db.execute(
        "INSERT INTO embeddings (chunk_id, embedding_model_id, segment_id, vector) VALUES (?1, ?2, ?3, ?4)",
        (chunk_id, embedding_model_id, segment_id, vector),
    )?;

    Ok(chunk_id)
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn resolve_embedding_model(db: &Connection, model_name: &str, dimension: i64) -> Result<i64> {
    let existing: Option<i64> = db
        .query_row(
            "SELECT id FROM embedding_models WHERE model_name = ?1 AND dimension = ?2",
            (model_name, dimension),
            |row| row.get(0),
        )
        .optional()?;

    if let Some(id) = existing {
        return Ok(id);
    }

    db.execute(
        "INSERT INTO embedding_models (model_name, dimension, generated_at) VALUES (?1, ?2, ?3)",
        (model_name, dimension, unix_timestamp()),
    )?;

    Ok(db.last_insert_rowid())
}
