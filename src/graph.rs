use rusqlite::{Connection, Result};

pub fn store_relation(
    db: &Connection,
    from_chunk_id: i64,
    to_chunk_id: i64,
    relation_type: Option<&str>,
    confidence: Option<f64>,
) -> Result<bool> {
    db.execute(
        "INSERT OR REPLACE INTO relations (from_chunk_id, to_chunk_id, relation_type, confidence) VALUES (?1, ?2, ?3, ?4)",
        (from_chunk_id, to_chunk_id, relation_type, confidence),
    )?;

    Ok(true)
}
