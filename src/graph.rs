use rusqlite::{Connection, Error, OptionalExtension, Result};

use crate::ann;
use crate::settings;

/// Vecinos actuales de un chunk (Graph Engine, paso 1 de la sección 6): la misma
/// búsqueda global fan-out + merge entre segmentos que usa `semanta_store_embedding`
/// al insertar, por si se quieren re-consultar candidatos más adelante sin volver
/// a insertar el chunk (p. ej. tras `semanta_rebuild_graph`).
pub fn get_candidates(db: &Connection, chunk_id: i64, top_k: Option<i64>) -> Result<String> {
    let vector: Vec<u8> = db
        .query_row(
            "SELECT vector FROM embeddings WHERE chunk_id = ?1",
            [chunk_id],
            |row| row.get(0),
        )
        .optional()?
        .ok_or_else(|| {
            Error::UserFunctionError(format!("el chunk {chunk_id} no tiene un embedding guardado").into())
        })?;

    let top_k = match top_k {
        Some(value) => value as usize,
        None => settings::get_usize(db, "top_k", 16)?,
    };

    let query_vector = ann::bytes_to_f32(&vector);
    // Se busca top_k + 1 porque el propio chunk siempre aparece como su vecino
    // más cercano (distancia ~0) y hay que descartarlo antes de devolver top_k.
    let candidates: Vec<(i64, f32)> = ann::search(db, &query_vector, top_k + 1)?
        .into_iter()
        .filter(|(candidate_chunk_id, _)| *candidate_chunk_id != chunk_id)
        .take(top_k)
        .collect();

    Ok(ann::candidates_to_json(&candidates))
}

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
