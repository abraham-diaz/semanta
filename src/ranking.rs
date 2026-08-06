use std::collections::HashMap;

use rusqlite::{Connection, Result};

use crate::ann;
use crate::settings;

/// Una fila del resultado final de `semanta_search`: o bien un candidato ancla
/// del ANN Engine (`origin = "ann"`), o bien un chunk alcanzado expandiendo el
/// grafo de relaciones desde un ancla (`origin = "graph"`).
pub struct RankedChunk {
    pub chunk_id: i64,
    pub score: f32,
    pub origin: &'static str,
    pub relation_type: Option<String>,
    pub hop_count: Option<i64>,
}

/// Ranking Engine (sección 7 del diseño): expande los candidatos ancla del ANN
/// consultando `relations` hasta `max_hops` saltos, puntúa cada expansión con
/// `score_ancla × confidence × hop_decay^hops`, y devuelve una única lista
/// mezclada, ordenada por score y deduplicada por `chunk_id`.
pub fn expand_and_rank(db: &Connection, anchors: &[(i64, f32)], top_k: usize) -> Result<Vec<RankedChunk>> {
    let max_hops = settings::get_usize(db, "max_hops", 1)?;
    let hop_decay = settings::get_f64(db, "hop_decay", 0.5)? as f32;
    let allowed_types = settings::get_string_list(db, "expand_relation_types")?;

    let mut best: HashMap<i64, RankedChunk> = HashMap::new();
    // Frontera de expansión: (chunk_id desde el que expandir, score del ancla
    // de origen, producto de confidences acumuladas en la cadena hasta aquí).
    let mut frontier: Vec<(i64, f32, f32)> = Vec::with_capacity(anchors.len());

    for &(chunk_id, distance) in anchors {
        let score = ann::distance_to_similarity(distance);
        best.insert(
            chunk_id,
            RankedChunk {
                chunk_id,
                score,
                origin: "ann",
                relation_type: None,
                hop_count: None,
            },
        );
        frontier.push((chunk_id, score, 1.0));
    }

    for hop in 1..=max_hops {
        let mut next_frontier = Vec::new();

        for &(from_chunk_id, anchor_score, confidence_so_far) in &frontier {
            for (neighbour_id, relation_type, confidence) in
                relation_neighbours(db, from_chunk_id, allowed_types.as_deref())?
            {
                let confidence_so_far = confidence_so_far * confidence as f32;
                let score = anchor_score * confidence_so_far * hop_decay.powi(hop as i32);

                let is_better = best
                    .get(&neighbour_id)
                    .map(|existing| score > existing.score)
                    .unwrap_or(true);

                if is_better {
                    best.insert(
                        neighbour_id,
                        RankedChunk {
                            chunk_id: neighbour_id,
                            score,
                            origin: "graph",
                            relation_type: Some(relation_type),
                            hop_count: Some(hop as i64),
                        },
                    );
                }

                next_frontier.push((neighbour_id, anchor_score, confidence_so_far));
            }
        }

        frontier = next_frontier;
    }

    let mut results: Vec<RankedChunk> = best.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    results.truncate(top_k);
    Ok(results)
}

/// Vecinos de `chunk_id` en `relations`, en cualquier dirección, excluyendo
/// `relation_type IS NULL` (pares evaluados sin relación — sección 6) y
/// filtrando por `expand_relation_types` si el usuario restringió el subconjunto.
/// `confidence` ausente se trata como 1.0 (sin penalización) porque guardarla es
/// opcional para el usuario al llamar a `semanta_store_relation`.
fn relation_neighbours(
    db: &Connection,
    chunk_id: i64,
    allowed_types: Option<&[String]>,
) -> Result<Vec<(i64, String, f64)>> {
    let mut stmt = db.prepare(
        "SELECT to_chunk_id, relation_type, confidence FROM relations
           WHERE from_chunk_id = ?1 AND relation_type IS NOT NULL
         UNION ALL
         SELECT from_chunk_id, relation_type, confidence FROM relations
           WHERE to_chunk_id = ?1 AND relation_type IS NOT NULL",
    )?;
    let mut rows = stmt.query([chunk_id])?;

    let mut neighbours = Vec::new();
    while let Some(row) = rows.next()? {
        let neighbour_id: i64 = row.get(0)?;
        let relation_type: String = row.get(1)?;
        let confidence: Option<f64> = row.get(2)?;

        if let Some(allowed) = allowed_types {
            if !allowed.iter().any(|t| t == &relation_type) {
                continue;
            }
        }

        neighbours.push((neighbour_id, relation_type, confidence.unwrap_or(1.0)));
    }

    Ok(neighbours)
}
