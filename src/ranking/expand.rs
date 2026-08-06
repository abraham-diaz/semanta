use std::collections::HashMap;

use rusqlite::{Connection, Result};

use crate::ann;
use crate::storage::settings;

/// One row of `semanta_search`'s final result: either an anchor candidate from
/// the ANN Engine (`origin = "ann"`), or a chunk reached by expanding the
/// relations graph from an anchor (`origin = "graph"`).
pub struct RankedChunk {
    pub chunk_id: i64,
    pub score: f32,
    pub origin: &'static str,
    pub relation_type: Option<String>,
    pub hop_count: Option<i64>,
}

/// Ranking Engine (design section 7): expands the ANN's anchor candidates by
/// querying `relations` up to `max_hops` hops, scores each expansion with
/// `anchor_score × confidence × hop_decay^hops`, and returns a single merged
/// list, sorted by score and deduplicated by `chunk_id`.
pub fn expand_and_rank(db: &Connection, anchors: &[(i64, f32)], top_k: usize) -> Result<Vec<RankedChunk>> {
    let max_hops = settings::get_usize(db, "max_hops", 1)?;
    let hop_decay = settings::get_f64(db, "hop_decay", 0.5)? as f32;
    let allowed_types = settings::get_string_list(db, "expand_relation_types")?;

    let mut best: HashMap<i64, RankedChunk> = HashMap::new();
    // Expansion frontier: (chunk_id to expand from, the originating anchor's
    // score, product of confidences accumulated along the chain so far).
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

/// Neighbours of `chunk_id` in `relations`, in either direction, excluding
/// `relation_type IS NULL` (pairs evaluated with no relation — section 6) and
/// filtering by `expand_relation_types` if the user restricted the subset.
/// A missing `confidence` is treated as 1.0 (no penalty) because storing it is
/// optional for the user when calling `semanta_store_relation`.
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
