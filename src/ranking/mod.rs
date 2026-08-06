mod expand;

use rusqlite::{Connection, Result};

use expand::RankedChunk;

use crate::ann;
use crate::storage::settings;
use crate::util::json_escape;

/// `semanta_search(query_vector, top_k?, expand?)`: ANN Engine (fan-out + merge
/// across segments) and, if `expand` (default true), the Ranking Engine on top
/// (design section 7). `expand = false` gives pure ANN, without touching
/// `relations`.
pub fn search(db: &Connection, vector: &[u8], top_k: Option<i64>, expand: Option<bool>) -> Result<String> {
    let query_vector = ann::bytes_to_f32(vector);
    let top_k = match top_k {
        Some(value) => value as usize,
        None => settings::get_usize(db, "top_k", 16)?,
    };
    let expand = expand.unwrap_or(true);

    let anchors = ann::search(db, &query_vector, top_k)?;

    let results = if expand {
        expand::expand_and_rank(db, &anchors, top_k)?
    } else {
        // ann::search already returns ascending distance = descending similarity,
        // so the order is preserved when converting one by one.
        anchors
            .into_iter()
            .map(|(chunk_id, distance)| RankedChunk {
                chunk_id,
                score: ann::distance_to_similarity(distance),
                origin: "ann",
                relation_type: None,
                hop_count: None,
            })
            .collect()
    };

    Ok(results_to_json(&results))
}

fn results_to_json(results: &[RankedChunk]) -> String {
    let items: Vec<String> = results
        .iter()
        .map(|result| {
            let relation_type = match &result.relation_type {
                Some(value) => format!("\"{}\"", json_escape(value)),
                None => "null".to_string(),
            };
            let hop_count = match result.hop_count {
                Some(value) => value.to_string(),
                None => "null".to_string(),
            };

            format!(
                "{{\"chunk_id\":{},\"score\":{},\"origin\":\"{}\",\"relation_type\":{},\"hop_count\":{}}}",
                result.chunk_id, result.score, result.origin, relation_type, hop_count
            )
        })
        .collect();

    format!("[{}]", items.join(","))
}
