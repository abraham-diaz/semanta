use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

mod persistence;
#[cfg(all(test, feature = "testing"))]
mod tests;

use hnsw_rs::prelude::*;
use rusqlite::{Connection, OptionalExtension, Result, params_from_iter};

use crate::storage::settings;
use crate::util::unix_timestamp;

pub const SEGMENT_MAX_SIZE: i64 = 10_000;
const MAX_LAYER: usize = 16;

type Segment = Hnsw<'static, f32, DistL2>;

static SEGMENTS: OnceLock<Mutex<HashMap<i64, Segment>>> = OnceLock::new();

fn segments() -> &'static Mutex<HashMap<i64, Segment>> {
    SEGMENTS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Loads the ANN Engine's state into memory from SQLite when the process starts:
/// `sealed` segments are reloaded directly from their `index_blob`, the
/// `appendable` segment (if any) is rebuilt by reinserting its chunks in order.
pub fn reload(db: &Connection) -> Result<()> {
    reload_sealed_segments(db)?;
    reload_appendable_segment(db)?;
    Ok(())
}

fn reload_sealed_segments(db: &Connection) -> Result<()> {
    let mut sealed = Vec::new();
    {
        let mut stmt = db.prepare(
            "SELECT id, index_blob FROM segments WHERE status = 'sealed' AND index_blob IS NOT NULL",
        )?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            sealed.push((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?));
        }
    }

    for (segment_id, blob) in sealed {
        let hnsw = persistence::load_from_blob(segment_id, &blob).map_err(persistence::to_sql_error)?;
        segments().lock().unwrap().insert(segment_id, hnsw);
    }

    Ok(())
}

fn reload_appendable_segment(db: &Connection) -> Result<()> {
    let appendable: Option<(i64, usize, usize)> = db
        .query_row(
            "SELECT id, m, ef_construction FROM segments WHERE status = 'appendable'",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)? as usize,
                    row.get::<_, i64>(2)? as usize,
                ))
            },
        )
        .optional()?;

    let Some((segment_id, m, ef_construction)) = appendable else {
        return Ok(());
    };

    let hnsw =
        Hnsw::<f32, DistL2>::new(m, SEGMENT_MAX_SIZE as usize, MAX_LAYER, ef_construction, DistL2 {});

    let mut stmt =
        db.prepare("SELECT chunk_id, vector FROM embeddings WHERE segment_id = ?1 ORDER BY chunk_id")?;
    let mut rows = stmt.query([segment_id])?;
    while let Some(row) = rows.next()? {
        let chunk_id: i64 = row.get(0)?;
        let vector: Vec<u8> = row.get(1)?;
        let normalized = normalize(&bytes_to_f32(&vector));
        hnsw.insert((normalized.as_slice(), chunk_id as usize));
    }

    segments().lock().unwrap().insert(segment_id, hnsw);
    Ok(())
}

pub fn insert(db: &Connection, chunk_id: i64, vector: &[f32]) -> Result<i64> {
    let normalized = normalize(vector);
    let (segment_id, m, ef_construction, node_count) = current_appendable_segment(db)?;

    {
        let mut segments = segments().lock().unwrap();
        let hnsw = segments.entry(segment_id).or_insert_with(|| {
            Hnsw::<f32, DistL2>::new(m, SEGMENT_MAX_SIZE as usize, MAX_LAYER, ef_construction, DistL2 {})
        });
        hnsw.insert((normalized.as_slice(), chunk_id as usize));
    }

    let new_count = node_count + 1;
    db.execute(
        "UPDATE segments SET node_count = ?1 WHERE id = ?2",
        (new_count, segment_id),
    )?;

    if new_count >= SEGMENT_MAX_SIZE {
        let blob = {
            let segments = segments().lock().unwrap();
            let hnsw = segments
                .get(&segment_id)
                .expect("el segmento recien insertado debe estar en memoria");
            persistence::dump_to_blob(segment_id, hnsw).map_err(persistence::to_sql_error)?
        };
        db.execute(
            "UPDATE segments SET status = 'sealed', sealed_at = ?1, index_blob = ?2 WHERE id = ?3",
            (unix_timestamp(), blob, segment_id),
        )?;
    }

    Ok(segment_id)
}

pub fn search(db: &Connection, query: &[f32], top_k: usize) -> Result<Vec<(i64, f32)>> {
    let normalized_query = normalize(query);
    let ef_search = settings::get_usize(db, "ef_search", 100)?;

    let mut segment_ids = Vec::new();
    {
        let mut stmt = db.prepare("SELECT id FROM segments")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            segment_ids.push(row.get::<_, i64>(0)?);
        }
    }

    // HNSW is used only to discover *which* chunk_ids are nearby (fast,
    // approximate) — hnswlib-rs has no point-deletion, so re-storing an
    // embedding for a chunk_id that hasn't moved to a new segment yet leaves
    // both the old and the new vector as independent nodes in the same
    // segment, indistinguishable by segment alone. So the distance reported
    // by the graph traversal is never trusted directly: below, every
    // candidate chunk_id is deduplicated and its distance recomputed from the
    // live `embeddings.vector`, which also drops chunk_ids that were deleted
    // outright (no row left to recompute against).
    let mut candidate_ids: HashSet<i64> = HashSet::new();
    {
        let segments = segments().lock().unwrap();
        for segment_id in segment_ids {
            if let Some(hnsw) = segments.get(&segment_id) {
                for neighbour in hnsw.search(&normalized_query, top_k, ef_search) {
                    candidate_ids.insert(neighbour.get_origin_id() as i64);
                }
            }
        }
    }

    let ids: Vec<i64> = candidate_ids.into_iter().collect();
    let mut candidates: Vec<(i64, f32)> = current_vectors_for_chunks(db, &ids)?
        .into_iter()
        .map(|(chunk_id, vector)| {
            let distance = l2_distance(&normalized_query, &normalize(&vector));
            (chunk_id, distance)
        })
        .collect();

    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    candidates.truncate(top_k);

    Ok(candidates)
}

/// Current `(chunk_id, vector)` for each of `chunk_ids` that still has a live
/// row in `embeddings`, batched into one query. A `chunk_id` missing from the
/// result has no live embedding at all (deleted) and is silently dropped.
fn current_vectors_for_chunks(db: &Connection, chunk_ids: &[i64]) -> Result<Vec<(i64, Vec<f32>)>> {
    if chunk_ids.is_empty() {
        return Ok(Vec::new());
    }

    let placeholders = vec!["?"; chunk_ids.len()].join(",");
    let sql = format!("SELECT chunk_id, vector FROM embeddings WHERE chunk_id IN ({placeholders})");

    let mut stmt = db.prepare(&sql)?;
    let mut rows = stmt.query(params_from_iter(chunk_ids.iter()))?;

    let mut live = Vec::new();
    while let Some(row) = rows.next()? {
        let chunk_id: i64 = row.get(0)?;
        let vector: Vec<u8> = row.get(1)?;
        live.push((chunk_id, bytes_to_f32(&vector)));
    }

    Ok(live)
}

/// Mirrors `hnsw_rs::DistL2` (design section 5) so the recomputed distance in
/// `search` is exactly what the graph would have reported for a fresh node.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// `semanta_rebuild_graph()`: full manual reset of the ANN Engine (section 5).
/// Deletes all segments (in memory and in `segments`) and reinserts every
/// already-stored embedding, in `chunk_id` order, under the current `settings`
/// — the same path `semanta_store_embedding` uses on first insert, so the
/// result is indistinguishable from having loaded the corpus from scratch with
/// a uniform `M`/`ef_construction`. `relations` is untouched: re-evaluating
/// candidates after a rebuild is out of scope (design section 11).
///
/// Between deleting `segments` and finishing the reinsert loop, live
/// `embeddings.segment_id` rows briefly point at segments that no longer
/// exist (each row is only fixed up once its own turn in the loop comes up)
/// — a real `FOREIGN KEY` violation on any connection that enforces them
/// (found by testing with one that does). `PRAGMA defer_foreign_keys` inside
/// a `SAVEPOINT` postpones that check to `RELEASE`, by which point every row
/// has been repointed at a real segment; `SAVEPOINT` (not `BEGIN`) so this
/// still nests inside a transaction the caller may already have open, same
/// reasoning as `document::delete_document`.
pub fn rebuild(db: &Connection) -> Result<i64> {
    db.execute("SAVEPOINT semanta_rebuild_graph", [])?;
    db.execute("PRAGMA defer_foreign_keys = ON", [])?;

    let result: Result<i64> = (|| {
        segments().lock().unwrap().clear();
        db.execute("DELETE FROM segments", [])?;

        let mut rows = Vec::new();
        {
            let mut stmt = db.prepare("SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id")?;
            let mut query_rows = stmt.query([])?;
            while let Some(row) = query_rows.next()? {
                rows.push((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?));
            }
        }

        let reindexed = rows.len() as i64;
        for (chunk_id, vector) in rows {
            let query_vector = bytes_to_f32(&vector);
            let segment_id = insert(db, chunk_id, &query_vector)?;
            db.execute(
                "UPDATE embeddings SET segment_id = ?1 WHERE chunk_id = ?2",
                (segment_id, chunk_id),
            )?;
        }

        Ok(reindexed)
    })();

    match result {
        Ok(reindexed) => {
            db.execute("RELEASE semanta_rebuild_graph", [])?;
            Ok(reindexed)
        }
        Err(err) => {
            let _ = db.execute("ROLLBACK TO semanta_rebuild_graph", []);
            let _ = db.execute("RELEASE semanta_rebuild_graph", []);
            Err(err)
        }
    }
}

/// `semanta_graph_stats()`: per-segment `total_nodes` (historical, everything
/// ever inserted, `segments.node_count`) vs `live_nodes` (rows in `embeddings`
/// still pointing at that segment) — the gap between them is dead weight left
/// behind by updates/deletes that `search`'s phantom filter already hides from
/// results, but that `semanta_rebuild_graph()` is the only way to reclaim
/// (design section 10). Exposed so the caller decides when a rebuild is worth
/// it, instead of Semanta guessing with a background process or a threshold.
pub fn stats(db: &Connection) -> Result<String> {
    let mut segments_info = Vec::new();
    {
        let mut stmt = db.prepare("SELECT id, status, node_count FROM segments ORDER BY id")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            segments_info.push((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ));
        }
    }

    let mut items = Vec::with_capacity(segments_info.len());
    for (segment_id, status, total_nodes) in segments_info {
        let live_nodes: i64 = db.query_row(
            "SELECT COUNT(*) FROM embeddings WHERE segment_id = ?1",
            [segment_id],
            |row| row.get(0),
        )?;
        items.push(format!(
            "{{\"segment_id\":{segment_id},\"status\":\"{status}\",\"total_nodes\":{total_nodes},\"live_nodes\":{live_nodes}}}"
        ));
    }

    Ok(format!("[{}]", items.join(",")))
}

fn current_appendable_segment(db: &Connection) -> Result<(i64, usize, usize, i64)> {
    let existing = db
        .query_row(
            "SELECT id, m, ef_construction, node_count FROM segments WHERE status = 'appendable'",
            [],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)? as usize,
                    row.get::<_, i64>(2)? as usize,
                    row.get::<_, i64>(3)?,
                ))
            },
        )
        .optional()?;

    if let Some(segment) = existing {
        return Ok(segment);
    }

    let m = settings::get_usize(db, "m", 16)?;
    let ef_construction = settings::get_usize(db, "ef_construction", 200)?;
    let created_at = unix_timestamp();

    db.execute(
        "INSERT INTO segments (status, m, ef_construction, node_count, created_at) VALUES ('appendable', ?1, ?2, 0, ?3)",
        (m as i64, ef_construction as i64, created_at),
    )?;
    let segment_id = db.last_insert_rowid();

    Ok((segment_id, m, ef_construction, 0))
}

/// hnsw_rs::DistL2 returns `‖a-b‖` (not squared) over vectors already normalized
/// to unit norm. For unit vectors, ‖a-b‖² = 2 - 2·cos(a,b), so cos(a,b) =
/// 1 - distance²/2 recovers cosine similarity exactly (design section 5). Lives
/// here because the ANN Engine is the one that knows the internal metric.
pub fn distance_to_similarity(distance: f32) -> f32 {
    1.0 - (distance * distance) / 2.0
}

fn normalize(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vector.to_vec();
    }
    vector.iter().map(|x| x / norm).collect()
}

/// JSON format shared by `semanta_store_embedding` and `semanta_get_candidates`
/// to return candidates (chunk_id, distance) to the user, who passes them to
/// their own LLM to evaluate relations (Graph Engine, design section 6).
pub fn candidates_to_json(candidates: &[(i64, f32)]) -> String {
    let items: Vec<String> = candidates
        .iter()
        .map(|(chunk_id, distance)| format!("{{\"chunk_id\":{},\"distance\":{}}}", chunk_id, distance))
        .collect();
    format!("[{}]", items.join(","))
}

pub(crate) fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
