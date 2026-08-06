use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

mod persistence;

use hnsw_rs::prelude::*;
use rusqlite::{Connection, OptionalExtension, Result};

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
    let normalized = normalize(query);
    let ef_search = settings::get_usize(db, "ef_search", 100)?;

    let mut segment_ids = Vec::new();
    {
        let mut stmt = db.prepare("SELECT id FROM segments")?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            segment_ids.push(row.get::<_, i64>(0)?);
        }
    }

    let mut candidates = Vec::new();
    {
        let segments = segments().lock().unwrap();
        for segment_id in segment_ids {
            if let Some(hnsw) = segments.get(&segment_id) {
                for neighbour in hnsw.search(&normalized, top_k, ef_search) {
                    candidates.push((neighbour.get_origin_id() as i64, neighbour.get_distance()));
                }
            }
        }
    }

    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    candidates.truncate(top_k);

    Ok(candidates)
}

/// `semanta_rebuild_graph()`: full manual reset of the ANN Engine (section 5).
/// Deletes all segments (in memory and in `segments`) and reinserts every
/// already-stored embedding, in `chunk_id` order, under the current `settings`
/// — the same path `semanta_store_embedding` uses on first insert, so the
/// result is indistinguishable from having loaded the corpus from scratch with
/// a uniform `M`/`ef_construction`. `relations` is untouched: re-evaluating
/// candidates after a rebuild is out of scope (design section 10).
pub fn rebuild(db: &Connection) -> Result<i64> {
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
