use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use hnsw_rs::prelude::*;
use rusqlite::{Connection, OptionalExtension, Result};

use crate::settings;
use crate::util::unix_timestamp;

pub const SEGMENT_MAX_SIZE: i64 = 10_000;
const MAX_LAYER: usize = 16;

type Segment = Hnsw<'static, f32, DistL2>;

static SEGMENTS: OnceLock<Mutex<HashMap<i64, Segment>>> = OnceLock::new();

fn segments() -> &'static Mutex<HashMap<i64, Segment>> {
    SEGMENTS.get_or_init(|| Mutex::new(HashMap::new()))
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
        db.execute(
            "UPDATE segments SET status = 'sealed', sealed_at = ?1 WHERE id = ?2",
            (unix_timestamp(), segment_id),
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

fn normalize(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vector.to_vec();
    }
    vector.iter().map(|x| x / norm).collect()
}
