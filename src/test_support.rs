//! Shared setup for the `testing`-feature integration test suites in
//! `ann::tests`, `document::tests`, and `embedding::tests`.
//!
//! `ann`'s in-memory segment state lives in a process-global `static`, not
//! per-connection (see `ann::segments`) — real usage assumes one SQLite
//! connection per loaded extension process, but `cargo test` runs tests in
//! parallel in one process, so two tests with unrelated `:memory:` databases
//! could both mint `segment_id = 1` and corrupt each other's HNSW segment.
//! `GUARD` serializes every test that touches the ANN Engine (through
//! `document`/`embedding`/`ann` directly) to work around that -- it has to
//! stay a single static shared across all three test modules, not one per
//! module, or tests in different modules could still race each other.

use std::sync::Mutex;

use rusqlite::Connection;

use crate::storage;

pub static GUARD: Mutex<()> = Mutex::new(());

pub fn setup() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    storage::create_schema(&db).unwrap();
    db
}

pub fn vec_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn set_chunking(db: &Connection, chunk_size: &str, overlap: &str, chars_per_token: &str) {
    db.execute(
        "UPDATE settings SET value = ?1 WHERE key = 'chunk_size'",
        [chunk_size],
    )
    .unwrap();
    db.execute(
        "UPDATE settings SET value = ?1 WHERE key = 'chunk_overlap'",
        [overlap],
    )
    .unwrap();
    db.execute(
        "UPDATE settings SET value = ?1 WHERE key = 'chars_per_token'",
        [chars_per_token],
    )
    .unwrap();
}

pub fn chunk_ids_by_position(db: &Connection, document_id: i64) -> Vec<i64> {
    let mut stmt = db
        .prepare("SELECT id FROM chunks WHERE document_id = ?1 ORDER BY position")
        .unwrap();
    stmt.query_map([document_id], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<i64>>>()
        .unwrap()
}

pub fn first_chunk_id(db: &Connection, document_id: i64) -> i64 {
    chunk_ids_by_position(db, document_id)[0]
}
