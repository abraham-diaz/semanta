//! Integration tests for the update/delete flow (design section 10), calling
//! each engine's Rust API directly against an in-memory `Connection` — no
//! SQL scalar-function registration or extension loading needed, since
//! `lib.rs` only wires those functions to these same module functions.
//!
//! The normal build compiles `rusqlite` in loadable-extension mode (see the
//! `extension` Cargo feature), which gets the SQLite C API from the host
//! process at load time and has no usable standalone `Connection` — that
//! mode can't run these tests at all (`Connection::open_in_memory()` panics
//! with "SQLite API not initialized"). Run this suite with the `testing`
//! feature instead, which links a real SQLite:
//!
//!     cargo test --no-default-features --features testing
//!
//! `ann`'s in-memory segment state lives in a process-global `static`, not
//! per-connection (see `ann::segments`) — real usage assumes one SQLite
//! connection per loaded extension process, but `cargo test` runs tests in
//! parallel in one process, so two tests with unrelated `:memory:` databases
//! could both mint `segment_id = 1` and corrupt each other's HNSW segment.
//! `GUARD` serializes every test that touches the ANN Engine (through
//! `document`/`embedding`/`ann` directly) to work around that.

use std::sync::Mutex;

use rusqlite::{Connection, OptionalExtension};

use crate::{ann, document, embedding, graph, storage};

static GUARD: Mutex<()> = Mutex::new(());

fn setup() -> Connection {
    let db = Connection::open_in_memory().unwrap();
    storage::create_schema(&db).unwrap();
    db
}

fn vec_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn set_chunking(db: &Connection, chunk_size: &str, overlap: &str, chars_per_token: &str) {
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

fn chunk_ids_by_position(db: &Connection, document_id: i64) -> Vec<i64> {
    let mut stmt = db
        .prepare("SELECT id FROM chunks WHERE document_id = ?1 ORDER BY position")
        .unwrap();
    stmt.query_map([document_id], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<i64>>>()
        .unwrap()
}

fn first_chunk_id(db: &Connection, document_id: i64) -> i64 {
    chunk_ids_by_position(db, document_id)[0]
}

// --- documents: external_id, upsert, version --------------------------

#[test]
fn add_document_without_external_id_falls_back_to_internal_id() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = document::add_document(&db, "loose.md", "texto suelto", None, None, None).unwrap();

    let external_id: String = db
        .query_row(
            "SELECT external_id FROM documents WHERE id = ?1",
            [doc_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(external_id, doc_id.to_string());
}

#[test]
fn add_document_rejects_blank_external_id() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let err = document::add_document(&db, "x.md", "algo", None, None, Some("   ")).unwrap_err();
    assert!(err.to_string().contains("external_id"));

    let count: i64 = db.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 0, "a rejected external_id must not leave a half-inserted document");
}

#[test]
fn add_document_is_noop_when_hash_unchanged() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = document::add_document(&db, "a.md", "contenido", None, None, Some("ext-noop")).unwrap();
    let doc_id_again =
        document::add_document(&db, "a.md", "contenido", None, None, Some("ext-noop")).unwrap();

    assert_eq!(doc_id, doc_id_again);
    let version: i64 = db
        .query_row("SELECT version FROM documents WHERE id = ?1", [doc_id], |row| row.get(0))
        .unwrap();
    assert_eq!(version, 1);
}

#[test]
fn update_preserves_chunk_id_for_reused_position_and_deletes_orphaned_position() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();
    set_chunking(&db, "5", "0", "1"); // 5-char chunks, no overlap

    let doc_id = document::add_document(&db, "a.md", &"A".repeat(15), None, None, Some("ext-upd"))
        .unwrap(); // -> 3 chunks of 5 chars
    let chunks_v1 = chunk_ids_by_position(&db, doc_id);
    assert_eq!(chunks_v1.len(), 3);

    document::add_document(&db, "a.md", &"B".repeat(10), None, None, Some("ext-upd")).unwrap(); // -> 2 chunks

    let version: i64 = db
        .query_row("SELECT version FROM documents WHERE id = ?1", [doc_id], |row| row.get(0))
        .unwrap();
    assert_eq!(version, 2);

    let chunks_v2 = chunk_ids_by_position(&db, doc_id);
    assert_eq!(chunks_v2, &chunks_v1[..2], "surviving positions must keep their chunk_id");

    let orphan_still_there: Option<i64> = db
        .query_row("SELECT id FROM chunks WHERE id = ?1", [chunks_v1[2]], |row| row.get(0))
        .optional()
        .unwrap();
    assert!(orphan_still_there.is_none(), "position dropped by the shorter document must be deleted");

    let chunk0_version: i64 = db
        .query_row("SELECT version FROM chunks WHERE id = ?1", [chunks_v2[0]], |row| row.get(0))
        .unwrap();
    assert_eq!(chunk0_version, 2, "a reused position's version must follow the document's");
}

#[test]
fn update_purges_relations_touching_changed_or_removed_chunks() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();
    set_chunking(&db, "5", "0", "1");

    let doc_id =
        document::add_document(&db, "a.md", &"A".repeat(15), None, None, Some("ext-rel")).unwrap();
    let chunks = chunk_ids_by_position(&db, doc_id);
    graph::store_relation(&db, chunks[0], chunks[1], Some("continues"), Some(0.9)).unwrap();
    graph::store_relation(&db, chunks[1], chunks[2], Some("continues"), Some(0.9)).unwrap();

    document::add_document(&db, "a.md", &"B".repeat(10), None, None, Some("ext-rel")).unwrap();

    let relation_count: i64 = db.query_row("SELECT COUNT(*) FROM relations", [], |row| row.get(0)).unwrap();
    assert_eq!(
        relation_count, 0,
        "relations judged on old text (or on a now-deleted chunk) must not survive an update"
    );
}

// --- embeddings: upsert --------------------------------------------------

#[test]
fn store_embedding_upserts_same_chunk_id_without_error() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = document::add_document(&db, "a.md", "hola", None, None, Some("ext-emb")).unwrap();
    let chunk_id = first_chunk_id(&db, doc_id);

    embedding::store_embedding(&db, chunk_id, &vec_bytes(&[1.0, 0.0, 0.0]), "m", 3).unwrap();
    embedding::store_embedding(&db, chunk_id, &vec_bytes(&[0.0, 1.0, 0.0]), "m", 3).unwrap();

    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM embeddings WHERE chunk_id = ?1", [chunk_id], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 1, "re-storing an embedding must replace the row, not duplicate the PK");
}

// --- ann: candidate discovery vs. live-vector distance -------------------

#[test]
fn search_recomputes_distance_from_live_vector_and_dedupes_stale_node() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = document::add_document(&db, "a.md", "hola", None, None, Some("ext-search")).unwrap();
    let chunk_id = first_chunk_id(&db, doc_id);

    // Replace the embedding while the segment is still appendable -- old and
    // new nodes for the same chunk_id land in the same segment, which is
    // exactly the case segment-id comparison alone could not tell apart.
    embedding::store_embedding(&db, chunk_id, &vec_bytes(&[1.0, 0.0, 0.0]), "m", 3).unwrap();
    embedding::store_embedding(&db, chunk_id, &vec_bytes(&[0.0, 0.0, 1.0]), "m", 3).unwrap();

    let results_for_old_vector = ann::search(&db, &[1.0, 0.0, 0.0], 5).unwrap();
    assert_eq!(
        results_for_old_vector.len(),
        1,
        "the stale node must not show up as a second, duplicate hit for the same chunk_id"
    );
    assert_eq!(results_for_old_vector[0].0, chunk_id);

    let results_for_new_vector = ann::search(&db, &[0.0, 0.0, 1.0], 5).unwrap();
    assert_eq!(results_for_new_vector.len(), 1);
    assert_eq!(results_for_new_vector[0].0, chunk_id);

    // Querying with the vector that is no longer current must score worse
    // than querying with the one that actually is -- proves the distance is
    // recomputed against embeddings.vector, not trusted from the graph hit.
    assert!(results_for_old_vector[0].1 > results_for_new_vector[0].1);
}

#[test]
fn search_never_returns_a_deleted_chunk() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = document::add_document(&db, "a.md", "hola", None, None, Some("ext-del-search")).unwrap();
    let chunk_id = first_chunk_id(&db, doc_id);
    embedding::store_embedding(&db, chunk_id, &vec_bytes(&[1.0, 0.0, 0.0]), "m", 3).unwrap();

    document::delete_document(&db, "ext-del-search").unwrap();

    let results = ann::search(&db, &[1.0, 0.0, 0.0], 5).unwrap();
    assert!(results.is_empty(), "a deleted chunk's dead HNSW node must never resurface as a phantom");
}

#[test]
fn graph_stats_reports_total_vs_live_nodes_per_segment() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = document::add_document(&db, "a.md", "hola", None, None, Some("ext-stats")).unwrap();
    let chunk_id = first_chunk_id(&db, doc_id);
    embedding::store_embedding(&db, chunk_id, &vec_bytes(&[1.0, 0.0, 0.0]), "m", 3).unwrap();
    embedding::store_embedding(&db, chunk_id, &vec_bytes(&[0.0, 1.0, 0.0]), "m", 3).unwrap();

    let stats_json = ann::stats(&db).unwrap();
    assert!(stats_json.contains("\"total_nodes\":2"), "{stats_json}");
    assert!(stats_json.contains("\"live_nodes\":1"), "{stats_json}");
}

#[test]
fn rebuild_graph_reindexes_only_live_embeddings() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_a = document::add_document(&db, "a.md", "hola", None, None, Some("ext-rebuild-a")).unwrap();
    let doc_b = document::add_document(&db, "b.md", "mundo", None, None, Some("ext-rebuild-b")).unwrap();
    embedding::store_embedding(
        &db,
        first_chunk_id(&db, doc_a),
        &vec_bytes(&[1.0, 0.0, 0.0]),
        "m",
        3,
    )
    .unwrap();
    embedding::store_embedding(
        &db,
        first_chunk_id(&db, doc_b),
        &vec_bytes(&[0.0, 1.0, 0.0]),
        "m",
        3,
    )
    .unwrap();

    document::delete_document(&db, "ext-rebuild-a").unwrap();

    let reindexed = ann::rebuild(&db).unwrap();
    assert_eq!(reindexed, 1);

    let stats_json = ann::stats(&db).unwrap();
    assert!(stats_json.contains("\"total_nodes\":1"), "{stats_json}");
    assert!(stats_json.contains("\"live_nodes\":1"), "{stats_json}");
}

// --- semanta_delete_document ----------------------------------------------

#[test]
fn delete_document_removes_everything_it_owns() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = document::add_document(&db, "a.md", "hola", None, None, Some("ext-full-del")).unwrap();
    let chunk_id = first_chunk_id(&db, doc_id);
    embedding::store_embedding(&db, chunk_id, &vec_bytes(&[1.0, 0.0, 0.0]), "m", 3).unwrap();

    let deleted = document::delete_document(&db, "ext-full-del").unwrap();
    assert!(deleted);

    let documents: i64 = db.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0)).unwrap();
    let chunks: i64 = db.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0)).unwrap();
    let embeddings: i64 = db.query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0)).unwrap();
    assert_eq!((documents, chunks, embeddings), (0, 0, 0));
}

#[test]
fn delete_document_errors_when_external_id_is_missing() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let err = document::delete_document(&db, "does-not-exist").unwrap_err();
    assert!(err.to_string().contains("does-not-exist"));
}

/// Regression test for the bug `update_delete_demo.py` caught: the first
/// implementation used bare `BEGIN`, which fails with "cannot start a
/// transaction within a transaction" the moment the caller's own connection
/// already has one open (e.g. Python's `sqlite3` module auto-opens one
/// before the first write). `SAVEPOINT` must nest cleanly instead.
#[test]
fn delete_document_works_inside_callers_open_transaction() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    document::add_document(&db, "a.md", "hola", None, None, Some("ext-txn")).unwrap();

    db.execute("BEGIN", []).unwrap();
    let result = document::delete_document(&db, "ext-txn");
    assert!(result.is_ok(), "{result:?}");
    db.execute("COMMIT", []).unwrap();

    let count: i64 = db.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 0);
}
