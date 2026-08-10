//! `add_document` (upsert by `external_id`, design section 10) and
//! `delete_document`.

use rusqlite::OptionalExtension;

use crate::graph;
use crate::test_support::{GUARD, chunk_ids_by_position, first_chunk_id, set_chunking, setup, vec_bytes};
use crate::embedding;

use super::*;

// --- documents: external_id, upsert, version --------------------------

#[test]
fn add_document_without_external_id_falls_back_to_internal_id() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = add_document(&db, "loose.md", "texto suelto", None, None, None).unwrap();

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

    let err = add_document(&db, "x.md", "algo", None, None, Some("   ")).unwrap_err();
    assert!(err.to_string().contains("external_id"));

    let count: i64 = db.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 0, "a rejected external_id must not leave a half-inserted document");
}

#[test]
fn add_document_is_noop_when_hash_unchanged() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = add_document(&db, "a.md", "contenido", None, None, Some("ext-noop")).unwrap();
    let doc_id_again = add_document(&db, "a.md", "contenido", None, None, Some("ext-noop")).unwrap();

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

    let doc_id = add_document(&db, "a.md", &"A".repeat(15), None, None, Some("ext-upd")).unwrap(); // -> 3 chunks of 5 chars
    let chunks_v1 = chunk_ids_by_position(&db, doc_id);
    assert_eq!(chunks_v1.len(), 3);

    add_document(&db, "a.md", &"B".repeat(10), None, None, Some("ext-upd")).unwrap(); // -> 2 chunks

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

    let doc_id = add_document(&db, "a.md", &"A".repeat(15), None, None, Some("ext-rel")).unwrap();
    let chunks = chunk_ids_by_position(&db, doc_id);
    graph::store_relation(&db, chunks[0], chunks[1], Some("continues"), Some(0.9)).unwrap();
    graph::store_relation(&db, chunks[1], chunks[2], Some("continues"), Some(0.9)).unwrap();

    add_document(&db, "a.md", &"B".repeat(10), None, None, Some("ext-rel")).unwrap();

    let relation_count: i64 = db.query_row("SELECT COUNT(*) FROM relations", [], |row| row.get(0)).unwrap();
    assert_eq!(
        relation_count, 0,
        "relations judged on old text (or on a now-deleted chunk) must not survive an update"
    );
}

// --- semanta_delete_document ----------------------------------------------

#[test]
fn delete_document_removes_everything_it_owns() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = add_document(&db, "a.md", "hola", None, None, Some("ext-full-del")).unwrap();
    let chunk_id = first_chunk_id(&db, doc_id);
    embedding::store_embedding(&db, chunk_id, &vec_bytes(&[1.0, 0.0, 0.0]), "m", 3).unwrap();

    let deleted = delete_document(&db, "ext-full-del").unwrap();
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

    let err = delete_document(&db, "does-not-exist").unwrap_err();
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

    add_document(&db, "a.md", "hola", None, None, Some("ext-txn")).unwrap();

    db.execute("BEGIN", []).unwrap();
    let result = delete_document(&db, "ext-txn");
    assert!(result.is_ok(), "{result:?}");
    db.execute("COMMIT", []).unwrap();

    let count: i64 = db.query_row("SELECT COUNT(*) FROM documents", [], |row| row.get(0)).unwrap();
    assert_eq!(count, 0);
}
