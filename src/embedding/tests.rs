//! `store_embedding` (upsert by `chunk_id`).

use crate::document;
use crate::test_support::{GUARD, first_chunk_id, setup, vec_bytes};

use super::*;

#[test]
fn store_embedding_upserts_same_chunk_id_without_error() {
    let _guard = GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let db = setup();

    let doc_id = document::add_document(&db, "a.md", "hola", None, None, Some("ext-emb")).unwrap();
    let chunk_id = first_chunk_id(&db, doc_id);

    store_embedding(&db, chunk_id, &vec_bytes(&[1.0, 0.0, 0.0]), "m", 3).unwrap();
    store_embedding(&db, chunk_id, &vec_bytes(&[0.0, 1.0, 0.0]), "m", 3).unwrap();

    let count: i64 = db
        .query_row("SELECT COUNT(*) FROM embeddings WHERE chunk_id = ?1", [chunk_id], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(count, 1, "re-storing an embedding must replace the row, not duplicate the PK");
}
