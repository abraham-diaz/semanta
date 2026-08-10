//! `search` (live-vector distance, dedup, tombstoning), `stats`, and
//! `rebuild`.

use crate::test_support::{GUARD, first_chunk_id, setup, vec_bytes};
use crate::{document, embedding};

use super::*;

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

    let results_for_old_vector = search(&db, &[1.0, 0.0, 0.0], 5).unwrap();
    assert_eq!(
        results_for_old_vector.len(),
        1,
        "the stale node must not show up as a second, duplicate hit for the same chunk_id"
    );
    assert_eq!(results_for_old_vector[0].0, chunk_id);

    let results_for_new_vector = search(&db, &[0.0, 0.0, 1.0], 5).unwrap();
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

    let results = search(&db, &[1.0, 0.0, 0.0], 5).unwrap();
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

    let stats_json = stats(&db).unwrap();
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

    let reindexed = rebuild(&db).unwrap();
    assert_eq!(reindexed, 1);

    let stats_json = stats(&db).unwrap();
    assert!(stats_json.contains("\"total_nodes\":1"), "{stats_json}");
    assert!(stats_json.contains("\"live_nodes\":1"), "{stats_json}");
}
