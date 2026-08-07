use rusqlite::{Connection, Error, OptionalExtension, Result};

mod chunk;

use chunk::chunk_text;

use crate::storage::settings;
use crate::util::unix_timestamp;

pub trait DocumentExtractor {
    fn extract(&self, content: &str) -> String;
}

pub struct MarkdownExtractor;

impl DocumentExtractor for MarkdownExtractor {
    fn extract(&self, content: &str) -> String {
        content.to_string()
    }
}

const DEFAULT_CHUNK_SIZE_TOKENS: usize = 1000;
const DEFAULT_CHARS_PER_TOKEN: f64 = 3.5;
const DEFAULT_OVERLAP_RATIO: f64 = 0.125;

/// `semanta_add_document`: version-aware upsert (design section 2/10).
/// `external_id` is the caller's stable identity for the document (BYOE, same
/// principle as embeddings/relation vocabulary) — if omitted, the document can
/// never be recognised as "the same one" on a later call, so it's always
/// inserted as new (falling back to the internal `id` as its own `external_id`).
/// If `external_id` already exists: same `hash` is a pure no-op (content did
/// not change); different `hash` bumps `version` and re-chunks.
pub fn add_document(
    db: &Connection,
    name: &str,
    content: &str,
    metadata: Option<&str>,
    tags: Option<&str>,
    external_id: Option<&str>,
) -> Result<i64> {
    if let Some(id) = external_id
        && id.trim().is_empty()
    {
        return Err(Error::UserFunctionError(
            "external_id no puede ser una cadena vacía".into(),
        ));
    }

    let extracted = MarkdownExtractor.extract(content);
    let hash = content_hash(&extracted);

    let existing: Option<(i64, String)> = match external_id {
        Some(id) => db
            .query_row(
                "SELECT id, hash FROM documents WHERE external_id = ?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?,
        None => None,
    };

    if let Some((document_id, existing_hash)) = existing {
        if existing_hash == hash {
            return Ok(document_id);
        }
        return update_document(db, document_id, &extracted, &hash);
    }

    insert_document(db, name, &extracted, &hash, metadata, tags, external_id)
}

fn insert_document(
    db: &Connection,
    name: &str,
    extracted: &str,
    hash: &str,
    metadata: Option<&str>,
    tags: Option<&str>,
    external_id: Option<&str>,
) -> Result<i64> {
    let created_at = unix_timestamp();

    db.execute(
        "INSERT INTO documents (name, external_id, version, hash, created_at, metadata, tags) VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6)",
        (name, external_id, hash, created_at, metadata, tags),
    )?;
    let document_id = db.last_insert_rowid();

    // No caller-supplied identity: fall back to the internal id, which is
    // unique by construction and guarantees this document is never matched
    // against another one purely by accident (see design section 10).
    if external_id.is_none() {
        db.execute(
            "UPDATE documents SET external_id = ?1 WHERE id = ?2",
            (document_id.to_string(), document_id),
        )?;
    }

    upsert_chunks(db, document_id, extracted, 1)?;

    Ok(document_id)
}

fn update_document(db: &Connection, document_id: i64, extracted: &str, hash: &str) -> Result<i64> {
    let version: i64 = db.query_row(
        "UPDATE documents SET hash = ?1, version = version + 1 WHERE id = ?2 RETURNING version",
        (hash, document_id),
        |row| row.get(0),
    )?;

    upsert_chunks(db, document_id, extracted, version)?;

    Ok(document_id)
}

/// Re-chunks `extracted` and upserts by `(document_id, position)`: a position
/// that already existed keeps its `chunk_id` (so `embeddings`/`relations` FKs
/// stay valid) but its text and `version` are updated, and any relation
/// previously judged on the old text is purged. Positions that no longer exist
/// (the document got shorter) are deleted outright, chunk/embedding/relations
/// alike — design section 10.
fn upsert_chunks(db: &Connection, document_id: i64, extracted: &str, version: i64) -> Result<()> {
    let chunk_size_tokens = settings::get_usize(db, "chunk_size", DEFAULT_CHUNK_SIZE_TOKENS)?;
    let chars_per_token = settings::get_f64(db, "chars_per_token", DEFAULT_CHARS_PER_TOKEN)?;
    let overlap_ratio = settings::get_f64(db, "chunk_overlap", DEFAULT_OVERLAP_RATIO)?;

    let chunks = chunk_text(extracted, chunk_size_tokens, chars_per_token, overlap_ratio);

    for (position, text) in chunks.iter().enumerate() {
        let position = position as i64;
        let existing_chunk_id: Option<i64> = db
            .query_row(
                "SELECT id FROM chunks WHERE document_id = ?1 AND position = ?2",
                (document_id, position),
                |row| row.get(0),
            )
            .optional()?;

        match existing_chunk_id {
            Some(chunk_id) => {
                db.execute(
                    "UPDATE chunks SET text = ?1, version = ?2 WHERE id = ?3",
                    (text, version, chunk_id),
                )?;
                purge_relations_for_chunk(db, chunk_id)?;
            }
            None => {
                db.execute(
                    "INSERT INTO chunks (document_id, text, position, version) VALUES (?1, ?2, ?3, ?4)",
                    (document_id, text, position, version),
                )?;
            }
        }
    }

    delete_chunks_from_position(db, document_id, chunks.len() as i64)
}

fn delete_chunks_from_position(db: &Connection, document_id: i64, from_position: i64) -> Result<()> {
    let mut orphaned_ids = Vec::new();
    {
        let mut stmt =
            db.prepare("SELECT id FROM chunks WHERE document_id = ?1 AND position >= ?2")?;
        let mut rows = stmt.query((document_id, from_position))?;
        while let Some(row) = rows.next()? {
            orphaned_ids.push(row.get::<_, i64>(0)?);
        }
    }

    for chunk_id in orphaned_ids {
        delete_chunk_completely(db, chunk_id)?;
    }

    Ok(())
}

fn delete_chunk_completely(db: &Connection, chunk_id: i64) -> Result<()> {
    purge_relations_for_chunk(db, chunk_id)?;
    db.execute("DELETE FROM embeddings WHERE chunk_id = ?1", [chunk_id])?;
    db.execute("DELETE FROM chunks WHERE id = ?1", [chunk_id])?;
    Ok(())
}

fn purge_relations_for_chunk(db: &Connection, chunk_id: i64) -> Result<()> {
    db.execute(
        "DELETE FROM relations WHERE from_chunk_id = ?1 OR to_chunk_id = ?1",
        [chunk_id],
    )?;
    Ok(())
}

/// `semanta_delete_document`: hard delete, consistent with how `upsert_chunks`
/// already handles chunks that fall out of a document (design section 10) —
/// no tombstone, just removed. Runs inside a `SAVEPOINT` (not `BEGIN`) so a
/// failure partway through never leaves the document half-deleted, and so it
/// still works when the caller's own connection already has a transaction
/// open — `BEGIN` inside an existing transaction is an error, `SAVEPOINT`
/// nests fine either way (found by testing against Python's `sqlite3`
/// module, which opens an implicit transaction before the first write).
pub fn delete_document(db: &Connection, external_id: &str) -> Result<bool> {
    let document_id: i64 = db
        .query_row(
            "SELECT id FROM documents WHERE external_id = ?1",
            [external_id],
            |row| row.get(0),
        )
        .optional()?
        .ok_or_else(|| {
            Error::UserFunctionError(
                format!("no existe ningún documento con external_id '{external_id}'").into(),
            )
        })?;

    db.execute("SAVEPOINT semanta_delete_document", [])?;

    let result: Result<()> = (|| {
        let mut chunk_ids = Vec::new();
        {
            let mut stmt = db.prepare("SELECT id FROM chunks WHERE document_id = ?1")?;
            let mut rows = stmt.query([document_id])?;
            while let Some(row) = rows.next()? {
                chunk_ids.push(row.get::<_, i64>(0)?);
            }
        }

        for chunk_id in chunk_ids {
            delete_chunk_completely(db, chunk_id)?;
        }

        db.execute("DELETE FROM documents WHERE id = ?1", [document_id])?;
        Ok(())
    })();

    match result {
        Ok(()) => {
            db.execute("RELEASE semanta_delete_document", [])?;
            Ok(true)
        }
        Err(err) => {
            let _ = db.execute("ROLLBACK TO semanta_delete_document", []);
            let _ = db.execute("RELEASE semanta_delete_document", []);
            Err(err)
        }
    }
}

fn content_hash(content: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}
