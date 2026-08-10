#[cfg(feature = "extension")]
use std::os::raw::{c_char, c_int};

#[cfg(feature = "extension")]
use rusqlite::functions::FunctionFlags;
#[cfg(feature = "extension")]
use rusqlite::{Connection, Result, ffi};

mod ann;
mod document;
mod embedding;
mod graph;
mod ranking;
mod storage;
// Needs the `testing` feature (real linked SQLite, see Cargo.toml) to even
// compile, so plain `cargo test` (default features = loadable-extension mode)
// just skips it instead of failing every test with "SQLite API not
// initialized" — run `cargo test --no-default-features --features testing`
// to actually run the `tests` submodules it backs (`ann::tests`,
// `document::tests`, `embedding::tests`).
#[cfg(all(test, feature = "testing"))]
mod test_support;
mod util;

// `extension_init2` only exists in rusqlite's loadable-extension mode (the
// `extension` Cargo feature, on by default) — the `testing` feature swaps in
// a normal linked SQLite instead, which has no such entry point and isn't
// needed for it: each engine's `tests` submodule calls document::/embedding::
// /ann::/graph:: directly, bypassing this FFI boundary entirely.
#[cfg(feature = "extension")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sqlite3_semanta_init(
    db: *mut ffi::sqlite3,
    pz_err_msg: *mut *mut c_char,
    p_api: *mut ffi::sqlite3_api_routines,
) -> c_int {
    unsafe { Connection::extension_init2(db, pz_err_msg, p_api, semanta_init) }
}

#[cfg(feature = "extension")]
fn semanta_init(db: Connection) -> Result<bool> {
    storage::create_schema(&db)?;
    ann::reload(&db)?;

    db.create_scalar_function(
        "semanta_add_document",
        -1,
        FunctionFlags::SQLITE_UTF8,
        |ctx| {
            let name: String = ctx.get(0)?;
            let content: String = ctx.get(1)?;
            let metadata: Option<String> = ctx.get(2)?;
            let tags: Option<String> = ctx.get(3)?;
            let external_id: Option<String> = if ctx.len() > 4 { ctx.get(4)? } else { None };

            let conn = unsafe { ctx.get_connection()? };
            document::add_document(
                &conn,
                &name,
                &content,
                metadata.as_deref(),
                tags.as_deref(),
                external_id.as_deref(),
            )
        },
    )?;

    db.create_scalar_function(
        "semanta_delete_document",
        1,
        FunctionFlags::SQLITE_UTF8,
        |ctx| {
            let external_id: String = ctx.get(0)?;

            let conn = unsafe { ctx.get_connection()? };
            document::delete_document(&conn, &external_id)
        },
    )?;

    db.create_scalar_function(
        "semanta_store_embedding",
        4,
        FunctionFlags::SQLITE_UTF8,
        |ctx| {
            let chunk_id: i64 = ctx.get(0)?;
            let vector: Vec<u8> = ctx.get(1)?;
            let model_name: String = ctx.get(2)?;
            let dimension: i64 = ctx.get(3)?;

            let conn = unsafe { ctx.get_connection()? };
            embedding::store_embedding(&conn, chunk_id, &vector, &model_name, dimension)
        },
    )?;

    db.create_scalar_function(
        "semanta_store_relation",
        4,
        FunctionFlags::SQLITE_UTF8,
        |ctx| {
            let from_chunk_id: i64 = ctx.get(0)?;
            let to_chunk_id: i64 = ctx.get(1)?;
            let relation_type: Option<String> = ctx.get(2)?;
            let confidence: Option<f64> = ctx.get(3)?;

            let conn = unsafe { ctx.get_connection()? };
            graph::store_relation(
                &conn,
                from_chunk_id,
                to_chunk_id,
                relation_type.as_deref(),
                confidence,
            )
        },
    )?;

    db.create_scalar_function("semanta_rebuild_graph", 0, FunctionFlags::SQLITE_UTF8, |ctx| {
        let conn = unsafe { ctx.get_connection()? };
        ann::rebuild(&conn)
    })?;

    db.create_scalar_function("semanta_graph_stats", 0, FunctionFlags::SQLITE_UTF8, |ctx| {
        let conn = unsafe { ctx.get_connection()? };
        ann::stats(&conn)
    })?;

    db.create_scalar_function(
        "semanta_get_candidates",
        -1,
        FunctionFlags::SQLITE_UTF8,
        |ctx| {
            let chunk_id: i64 = ctx.get(0)?;
            let top_k: Option<i64> = if ctx.len() > 1 { ctx.get(1)? } else { None };

            let conn = unsafe { ctx.get_connection()? };
            graph::get_candidates(&conn, chunk_id, top_k)
        },
    )?;

    db.create_scalar_function(
        "semanta_search",
        -1,
        FunctionFlags::SQLITE_UTF8,
        |ctx| {
            let vector: Vec<u8> = ctx.get(0)?;
            let top_k: Option<i64> = if ctx.len() > 1 { ctx.get(1)? } else { None };
            let expand: Option<bool> = if ctx.len() > 2 { ctx.get(2)? } else { None };

            let conn = unsafe { ctx.get_connection()? };
            ranking::search(&conn, &vector, top_k, expand)
        },
    )?;

    Ok(false)
}
