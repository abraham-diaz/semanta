use std::os::raw::{c_char, c_int};

use rusqlite::functions::FunctionFlags;
use rusqlite::{Connection, Result, ffi};

mod ann;
mod chunk;
mod document;
mod embedding;
mod graph;
mod ranking;
mod search;
mod settings;
mod storage;
mod util;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sqlite3_semanta_init(
    db: *mut ffi::sqlite3,
    pz_err_msg: *mut *mut c_char,
    p_api: *mut ffi::sqlite3_api_routines,
) -> c_int {
    unsafe { Connection::extension_init2(db, pz_err_msg, p_api, semanta_init) }
}

fn semanta_init(db: Connection) -> Result<bool> {
    storage::create_schema(&db)?;
    ann::reload(&db)?;

    db.create_scalar_function(
        "semanta_add_document",
        4,
        FunctionFlags::SQLITE_UTF8,
        |ctx| {
            let name: String = ctx.get(0)?;
            let content: String = ctx.get(1)?;
            let metadata: Option<String> = ctx.get(2)?;
            let tags: Option<String> = ctx.get(3)?;

            let conn = unsafe { ctx.get_connection()? };
            document::add_document(&conn, &name, &content, metadata.as_deref(), tags.as_deref())
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
            search::search(&conn, &vector, top_k, expand)
        },
    )?;

    Ok(false)
}
