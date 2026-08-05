use std::os::raw::{c_char, c_int};

use rusqlite::functions::FunctionFlags;
use rusqlite::{Connection, Result, ffi};

mod chunk;
mod document;
mod storage;

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

    Ok(false)
}
