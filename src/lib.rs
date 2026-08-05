use std::os::raw::{c_char, c_int};

use rusqlite::{Connection, Result, ffi};

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
    Ok(false)
}
