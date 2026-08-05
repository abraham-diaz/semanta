use rusqlite::{Connection, OptionalExtension, Result};

pub fn get_usize(db: &Connection, key: &str, default: usize) -> Result<usize> {
    Ok(get_raw(db, key)?
        .and_then(|value| value.parse().ok())
        .unwrap_or(default))
}

pub fn get_f64(db: &Connection, key: &str, default: f64) -> Result<f64> {
    Ok(get_raw(db, key)?
        .and_then(|value| value.parse().ok())
        .unwrap_or(default))
}

fn get_raw(db: &Connection, key: &str) -> Result<Option<String>> {
    db.query_row("SELECT value FROM settings WHERE key = ?1", [key], |row| {
        row.get(0)
    })
    .optional()
}
