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

/// `expand_relation_types` is stored as a comma-separated string; empty or
/// absent means "all types", matching `Semanta_Design.md` section 7.
pub fn get_string_list(db: &Connection, key: &str) -> Result<Option<Vec<String>>> {
    let raw = get_raw(db, key)?.unwrap_or_default();
    let items: Vec<String> = raw
        .split(',')
        .map(|item| item.trim().to_string())
        .filter(|item| !item.is_empty())
        .collect();

    Ok(if items.is_empty() { None } else { Some(items) })
}

fn get_raw(db: &Connection, key: &str) -> Result<Option<String>> {
    db.query_row("SELECT value FROM settings WHERE key = ?1", [key], |row| {
        row.get(0)
    })
    .optional()
}
