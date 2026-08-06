use rusqlite::{Connection, Result};

pub fn create_schema(db: &Connection) -> Result<()> {
    db.execute_batch(SCHEMA_SQL)?;
    Ok(())
}

const SCHEMA_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS documents (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    hash       TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    metadata   TEXT,
    tags       TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id),
    text        TEXT NOT NULL,
    position    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS embedding_models (
    id           INTEGER PRIMARY KEY,
    model_name   TEXT NOT NULL,
    dimension    INTEGER NOT NULL,
    generated_at INTEGER NOT NULL,
    parameters   TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id                  INTEGER PRIMARY KEY,
    status              TEXT NOT NULL CHECK (status IN ('appendable', 'sealed')),
    m                   INTEGER NOT NULL,
    ef_construction     INTEGER NOT NULL,
    entry_point_node_id INTEGER,
    top_layer           INTEGER,
    node_count          INTEGER NOT NULL DEFAULT 0,
    index_blob          BLOB,
    created_at          INTEGER NOT NULL,
    sealed_at           INTEGER
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id           INTEGER PRIMARY KEY REFERENCES chunks(id),
    embedding_model_id INTEGER NOT NULL REFERENCES embedding_models(id),
    segment_id         INTEGER NOT NULL REFERENCES segments(id),
    vector             BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_segment ON embeddings(segment_id);

CREATE TABLE IF NOT EXISTS relations (
    from_chunk_id INTEGER NOT NULL REFERENCES chunks(id),
    to_chunk_id   INTEGER NOT NULL REFERENCES chunks(id),
    relation_type TEXT,
    confidence    REAL,
    PRIMARY KEY (from_chunk_id, to_chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_chunk_id);
CREATE INDEX IF NOT EXISTS idx_relations_to   ON relations(to_chunk_id);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO settings (key, value) VALUES
    ('chars_per_token',        '3.5'),
    ('chunk_size',             '1000'),
    ('chunk_overlap',          '0.125'),
    ('m',                      '16'),
    ('ef_construction',        '200'),
    ('ef_search',              '100'),
    ('top_k',                  '16'),
    ('max_hops',               '1'),
    ('hop_decay',              '0.5'),
    ('expand_relation_types',  '');
"#;
