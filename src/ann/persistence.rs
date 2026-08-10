use hnsw_rs::prelude::*;
use rusqlite::Error;

use super::Segment;

const DUMP_BASENAME: &str = "segment";

fn dump_dir(prefix: &str, segment_id: i64) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("semanta-{prefix}-{segment_id}"))
}

/// Dumps a sealed segment to a BLOB: hnswlib-rs only knows how to dump to files
/// (a `.hnsw.graph` + a `.hnsw.data`), so we go through a temp directory and
/// concatenate both files into a single BLOB, prefixed with the first file's
/// length so they can be split apart again on reload.
pub fn dump_to_blob(segment_id: i64, hnsw: &Segment) -> anyhow::Result<Vec<u8>> {
    let dir = dump_dir("dump", segment_id);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    let basename = hnsw.file_dump(&dir, DUMP_BASENAME)?;
    let graph_bytes = std::fs::read(dir.join(format!("{basename}.hnsw.graph")))?;
    let data_bytes = std::fs::read(dir.join(format!("{basename}.hnsw.data")))?;
    let _ = std::fs::remove_dir_all(&dir);

    let mut blob = Vec::with_capacity(4 + graph_bytes.len() + data_bytes.len());
    blob.extend_from_slice(&(graph_bytes.len() as u32).to_le_bytes());
    blob.extend_from_slice(&graph_bytes);
    blob.extend_from_slice(&data_bytes);
    Ok(blob)
}

pub fn load_from_blob(segment_id: i64, blob: &[u8]) -> anyhow::Result<Segment> {
    let graph_len = u32::from_le_bytes(blob[0..4].try_into()?) as usize;
    let graph_bytes = &blob[4..4 + graph_len];
    let data_bytes = &blob[4 + graph_len..];

    let dir = dump_dir("load", segment_id);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(format!("{DUMP_BASENAME}.hnsw.graph")), graph_bytes)?;
    std::fs::write(dir.join(format!("{DUMP_BASENAME}.hnsw.data")), data_bytes)?;

    let io: &'static mut HnswIo = Box::leak(Box::new(HnswIo::new(&dir, DUMP_BASENAME)));
    let hnsw = io.load_hnsw::<f32, DistL2>()?;

    let _ = std::fs::remove_dir_all(&dir);
    Ok(hnsw)
}

pub fn to_sql_error(err: anyhow::Error) -> Error {
    Error::UserFunctionError(err.to_string().into())
}
