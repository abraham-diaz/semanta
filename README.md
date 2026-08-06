# Semanta

Semanta is an embedded semantic engine for SQLite: a native Rust extension,
loadable via `load_extension(...)` from any client that speaks SQLite
(Python, a CLI, Node, C#...). The real interface is SQL — tables and
functions — not a library tied to one language.

## What makes it different

Semanta doesn't compete on "yet another vector index in SQLite"
(`sqlite-vec`, `sqlite-vss`, pgvector and friends already cover that well).
What nobody else offers in a single package is the **Graph Engine**: HNSW
isn't the product, it's the cheap mechanism to propose candidates to an
external LLM, which decides whether a real semantic relation exists between
two chunks — free-form vocabulary, no closed catalog — and Semanta persists
it. It builds a knowledge graph without Semanta ever having to "understand"
content.

- Pure vector stores give you nearest neighbours, not persisted relations —
  you'd build that on top yourself.
- Graph databases (Neo4j and the like) give you the graph, but with no
  native integration for the "find me semantic candidates to evaluate" step.

`semanta_search` closes the loop: it expands the ANN's nearest-neighbour
candidates through that relation graph, so results include chunks that are
far apart in embedding space but connected by a relation an LLM already
confirmed — something a plain vector store can't produce.

## Status

The core described in `Semanta_Design.md` (sections 1–9) is implemented and
verified end-to-end. All six SQL functions below work against a real
SQLite connection with HNSW-backed search, segment sealing/reload across
process restarts, and graph-expanded ranking.

There is no automated test suite yet beyond a few unit tests for chunking —
everything else has been verified manually by loading the compiled
extension from Python's `sqlite3` module. See "Out of scope" below for
what's deliberately not built yet.

## Design principles (already decided, see `Semanta_Design.md`)

- **BYOE (Bring Your Own Embeddings)**: Semanta never generates embeddings
  or calls an LLM itself. Both are supplied by the caller from their own
  code.
- **HNSW via `hnsw_rs`**, not a hand-rolled implementation — deliberately
  avoids the recall-bug risk of writing HNSW from scratch without benchmark
  infrastructure.
- **Metric**: vectors are normalized internally and indexed with `DistL2`
  (equivalent in ranking order to cosine similarity for unit vectors, while
  keeping HNSW's metric-space guarantees intact). Callers never need to
  normalize anything themselves.
- **Segmented graph**, Qdrant-style: HNSW never has edges crossing segment
  boundaries; global recall is resolved at query time via fan-out + merge,
  never by enriching the structure between segments.

## Architecture

`src/` has one folder per engine from the design doc. Each `mod.rs` is that
engine's public API; private submodules hold implementation details that
don't need to leak out.

```
src/
  lib.rs                 SQL function registration — no business logic
  util.rs                 shared helpers (timestamps, JSON escaping)
  storage/
    mod.rs                 schema (CREATE TABLE ...)
    settings.rs             key-value config accessors, used by every engine
  document/
    mod.rs                  semanta_add_document: extraction + chunking
    chunk.rs                 fixed-size chunking with overlap
  embedding/
    mod.rs                   semanta_store_embedding (BYOE)
  ann/
    mod.rs                   HNSW segments: insert, search, rebuild, reload
    persistence.rs            BLOB dump/reload for sealed segments
  graph/
    mod.rs                   semanta_store_relation, semanta_get_candidates
  ranking/
    mod.rs                   semanta_search: orchestrates ann + expand
    expand.rs                 Ranking Engine: graph expansion and scoring
```

## SQL surface

| Function | Description |
|---|---|
| `semanta_add_document(name, content, metadata, tags)` | Extracts and chunks a document (Markdown for now), returns the new `document_id`. |
| `semanta_store_embedding(chunk_id, vector, model_name, dimension)` | Stores a chunk's embedding, inserts it into the current HNSW segment (sealing it if it hits the size cap), and returns nearest-neighbour candidates as JSON for the caller to evaluate with their own LLM. |
| `semanta_get_candidates(chunk_id, top_k?)` | Re-queries the current nearest neighbours of an already-stored chunk, without reinserting it. |
| `semanta_store_relation(from_chunk_id, to_chunk_id, relation_type, confidence)` | Persists a semantic relation between two chunks (or records the pair as "evaluated, no relation" with a `NULL` type). |
| `semanta_search(query_vector, top_k?, expand?)` | Semantic search: ANN fan-out + merge across segments, then (by default) expands and reorders results through the relations graph. `expand = 0` returns pure ANN results. |
| `semanta_rebuild_graph()` | Full manual reset of the ANN index: deletes all segments and reinserts every stored embedding under the current settings. |

Vectors are passed as raw little-endian `f32` bytes (4 bytes per
dimension) — the same layout NumPy/`struct.pack` produce, no wrapper
format required.

## Configuration

Everything lives in the `settings` table (plain `key`/`value`, editable
with normal `UPDATE`/`INSERT`):

| Key | Default | Role |
|---|---|---|
| `chars_per_token` | `3.5` | token-length estimate for chunking (no real tokenizer) |
| `chunk_size` | `1000` | target chunk size, in estimated tokens |
| `chunk_overlap` | `0.125` | overlap ratio between consecutive chunks |
| `m` | `16` | HNSW neighbours per node/layer |
| `ef_construction` | `200` | HNSW search width on insert |
| `ef_search` | `100` | HNSW search width on query |
| `top_k` | `16` | default candidates returned |
| `max_hops` | `1` | graph expansion depth from each anchor |
| `hop_decay` | `0.5` | multiplicative penalty per extra hop |
| `expand_relation_types` | *(empty = all)* | comma-separated subset of `relation_type` to follow when expanding |

Changes only affect future operations (new chunks, new searches) —
`semanta_rebuild_graph()` is the explicit way to apply a structural change
retroactively.

## Building

```sh
cargo build --release
```

Produces `target/release/libsemanta.so` (or `.dylib`/`.dll` depending on
the platform). Load it from any SQLite client that supports extensions:

```python
import sqlite3

con = sqlite3.connect("my.db")
con.enable_load_extension(True)
con.load_extension("target/release/libsemanta")
con.enable_load_extension(False)
```

## Quick example

```python
import struct

def to_bytes(vector):
    return struct.pack(f"<{len(vector)}f", *vector)

# 1. Ingest — chunking is automatic, embeddings are BYOE
doc_id = con.execute(
    "SELECT semanta_add_document(?, ?, NULL, NULL)", ("manual.md", markdown_text)
).fetchone()[0]

for chunk_id, text in con.execute("SELECT id, text FROM chunks WHERE document_id = ?", (doc_id,)):
    vector = my_embedding_model.encode(text)  # your own model
    candidates = con.execute(
        "SELECT semanta_store_embedding(?, ?, 'my-model', ?)",
        (chunk_id, to_bytes(vector), len(vector)),
    ).fetchone()[0]
    # candidates = [{"chunk_id": ..., "distance": ...}, ...]
    # hand these to your own LLM, then persist what it decides:
    con.execute("SELECT semanta_store_relation(?, ?, 'continues', 0.9)", (chunk_id, other_chunk_id))

# 2. Search — ANN candidates expanded through the relation graph
query_vector = my_embedding_model.encode("how do I change the oil?")
results = con.execute(
    "SELECT semanta_search(?, 5, 1)", (to_bytes(query_vector),)
).fetchone()[0]
# [{"chunk_id": ..., "score": ..., "origin": "ann"|"graph", "relation_type": ..., "hop_count": ...}, ...]
```

## Out of scope (for now)

See section 10 of `Semanta_Design.md` for the full list — notably:
extractors beyond Markdown, structure-aware chunking, re-evaluating
candidates after a rebuild, a minimum similarity threshold, parallelizing
search across segments, and automated tests. None of these are implemented
without an explicit decision first.

## Further reading

- `Semanta_Overview.md` — product vision, what problem this solves.
- `Semanta_Design.md` — the technical design; source of truth for
  architecture decisions.
