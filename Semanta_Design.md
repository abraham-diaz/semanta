# Semanta — Technical Design

> This document lands the architecture decisions made from `Semanta_Overview.md`. Where the overview describes the vision, this document describes how it will be built.

------------------------------------------------------------------------

## 1. What it is, concretely

Semanta is a **native SQLite extension**, written in **Rust**, loadable via `load_extension(...)` from any client that speaks SQLite (Python, a CLI, Node, C#...). It is not a library tied to one language — the real interface is SQL: tables and functions.

Its responsibility is deliberately scoped:

- **What it does**: store documents/chunks/embeddings, index embeddings with HNSW (via the `hnsw_rs` crate), find semantically close candidates, expand them through the relations graph when searching, and store relations between chunks.
- **What it doesn't do**: generate embeddings, or call an LLM. Those two pieces of "intelligence" are provided by whoever uses Semanta, from their own code (for example, Python with `sentence-transformers` plus their own LLM).

This boundary is deliberate: it avoids tying Semanta to a specific embedding model, an LLM provider, or requiring the user to trust that Semanta implements that inference well inside a native extension written in a language (Rust) that was new to the author.

------------------------------------------------------------------------

## 2. Storage Engine (SQLite schema)

SQLite acts as pure storage. All search and indexing logic lives in memory, in the process that loads the extension, and is synchronized with these tables.

| Table | Role |
|---|---|
| `documents` | id, name, external_id (UNIQUE, nullable at the column level but always populated — see section 10), version, hash, created_at, metadata, tags |
| `chunks` | id, document_id (FK), text, position, version — UNIQUE(document_id, position) |
| `embedding_models` | id, model_name, dimension, generated_at, parameters — records which model generated each embedding (data supplied by the user, not computed by Semanta) |
| `embeddings` | chunk_id (FK 1:1), embedding_model_id (FK), segment_id (FK, indexed), vector (BLOB) — `segment_id` records which HNSW segment the vector was assigned to on insert, needed to rebuild the `appendable` segment on startup (see section 5): without this column there's no way to know which chunks to reinsert |
| `segments` | id, status (`appendable`/`sealed`), m, ef_construction, entry_point_node_id, top_layer, node_count, index_blob (BLOB, native `hnsw_rs` dump; `NULL` while the segment is `appendable`), created_at, sealed_at — each row is an independent, self-contained HNSW graph (see section 5). `m`/`ef_construction` are fixed per row when created: not read live from `settings` |
| `relations` | from_chunk_id, to_chunk_id, relation_type (free text, nullable), confidence — the semantic knowledge graph. Indexed on both chunk_id columns: queried in both directions on every search (Ranking Engine, section 7), not just on insert |
| `settings` | key, value — configurable engine parameters |

**Why a BLOB per segment instead of a row per node/edge**: with a hand-rolled HNSW, one row per node-layer matched the write pattern (rewriting neighbour lists on pruning). By delegating the algorithm to `hnsw_rs`, the index already lives serialized in the crate's own format (`hnswio`); replicating that as SQL rows would mean reinventing a serialization the crate already solves. The BLOB is opaque to Semanta except for what the crate itself exposes when reloading it.

**Why the vector is a plain BLOB**: depending on an external extension (`sqlite-vec`) was rejected as a piece outside our control. An embedding is just an array of floats serialized to bytes — no dependencies, no external libraries that could cause problems.

------------------------------------------------------------------------

## 3. Document/Chunk Engine

**Extraction**: a `DocumentExtractor`-style abstraction (a Rust trait) with per-format implementations. First implementation: **Markdown**. Other formats (PDF, etc.) are added later on the same interface, without touching the rest of the system — incremental growth, not everything at once.

**Chunking**: fixed size by estimated tokens (not aware of Markdown structure, at least in this first version), targeting **~1000 tokens** per chunk.

- Token counting: approximated by character count (no real tokenizer, so as not to couple the Chunk Engine to a specific model). Configurable ratio, defaulting to **~3.5 characters/token**, calibrated as an average between Spanish and French (the project's main languages), without closing the door to other languages.
- Overlap between consecutive chunks: configurable, defaulting to **~10-15%** of the chunk size, so context isn't lost at the cut boundaries.

------------------------------------------------------------------------

## 4. Embedding Engine — BYOE (Bring Your Own Embeddings)

Semanta **does not generate embeddings**. The user already has that step solved in their own stack (in this project: `sentence-transformers` from Python).

- **When ingesting**: the user computes the vector for each chunk and passes it to Semanta to store.
- **When searching**: the user computes the query vector and passes it to Semanta already vectorized — Semanta never tokenizes or infers anything from free text at this point.
- Semanta validates that the vector's dimension is consistent with the model registered in `embedding_models`, so incompatible vectors never get mixed into the same index.

------------------------------------------------------------------------

## 5. ANN Engine — HNSW via `hnsw_rs`

Implementing HNSW from scratch is rejected: it's an algorithm (Malkov-Yashunin) already solved by mature implementations, and hand-writing it adds no differentiation to Semanta — it does add real risk of subtle recall bugs, hard to detect without dedicated benchmarking infrastructure. The **`hnsw_rs`** crate is used: pure Rust (no native dependencies outside Rust, relevant for compiling/distributing the extension), incremental insertion, index dump/reload, actively maintained.

**Distance metric**: vectors are normalized internally to unit norm (O(d) cost, transparent to the caller) and indexed with the crate's `DistL2`. For unit vectors, `‖a-b‖² = 2 - 2·cos(a,b)`, so the resulting neighbour order is identical to cosine similarity. This is preferred over using `DistCosine` directly because L2 is the metric under which HNSW's theoretical guarantees (metric space, triangle inequality) hold without caveats — cosine distance doesn't strictly satisfy them. Section 4 doesn't change: the SQL surface still never requires normalized input vectors, normalization is an internal detail of the ANN Engine.

**Parameters** (all configurable via `settings`, with defaults from the HNSW literature, passed to the crate when creating each segment instance):

| Parameter | Role | Default |
|---|---|---|
| `M` | neighbours per node and layer | 16 |
| `ef_construction` | search width on insert | 200 |
| `ef_search` | search width on query | 100 |
| `top_k` | candidates returned by default | 16 |

**Inserting and searching within a segment**: internal mechanics of `hnsw_rs` (layer assignment, navigation, neighbour pruning on insert; navigation descent + `ef_search` at the base layer on search) — no longer Semanta's code. What remains Semanta's responsibility:
- Keeping one `Hnsw` instance in memory per segment (one per `appendable`/`sealed` row in `segments`).
- Fan-out + merge across segments (see "Graph segmentation" below, unchanged: the crate has no notion of segments, that layer is still built by Semanta on top).
- Persistence, which does change shape compared to a hand-rolled HNSW (see below).

**Persistence**: there's no longer a row per node (`graph_nodes_layers` is gone, see section 2). Instead:
- When a segment is **sealed**, its full index is dumped with `hnsw_rs`'s native dump (`hnswio`) into the `index_blob` BLOB in `segments`. This happens once per segment (the sealing event), an acceptable cost.
- The `appendable` segment is **not** dumped on every insert — that would get expensive as it grows toward `SEGMENT_MAX_SIZE`. It lives only in memory while the process is active.
- On **startup**: `sealed` segments are loaded directly from their `index_blob` (`hnswio` reload). The `appendable` segment is **rebuilt** by reinserting, in order, the chunks that belong to it (bounded to ≤10,000 nodes by design, so startup cost is predictable) — using the `m`/`ef_construction` stored in its own `segments` row, not the current values in `settings`, so the rebuild is deterministic even if `settings` changed midway through the segment's life.

### Graph segmentation — inspired by Qdrant

Instead of one giant HNSW graph over every chunk, the graph is split into **segments**: each one is an independent, self-contained `hnsw_rs` `Hnsw` instance (its own entry point, its own `top_layer`, its own `index_blob`). Two motivations: a smaller graph searches faster, and changing a parameter like `M` no longer affects the entire graph at once.

- Only one segment is `appendable` at a time — that's where all new insertions land.
- The rest of the segments are `sealed`: closed to new insertions, but queried the same as any other during search.
- **Sealing rule**: each segment has a fixed cap, `SEGMENT_MAX_SIZE = 10,000` nodes, **hardcoded** — it doesn't live in `settings`, because it's not a parameter the user should tune, it's part of the engine's internal design. When inserting a chunk, if the current `appendable` segment is already at the cap, it's marked `sealed` as-is (no need to rebuild it, the HNSW graph is already valid incrementally) and a new, empty `appendable` segment is opened, which receives the chunk that triggered the sealing.
- All of this happens synchronously, within the same call to `semanta_store_embedding` — no threads or background processes. The only observable consequence is that the insertion crossing the threshold is somewhat slower (it creates the new segment).
- **Changing `M` on the fly**: only affects the `appendable` segment opened from that point on — already-`sealed` segments are untouched and never mixed with the new value. `semanta_rebuild_graph()` still exists as a full manual reset (deletes all segments and rebuilds from scratch under the current `settings`), but it's no longer the only way to avoid a graph with mixed `M` values.

**Searching a segmented graph**: each segment (appendable + sealed) is traversed with its own layer descent and its own `ef_search`, candidates from all of them are pooled, and merged by distance to keep the global `top_k`. In this first version, segment traversal is sequential; parallelizing it (one thread per segment) is noted as a future improvement in section 11, not part of the base design.

This starting value (10k) is a reasonable initial choice, to be validated against real behaviour once there's data — not a figure derived from a benchmark.

**Insertion and candidates: two decoupled steps**. Segmenting the graph separates two things that used to be one:

1. **Structural connection** — the new node connects to its `M` best neighbours only within the current `appendable` segment, using `ef_construction`. This is what keeps the graph fast to build and isolated by `M` (previous section).
2. **Candidate discovery for the Graph Engine** (section 6) — this can't be limited to the local segment, or a new chunk would fail to find a semantically identical chunk as a neighbour just because it landed in an already-`sealed` segment, simply for having arrived before the `SEGMENT_MAX_SIZE` cutoff. That's why `semanta_store_embedding` internally reuses the same fan-out + merge across all segments that `semanta_search` uses, with the freshly inserted vector as the query, to get the candidates that are returned and that feed the relations.

The same reasoning applies to `semanta_get_candidates`: it also re-queries all segments (not just the node's local edges), because its purpose is the same — finding real semantic neighbours, not edges of an internal structure that is, deliberately, local to each segment.

It's the same principle Qdrant uses to solve global recall despite having segments: the HNSW structure stays local to each segment (never any cross-segment edges), and global recall is resolved entirely at query time, via fan-out + merge — never by enriching the structure between segments. Qdrant has no equivalent to step 2 (it has no Graph Engine), so here the same principle is applied to a second entry point that Qdrant doesn't need.

------------------------------------------------------------------------

## 6. Graph Engine — semantic relations

The heart of the project: using HNSW candidates to build relations between chunks, without Semanta ever having to "understand" the content.

**Division of responsibilities** (same pattern as the Embedding Engine):

1. Semanta exposes a chunk's candidates — the neighbours already found by HNSW when it was inserted, sorted by closeness, count = configurable `top_k`.
2. The user, from their own code, sends those candidates (as many as fit their current LLM — it can be a small one or a large one, which is why `top_k` is adjustable) and asks the LLM what relation exists, if any.
3. The user sends the result back to Semanta to persist it.

**Relation types**: free text, no closed catalog in Semanta. The vocabulary (continues, requires, contradicts, or whatever it may be) is defined by the user in their LLM's prompt, not by Semanta.

**No relation**: the evaluated pair is stored anyway, with a null `relation_type` (not a reserved word like `"none"`, so as not to collide with the user's free vocabulary) — this avoids asking the LLM about the same pair again in the future.

------------------------------------------------------------------------

## 7. Ranking Engine — expansion and reordering

The piece that was missing from the design: connects the Graph Engine to the search flow. Without it, `semanta_search` would be indistinguishable from any other vector store — the relations graph (section 6) would be built but never used to answer a query. Picks up steps 4-5 of "Search flow" in `Semanta_Overview.md` (expand with the Knowledge Graph, reorder).

**Flow**:

1. The ANN Engine returns its usual candidates (fan-out + merge across segments, section 5) — the entry point, called **anchor candidates** here.
2. For each anchor, it's expanded by querying `relations`: chunks connected in either direction (`from_chunk_id` or `to_chunk_id` equal to the anchor), excluding rows with `relation_type IS NULL` (pairs already evaluated with no relation — they add nothing here).
3. The expanded chunks are scored and merged with the anchors into a single list, sorted and truncated to the requested limit.

**Why expansion and not just reordering what the ANN already brought**: the graph's value is in finding chunks the ANN *doesn't* bring — a chunk can be far away in embedding space (different vocabulary) and still be the logical continuation of an anchor candidate, if an LLM already established that relation in the Graph Engine. If the Ranking Engine only reordered the ANN's `top_k`, that chunk would never show up.

**Scoring**: Semanta doesn't judge which relation type is "more relevant" — `relation_type` remains the user's free vocabulary, same principle as in section 6. An expanded chunk's score is plain arithmetic over data Semanta already has:

```
expanded_score = anchor_score × relation_confidence × (hop_decay ^ n_hops)
```

- `anchor_score`: similarity of the anchor candidate the expansion started from.
- `relation_confidence`: the `confidence` value stored in `relations` (supplied by the user when persisting the relation, section 6).
- `hop_decay`: configurable factor, penalizes expansions of more than one hop.

If a chunk is reachable via multiple routes (multiple anchors, or multiple hops), it keeps the highest `expanded_score` — it's never duplicated in the results nor are scores summed.

**Parameters** (via `settings`, unless overridden in the call):

| Parameter | Role | Default |
|---|---|---|
| `max_hops` | expansion depth from each anchor | 1 |
| `hop_decay` | multiplicative penalty per hop | 0.5 |
| `expand_relation_types` | subset of `relation_type` to follow when expanding; empty/null = all non-null types | all |

`max_hops = 1` by default is not an accident: multi-hop expansion grows combinatorially (candidates × edges per node, for each additional hop) and the cost stops being predictable. It's left configurable for whoever wants to take that on, but the default prioritizes keeping `semanta_search` cheap — the same criterion already applied to `SEGMENT_MAX_SIZE` in section 5.

**Result**: each row returned by `semanta_search` includes `chunk_id`, `score`, `origin` (`ann` | `graph`) and, if `origin = graph`, `relation_type` and `hop_count` — so whoever consumes the result knows *why* that chunk showed up, not just that it did.

------------------------------------------------------------------------

## 8. Configuration via SQL

`settings` table (key-value), editable with plain SQL (`UPDATE`/`INSERT`), no special functions to learn. Parameters covered: `chars_per_token`, `chunk_size`, `chunk_overlap`, `M`, `ef_construction`, `ef_search`, `top_k`, `max_hops`, `hop_decay`, `expand_relation_types`.

**Note**: the segment size cap (`SEGMENT_MAX_SIZE`, section 5) is deliberately kept out of `settings` — it's a core engine parameter, not a user-facing knob.

Changing a value in `settings` has no retroactive effect — it only affects future operations (new chunks, new searches). To retroactively apply a structural change to the graph, there's the explicit rebuild function mentioned in section 5.

------------------------------------------------------------------------

## 9. SQL surface sketch

Not yet finalized in detail, but follows from the decisions above:

- `semanta_add_document(name, content, metadata, tags, external_id?)` → version-aware upsert by `external_id` (see section 10): a new document or a different `hash` chunks it and returns the `document_id`; the same `hash` is a no-op. `external_id` is optional — if omitted, the document can never be recognised as "the same one" in a later call.
- `semanta_store_embedding(chunk_id, vector, model_name, dimension)` → stores the vector (upsert by `chunk_id`), structurally connects the node within the current `appendable` segment, and returns as candidates the result of a global search (fan-out + merge over all segments) with that same vector — not just the insertion's local edges.
- `semanta_get_candidates(chunk_id, top_k?)` → a chunk's current neighbours, recomputed with the same global search (fan-out + merge across segments), in case they need to be re-queried later.
- `semanta_store_relation(from_chunk_id, to_chunk_id, relation_type, confidence)` → persists a relation (or marks the pair as evaluated with no relation).
- `semanta_search(query_vector, top_k?, expand?)` → semantic search: traverses every segment and merges candidates by distance (ANN Engine), and by default (`expand = true`) expands and reorders those candidates via the relations graph (Ranking Engine, section 7). `expand = false` gives pure ANN, without touching `relations`.
- `semanta_rebuild_graph()` → manual rebuild of the graph under the current `settings`; deletes every existing segment and re-segments from scratch.
- `semanta_graph_stats()` → live vs. total nodes per segment (section 10), to decide when a `semanta_rebuild_graph()` is worth it.
- `semanta_delete_document(external_id)` → deletes the document and everything that depends on it (section 10). Errors if `external_id` doesn't exist.
- `settings` table → direct read/write via SQL.

------------------------------------------------------------------------

## 10. Updating and deleting documents

With the base design closed (sections 1-9), this was the first real gap discovered by exercising the full flow with real components (embeddings + a local LLM): there was no way to update or delete anything. It was resolved without touching `hnsw_rs` — no point-deletion, no partial rebuild of a segment — relying entirely on what SQLite already gives (upserts, transactions) plus an adjustment to the ANN Engine (recompute instead of blindly trusting the graph).

**Document identity — BYOE, same as embeddings and relation vocabulary**: Semanta doesn't infer whether two ingests are "the same document" either by `name` (free text, can repeat or change between uploads) or by `hash` (which changes exactly when the document is updated — it's the change signal, it can't also be the identity). Identity is declared by the caller, via `external_id` in `semanta_add_document`. It's optional: if not passed, `documents.external_id` is filled with the internal `id` itself (autoincrement, unique by construction) — the document can never be intentionally matched on a later ingest, which is correct: without a stable identity from the source, there's no version detection to offer, and it's not Semanta's problem if the caller doesn't have or doesn't pass a stable identifier.

**Semantics of `semanta_add_document(name, content, metadata, tags, external_id?)`**:
- `external_id` not seen before → new document, `version = 1`.
- `external_id` seen before, extracted content `hash` unchanged → pure no-op (no re-chunking, nothing).
- `external_id` seen before, `hash` different → `version += 1`, re-chunking.

**Re-chunking as an upsert, not a full replacement**: chunks are upserted by `UNIQUE(document_id, position)`, preserving `chunks.id` when the position already existed (so `embeddings`/`relations`, which reference that `id` via FK, are never invalidated) — only `text` and `version` change (bumped to the document's `version`). Two consequences:
- A chunk whose position gets reused with different text drags along relations judged on the old text — they're purged (`DELETE FROM relations WHERE from_chunk_id = ? OR to_chunk_id = ?`) so the next `semanta_get_candidates`/`semanta_store_embedding` for that chunk offers fresh candidates to the user's LLM again, without carrying over stale judgments (not even the cached `relation_type IS NULL` from section 6, which would otherwise never be asked again even though the content already changed).
- Positions that no longer exist in the new version (the document got shorter) are removed outright — `chunks`, `embeddings`, and associated `relations` — within the same call, without leaving them as residue to clean up later.

**`embeddings`**: `semanta_store_embedding` switches to `INSERT OR REPLACE` on `chunk_id` (already the PK, not autoincrement, so replacing on the same `chunk_id` is safe). The old vector, however, stays alive inside the HNSW segment it was inserted into — `hnsw_rs` doesn't support point-deletion (section 5) — until the ANN Engine handles it accordingly (next point).

**ANN Engine — HNSW to discover candidates, SQL for the final distance**: the direct consequence of not being able to delete a point is that a `chunk_id` can have more than one live node in the graph (the replacement doesn't delete the previous node, and if both land in the same `appendable` segment — very likely if the update happens before that segment seals — they don't even end up in different segments that would let you tell them apart). That's why `search()` (used by `semanta_search`, `semanta_get_candidates`, and `semanta_store_embedding`, all through the same ANN Engine entry point) changes role: the per-segment HNSW traversal is used only to *discover* which `chunk_id`s are nearby candidates (fast, approximate), never to report the final distance. That distance is always recomputed against each candidate `chunk_id`'s current `embeddings.vector` — which, along the way, automatically deduplicates any stale node (there are no longer two entries for the same `chunk_id` with different distances) and simply drops any `chunk_id` that no longer has a row in `embeddings` (deleted). The extra cost is bounded: one distance recomputation per unique candidate returned by HNSW, not a full-corpus scan.

**`semanta_graph_stats()`**: a stale node still occupies space in its segment's `index_blob`/memory until a full `semanta_rebuild_graph()` — that hasn't changed, it's still the only way to purge the index for real (section 5, no partial rebuild of a `sealed` segment). What's new is visibility: per segment, `total_nodes` (historical, `segments.node_count`) vs. `live_nodes` (`COUNT(*)` in `embeddings` with that `segment_id`) — no automatic heuristic for when to rebuild, same principle as the rest of the design (the user decides the policy, Semanta provides the data).

**`semanta_delete_document(external_id)`**: hard delete, no tombstone — the same criterion already used for chunk positions dropped by an update. Looks up by `external_id` (errors if it doesn't exist, to avoid silently failing on the wrong document); within a single transaction it deletes `relations`, `embeddings`, and `chunks` for every `chunk_id` of the document, and finally the `documents` row. Being a hard delete, re-ingesting the same `external_id` afterwards falls straight into the "new document, `version = 1`" branch with no special-casing needed.

------------------------------------------------------------------------

## 11. Out of scope — reviewed point by point (2026-08-10)

A full review of the original list, item by item, evaluating whether each one reinforces the differentiator (Graph Engine) or is a distraction toward "being just another vector DB/document library" (see `CLAUDE.md`). Two resulting categories: **discarded permanently** (not reopened without a new reason, at the same level as the decisions already made in section 1) and **parked** (the design may already be resolved, but implementation waits for real corpus/usage data to justify it).

### Discarded permanently

- **Extractors beyond Markdown (PDF, etc.)**: contradicts BYOE. Semanta shouldn't know whether the source was PDF, Word, or Excel — the caller normalizes to plain text before calling `semanta_add_document`, just as they already do for embeddings. The `DocumentExtractor` trait stays a single passthrough (`MarkdownExtractor`), with no further implementations planned.
- **Structure-aware chunking** (Markdown headings, paragraphs, etc.): a softer version was also evaluated (cutting on paragraph breaks instead of blindly by character) and discarded just the same — a variable-size chunk can exceed what the caller's embedding/LLM can process, and there Semanta would be the one breaking the size contract, not the caller. Fixed, configurable size (`chunk_size`/`chunk_overlap`/`chars_per_token` in `settings`) is the correct, final design, not a provisional one. Any smarter chunking (structural or semantic) is the caller's responsibility, outside Semanta.
- **Calibrating `chars_per_token` to the caller's LLM** (having Semanta "know" which model the host project uses in order to tokenize precisely): already explicitly rejected in section 3 — it would couple the Chunk Engine to a specific tokenizer, breaking the same neutrality BYOE already guarantees for embeddings. The caller can already calibrate `chars_per_token` in `settings` to whatever they measure empirically for their model, with no need for Semanta to "know" anything.
- **LLM-guided chunking** (using an LLM to judge whether two fragments should belong to the same chunk): if Semanta called the LLM itself, that plainly violates BYOE. If instead it proposed "boundary candidates" for the caller's LLM to decide (the same pattern as the Graph Engine), it would turn `semanta_add_document` — today a synchronous, deterministic call — into a multi-step flow as heavy as the embeddings/relations one, and would trigger LLM cost on the mandatory ingestion of the *entire* corpus, not just the optional relation enrichment (worsening CLAUDE.md's risk #1 instead of bounding it). If the caller wants this pattern, they solve it in their own project (outside Semanta), as they in fact already do.
- **Re-evaluating candidates after `semanta_rebuild_graph()`**: automating this adds no new capability — `semanta_get_candidates(chunk_id)` already lets the caller loop over the whole corpus themselves after a rebuild if they decide it's worth the cost. Offering the candidates themselves is cheap (one more HNSW query, no LLM), but automating it inside Semanta would hide, behind an operation that's cheap and predictable today (`rebuild_graph`), an LLM cost potentially proportional to the entire corpus — worse, not better, than leaving it explicit and opt-in as it is now.
- **Minimum similarity threshold**: configurable `top_k` plus candidates always returned sorted by `distance` already let the caller trim the list with whatever logic suits them. A threshold inside Semanta would be a redundant heuristic on top of data that's already exposed.
- **Automatically weighting `relation_type` by semantic importance**: contradicts the free-vocabulary principle (section 6) — Semanta doesn't understand content, so it can't judge which relation type "matters more" in general. The right mechanism already exists at the instance level: `confidence` in `semanta_store_relation`, decided by the caller's own LLM on each call, not by Semanta at the type level globally.

### Parked — pending real corpus/usage

- **Parallelizing search across segments** (one thread per segment): with a small corpus (e.g. ~30 documents) there's almost certainly a single `appendable` segment — there's nothing to parallelize yet, and measuring against an inflated synthetic corpus wouldn't reflect real behaviour. Revisit once there are multiple `sealed` segments in production and real latency data (which will likely show the real bottleneck is the caller's own LLM calls during ingestion, not this fan-out — the `semanta_search` flow, section 7, never calls an LLM).
- **Optimizing/rebuilding a single `sealed` segment**: unlike the items above, the design is already resolved. No new "version" or "stale node" marker is needed: `embeddings.segment_id` is already the sole source of truth for what's live in each segment — a phantom node in the `index_blob` simply has no matching row in `embeddings` for that `segment_id`, so it's never reinserted on rebuild. Proposed extension: `semanta_rebuild_graph(segment_id?)` — no argument keeps today's behaviour (rebuild everything); with `segment_id`, it rebuilds only that segment in place (same `id`, same `m`/`ef_construction`), reinserting only the `chunk_id`s whose `embeddings.segment_id` matches, without going through the `insert()` path that manages the global appendable segment (to avoid spilling chunks into another segment). The stale node causes **no noise in search results today** — that's already neutralized by `ann::search()` recomputing distance against the live `embeddings.vector` (section 10) — the only cost is `index_blob` space and some extra traversal. Implement once the real corpus accumulates more than one `sealed` segment: with only one, this scoped operation and a full rebuild are identical, so there's nothing to gain yet.
- **Tuning `SEGMENT_MAX_SIZE` with real data**: still no benchmark-derived figure; revisit alongside the two items above once there's real usage.
- **`max_hops > 1` as the default**: not a pending design or code item at all — the loop in `expand_and_rank` (Ranking Engine, section 7) already supports any depth with no changes; the caller can already raise `max_hops` in `settings` today. The only thing pending is a product decision: whether `1` is still the right default or whether it's worth raising. It's left at `1` (the most conservative default, least noise) until there's real relation data to evaluate whether extra hops add signal or just dilute the ranking.
