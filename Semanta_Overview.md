# Semanta

## What is Semanta?

**Semanta** is an embedded semantic engine for SQLite that lets you
store documents, index them, relate them, and retrieve them using
natural language.

Its goal isn't to be just another vector database, but to offer a
complete pipeline for RAG applications without depending on multiple
external tools.

------------------------------------------------------------------------

# Goals

Semanta aims to simplify building applications that work with
technical documentation, manuals, or knowledge bases.

Instead of building a pipeline composed of numerous libraries:

Document → Chunking → Embeddings → Vector store → Relations →
Search → LLM

the developer only needs to interact with Semanta.

------------------------------------------------------------------------

# Ingestion flow

When a document is added, Semanta automatically runs the following
process:

1.  Extract the document's text.
2.  Split the content into chunks.
3.  Generate an embedding for each chunk.
4.  Find the closest semantic neighbours via an ANN index.
5.  Use an LLM to validate only those candidates and detect semantic
    relations.
6.  Store those relations in a knowledge graph.
7.  Update the vector index.
8.  Persist all of the information in SQLite.

This approach avoids comparing every chunk against every other one
and allows scaling efficiently.

------------------------------------------------------------------------

# Stored information

Semanta keeps several kinds of data:

## Documents

-   Name
-   Hash
-   Date
-   Metadata
-   Tags

## Chunks

-   Text
-   Position within the document
-   Parent document

## Embeddings

One embedding per chunk.

## ANN index

Initially based on HNSW for fast search.

## Knowledge Graph

Semantic relations between chunks.

Examples:

-   continues
-   requires
-   extends
-   is an example of
-   contradicts

Each relation can store a confidence level.

## Payload

Indexable metadata for filtering during search.

## Configuration

Information about the models in use:

-   Embedding model
-   Dimension
-   Generation date
-   Parameters

------------------------------------------------------------------------

# Search flow

When the user runs a query:

1.  The query's embedding is generated.
2.  The ANN index is queried.
3.  The best candidates are retrieved.
4.  They're expanded using the Knowledge Graph.
5.  The results are reordered.
6.  The optimal context is returned to the LLM.

This way, the vector index finds the entry points while the graph
contributes additional context.

------------------------------------------------------------------------

# Architecture

Semanta is meant to work as a semantic layer on top of SQLite.

The core will be split into independent modules:

-   Document Engine
-   Chunk Engine
-   Embedding Engine
-   ANN Engine
-   Graph Engine
-   Ranking Engine
-   Storage Engine

Each module can evolve without affecting the rest of the system.

------------------------------------------------------------------------

# Philosophy

Semanta isn't tied to any one implementation.

Providers are meant to be interchangeable.

## Embeddings

-   BGE
-   E5
-   Nomic
-   OpenAI
-   Gemma
-   Others

## ANN

-   HNSW
-   IVF
-   Brute force
-   DiskANN
-   Future implementations

## LLM

-   llama.cpp
-   Ollama
-   OpenAI
-   Gemini
-   Any compatible provider

------------------------------------------------------------------------

# Desired API

The developer experience should be extremely simple.

``` csharp
var semanta = new Semanta(database);

semanta.AddDocument("manual.pdf");

var results = semanta.Search(
    "How do I change the oil?"
);
```

All of the pipeline's complexity is meant to stay encapsulated inside
Semanta.

------------------------------------------------------------------------

# Vision

Semanta doesn't aim to be just a vector database.

Its goal is to become an embedded semantic engine that combines:

-   Document storage.
-   Chunking.
-   Embeddings.
-   ANN indexes.
-   Knowledge graph.
-   Hybrid retrieval.
-   Automatic context expansion.

All of it integrated on top of SQLite through a modular, extensible
architecture.
