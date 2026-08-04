# Semanta

Motor semántico embebido para SQLite: extensión nativa en Rust, cargable vía `load_extension(...)` desde cualquier cliente que hable SQLite. La interfaz real es SQL — no una librería atada a un lenguaje.

## Documentos fuente (leer antes de tocar diseño)

- `Semanta_Overview.md` — visión del producto, qué problema resuelve.
- `Semanta_Design.md` — decisiones de arquitectura ya cerradas. Es la fuente de verdad técnica; si una decisión de código contradice este documento, el documento gana o se actualiza explícitamente, no se ignora en silencio.

Estado actual: fase de diseño, sin código todavía.

## El diferenciador — no perderlo de vista

Semanta no compite en "otro índice vectorial en SQLite" (`sqlite-vec`, `sqlite-vss`, pgvector, etc. ya cubren eso, y bien). Lo que nadie más ofrece en una sola pieza es el **Graph Engine** (sección 6 del diseño): el HNSW no es el producto, es el mecanismo barato para proponerle candidatos a un LLM externo, que decide si existe una relación semántica real entre dos chunks — vocabulario libre, sin catálogo cerrado — y Semanta la persiste. Construye un grafo de conocimiento sin que Semanta "entienda" contenido.

Frente a alternativas:
- Vector stores puros: dan vecinos cercanos, no relaciones persistidas — eso se construye ad hoc por fuera.
- Graph DBs (Neo4j, etc.): dan el grafo, pero sin integración nativa con el paso "encuéntrame candidatos semánticos para evaluar".

Cualquier feature nueva debería preguntarse si refuerza este ciclo (candidatos → LLM del usuario → relación persistida) o si es una distracción hacia "ser una vector DB más".

## Decisiones ya tomadas, no reabrir sin razón nueva

- **BYOE**: Semanta no genera embeddings ni llama a un LLM. Eso lo aporta quien usa Semanta desde su propio código.
- **HNSW vía `hnswlib-rs`**, no implementación propia — ya se evaluó y se descartó escribir HNSW a mano (riesgo de bugs sutiles de recall sin infraestructura de benchmarking, y no aporta diferenciación). Ver sección 5 del diseño.
- **Métrica**: normalización interna + `DistL2` (equivalente en orden a coseno para vectores unitarios, pero con las garantías teóricas de HNSW intactas). El usuario nunca tiene que normalizar nada — eso es un detalle interno.
- **Grafo segmentado** estilo Qdrant: HNSW nunca tiene aristas cruzadas entre segmentos; el recall global se resuelve en el momento de consultar (fan-out + merge), nunca enriqueciendo la estructura entre segmentos. Aplica tanto a `semanta_search` como a `semanta_get_candidates`.
- `SEGMENT_MAX_SIZE` (10k) y los parámetros HNSW por segmento (`m`, `ef_construction`) están fijados por fila en `segments`, no se leen en vivo de `settings` — necesario para que la reconstrucción del segmento `appendable` al arrancar sea determinista.

## Riesgos a vigilar cuando haya uso real (no bloquean el diseño actual, pero condicionan si el diferenciador aguanta)

1. **Coste de LLM del Graph Engine**: evaluar `top_k` candidatos por cada chunk nuevo escala como llamadas a LLM proporcionales al corpus. Si esto se dispara en coste/latencia, el "grafo inteligente" deja de ser gratis de mantener y la gente lo desactiva. El `NULL` de "evaluado, sin relación" ya mitiga la repregunta, pero no la primera pasada sobre un corpus grande.
2. **`semanta_get_candidates` tiene que seguir siendo barato de recalcular**: si el grafo es el gancho de producto, se va a re-consultar mucho. El fan-out + merge entre segmentos no puede degradarse silenciosamente a medida que crecen los segmentos `sealed`.

## Fuera de alcance por ahora

Ver sección 9 de `Semanta_Design.md` (extractores más allá de Markdown, chunking consciente de estructura, re-evaluación de candidatos tras rebuild, umbral mínimo de similitud, paralelización entre segmentos, optimización de segmentos `sealed`, ajuste de `SEGMENT_MAX_SIZE` con datos reales). No implementar nada de esta lista sin decisión explícita primero.
