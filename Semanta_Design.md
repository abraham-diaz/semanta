# Semanta — Diseño técnico

> Este documento aterriza las decisiones de arquitectura tomadas a partir de `Semanta_Overview.md`. Donde el overview describe la visión, este documento describe cómo se va a construir.

------------------------------------------------------------------------

## 1. Qué es, concretamente

Semanta es una **extensión nativa de SQLite**, escrita en **Rust**, cargable con `load_extension(...)` desde cualquier cliente que hable SQLite (Python, CLI, Node, C#...). No es una librería atada a un lenguaje concreto — la interfaz de verdad es SQL: tablas y funciones.

Su responsabilidad es acotada a propósito:

- **Sí hace**: guardar documentos/chunks/embeddings, indexar embeddings con HNSW (vía la crate `hnswlib-rs`), encontrar candidatos semánticamente cercanos, expandirlos usando el grafo de relaciones al buscar, y guardar relaciones entre chunks.
- **No hace**: generar embeddings, ni llamar a un LLM. Esas dos piezas de "inteligencia" las aporta quien use Semanta, desde su propio código (por ejemplo, Python con `sentence-transformers` + un LLM propio).

Esta frontera es deliberada: evita atar Semanta a un modelo de embeddings, a un proveedor de LLM, o a que el usuario tenga que confiar en que Semanta implementa bien esa inferencia dentro de una extensión nativa en un lenguaje (Rust) nuevo para el autor.

------------------------------------------------------------------------

## 2. Storage Engine (esquema SQLite)

SQLite actúa como almacén puro. Toda la lógica de búsqueda e indexado vive en memoria, en el proceso que carga la extensión, y se sincroniza con estas tablas.

| Tabla | Rol |
|---|---|
| `documents` | id, name, hash, created_at, metadata, tags |
| `chunks` | id, document_id (FK), text, position |
| `embedding_models` | id, model_name, dimension, generated_at, parameters — registra qué modelo generó cada embedding (dato que aporta el usuario, no lo calcula Semanta) |
| `embeddings` | chunk_id (FK 1:1), embedding_model_id (FK), vector (BLOB) |
| `segments` | id, status (`appendable`/`sealed`), m, ef_construction, entry_point_node_id, top_layer, node_count, index_blob (BLOB, dump nativo de `hnswlib-rs`; `NULL` mientras el segmento está `appendable`), created_at, sealed_at — cada fila es un HNSW independiente y autocontenido (ver sección 5). `m`/`ef_construction` quedan fijados por fila al crearse: no se leen en vivo de `settings` |
| `relations` | from_chunk_id, to_chunk_id, relation_type (texto libre, nullable), confidence — el grafo de conocimiento semántico. Índice en ambas columnas de chunk_id: se consulta en las dos direcciones en cada búsqueda (Ranking Engine, sección 7), no solo al insertar |
| `settings` | key, value — parámetros configurables del motor |

**Por qué un BLOB por segmento y no una fila por nodo/arista**: con HNSW implementado a mano, una fila por nodo-capa encajaba con el patrón de escritura (reescritura de listas de vecinos al podar). Al delegar el algoritmo en `hnswlib-rs`, el índice ya vive serializado en el formato propio de la crate (`hnswio`); replicarlo como filas SQL sería reinventar una serialización que la crate ya resuelve. El BLOB es opaco a Semanta salvo por lo que la propia crate expone al recargarlo.

**Por qué el vector es un BLOB simple**: se rechazó depender de una extensión externa (`sqlite-vec`) por ser una pieza fuera de nuestro control. Un embedding es solo un array de floats serializado a bytes — sin dependencias, sin librerías externas verdes que puedan dar problemas.

------------------------------------------------------------------------

## 3. Document/Chunk Engine

**Extracción**: abstracción tipo `DocumentExtractor` (trait en Rust) con implementaciones por formato. Primera implementación: **Markdown**. El resto de formatos (PDF, etc.) se añaden después sobre la misma interfaz, sin tocar el resto del sistema — crecimiento incremental, no todo de golpe.

**Chunking**: tamaño fijo por tokens estimados (no consciente de la estructura del Markdown, al menos en esta primera versión), con objetivo de **~1000 tokens** por chunk.

- Conteo de tokens: aproximado por caracteres (sin tokenizador real, para no acoplar el Chunk Engine a un modelo concreto). Ratio configurable, por defecto **~3.5 caracteres/token**, calibrado como promedio entre español y francés (los idiomas principales del proyecto), sin cerrar la puerta a otros idiomas.
- Solapamiento entre chunks consecutivos: configurable, por defecto **~10-15%** del tamaño del chunk, para no perder contexto en los bordes de corte.

------------------------------------------------------------------------

## 4. Embedding Engine — BYOE (Bring Your Own Embeddings)

Semanta **no genera embeddings**. El usuario ya tiene resuelto ese paso en su propio stack (en este proyecto: `sentence-transformers` desde Python).

- **Al ingerir**: el usuario calcula el vector de cada chunk y se lo pasa a Semanta para guardar.
- **Al buscar**: el usuario calcula el vector de la consulta y se lo pasa a Semanta ya vectorizado — Semanta nunca tokeniza ni infiere nada de texto libre en este punto.
- Semanta valida que la dimensión del vector sea consistente con el modelo registrado en `embedding_models`, para no mezclar vectores incompatibles en el mismo índice.

------------------------------------------------------------------------

## 5. ANN Engine — HNSW sobre `hnswlib-rs`

Se descarta implementar HNSW desde cero: es un algoritmo (Malkov-Yashunin) ya resuelto por implementaciones maduras, y escribirlo a mano no aporta diferenciación a Semanta — sí aporta riesgo real de bugs sutiles de recall, difíciles de detectar sin infraestructura de benchmarking dedicada. Se usa la crate **`hnswlib-rs`**: Rust puro (sin dependencias nativas fuera de Rust, relevante para compilar/distribuir la extensión), inserción incremental, dump/reload de índice, mantenida activamente.

**Métrica de distancia**: los vectores se normalizan internamente a norma unitaria (coste O(d), transparente para quien llama) y se indexan con `DistL2` de la crate. Para vectores unitarios, `‖a-b‖² = 2 - 2·cos(a,b)`, así que el orden de vecinos resultante es idéntico al de similitud coseno. Se prefiere esto sobre usar `DistCosine` directamente porque L2 es la métrica bajo la que las garantías teóricas de HNSW (espacio métrico, desigualdad triangular) se sostienen sin matices — la distancia coseno no la cumple estrictamente. La sección 4 no cambia: la superficie SQL sigue sin exigir vectores normalizados de entrada, la normalización es un detalle interno del ANN Engine.

**Parámetros** (todos configurables vía `settings`, con valores por defecto de la literatura HNSW, pasados a la crate al crear cada instancia de segmento):

| Parámetro | Rol | Default |
|---|---|---|
| `M` | vecinos por nodo y capa | 16 |
| `ef_construction` | amplitud de búsqueda al insertar | 200 |
| `ef_search` | amplitud de búsqueda al consultar | 100 |
| `top_k` | candidatos devueltos por defecto | 16 |

**Insertar y buscar dentro de un segmento**: mecánica interna de `hnswlib-rs` (asignación de piso, navegación, poda de vecinos al insertar; descenso de navegación + `ef_search` en la capa base al buscar) — ya no es código de Semanta. Lo que sigue siendo responsabilidad de Semanta:
- Mantener una instancia `Hnsw` en memoria por segmento (una por fila `appendable`/`sealed` de `segments`).
- El fan-out + merge entre segmentos (sección "Segmentación del grafo", sin cambios: la crate no tiene noción de segmentos, esa capa la sigue construyendo Semanta encima).
- La persistencia, que sí cambia de forma respecto a un HNSW escrito a mano (ver abajo).

**Persistencia**: ya no hay una fila por nodo (`graph_nodes_layers` desaparece, ver sección 2). En su lugar:
- Al **sellar** un segmento, se vuelca su índice completo con el dump nativo de `hnswlib-rs` (`hnswio`) al BLOB `index_blob` en `segments`. Ocurre una sola vez por segmento (evento de sellado), coste asumible.
- El segmento `appendable` **no se vuelca en cada inserción** — sería caro a medida que crece hacia `SEGMENT_MAX_SIZE`. Vive solo en memoria mientras el proceso está activo.
- Al **arrancar**: los segmentos `sealed` se cargan directamente desde su `index_blob` (`hnswio` reload). El segmento `appendable` se **reconstruye** reinsertando, en orden, los chunks que le pertenecen (acotado a ≤10.000 nodos por diseño, así que el coste de arranque es predecible) — usando el `m`/`ef_construction` guardados en su propia fila de `segments`, no los valores actuales de `settings`, para que la reconstrucción sea determinista aunque `settings` haya cambiado a mitad de vida del segmento.

### Segmentación del grafo — inspirado en Qdrant

En lugar de un único HNSW gigante sobre todos los chunks, el grafo se divide en **segmentos**: cada uno es una instancia `Hnsw` de `hnswlib-rs` independiente y autocontenida (su propio entry point, su propio `top_layer`, su propio `index_blob`). Doble motivación: un grafo más pequeño se busca más rápido, y cambiar un parámetro como `M` deja de afectar a todo el grafo de golpe.

- Solo hay un segmento en estado `appendable` a la vez — ahí caen todas las inserciones nuevas.
- El resto de segmentos están `sealed`: cerrados a nueva inserción, pero se consultan igual que cualquier otro en búsqueda.
- **Regla de sellado**: cada segmento tiene un tope fijo, `SEGMENT_MAX_SIZE = 10_000` nodos, **hardcodeado** — no vive en `settings`, porque no es un parámetro que el usuario deba ajustar, es parte del diseño interno del motor. Al insertar un chunk, si el segmento `appendable` actual ya está en el tope, se marca `sealed` tal cual está (no hace falta reconstruirlo, el HNSW ya es válido incrementalmente) y se abre un segmento `appendable` nuevo y vacío, donde entra el chunk que disparó el sellado.
- Todo esto ocurre de forma síncrona, dentro de la misma llamada a `semanta_store_embedding` — sin hilos ni procesos en background. La única consecuencia observable es que la inserción que cruza el umbral es algo más lenta (crea el segmento nuevo).
- **Cambiar `M` en caliente**: solo afecta al segmento `appendable` que se abra a partir de ese momento — los segmentos ya `sealed` no se tocan ni se mezclan con el nuevo valor. `semanta_rebuild_graph()` sigue existiendo como reset manual total (borra todos los segmentos y reconstruye desde cero bajo los `settings` actuales), pero deja de ser la única forma de evitar un grafo con `M` mixto.

**Buscar en un grafo segmentado**: se recorre cada segmento (appendable + sealed) con su propio descenso de capas y su propio `ef_search`, se juntan los candidatos de todos, y se hace merge por distancia para quedarse con el `top_k` global. En esta primera versión el recorrido de segmentos es secuencial; paralelizarlo (un hilo por segmento) queda anotado como mejora futura en la sección 9, no como parte del diseño base.

Este valor de arranque (10k) es una elección razonable de partida, a validar con el comportamiento real una vez haya datos — no una cifra derivada de un benchmark.

**Inserción y candidatos: dos pasos desacoplados**. Segmentar el grafo separa dos cosas que antes eran una sola:

1. **Conexión estructural** — el nodo nuevo se conecta a sus `M` mejores vecinos únicamente dentro del segmento `appendable` actual, con `ef_construction`. Esto es lo que mantiene el grafo rápido de construir y aislado por `M` (sección anterior).
2. **Descubrimiento de candidatos para el Graph Engine** (sección 6) — no puede limitarse al segmento local, o un chunk nuevo dejaría de encontrar como vecino a un chunk semánticamente idéntico que quedó en un segmento ya `sealed`, simplemente por haber llegado antes del corte de `SEGMENT_MAX_SIZE`. Por eso `semanta_store_embedding` reutiliza internamente el mismo fan-out + merge entre todos los segmentos que usa `semanta_search`, con el vector recién insertado como query, para obtener los candidatos que se devuelven y que alimentan las relaciones.

Este mismo razonamiento aplica a `semanta_get_candidates`: también reconsulta todos los segmentos (no solo las aristas locales del nodo), porque su propósito es el mismo — encontrar vecinos semánticos reales, no aristas de una estructura interna que es, deliberadamente, local por segmento.

Es el mismo principio que usa Qdrant para resolver el recall global pese a tener segmentos: la estructura del HNSW se queda local a cada segmento (nunca hay aristas cruzadas), y el recall global se resuelve enteramente en el momento de consultar, vía fan-out + merge — nunca enriqueciendo la estructura entre segmentos. Qdrant no tiene un equivalente al paso 2 (no tiene Graph Engine), así que aquí se aplica el mismo principio a un segundo punto de entrada que Qdrant no necesita.

------------------------------------------------------------------------

## 6. Graph Engine — relaciones semánticas

El corazón del proyecto: usar los candidatos del HNSW para construir relaciones entre chunks, sin que Semanta tenga que "entender" el contenido.

**División de responsabilidades** (mismo patrón que en el Embedding Engine):

1. Semanta expone los candidatos de un chunk — los vecinos ya encontrados por el HNSW al insertarlo, ordenados por cercanía, cantidad = `top_k` configurable.
2. El usuario, desde su propio código, manda esos candidatos (tantos como le quepan a su LLM del momento — puede ser uno pequeño o uno grande, por eso `top_k` es ajustable) y le pregunta al LLM qué relación existe, si existe alguna.
3. El usuario devuelve el resultado a Semanta para persistirlo.

**Tipos de relación**: texto libre, sin catálogo cerrado en Semanta. El vocabulario (continúa, requiere, contradice, o lo que sea) lo define el usuario en el prompt de su LLM, no Semanta.

**Sin relación**: se guarda igualmente el par evaluado, con `relation_type` nulo (no una palabra reservada como `"ninguna"`, para no chocar con el vocabulario libre del usuario) — así se evita volver a preguntarle al LLM por el mismo par en el futuro.

------------------------------------------------------------------------

## 7. Ranking Engine — expansión y reordenado

Pieza que faltaba en el diseño: conecta el Graph Engine con el flujo de búsqueda. Sin esto, `semanta_search` sería indistinguible de un vector store cualquiera — el grafo de relaciones (sección 6) se construiría pero nunca se usaría para responder una consulta. Retoma los pasos 4-5 de "Flujo de búsqueda" en `Semanta_Overview.md` (expandir con el Knowledge Graph, reordenar).

**Flujo**:

1. El ANN Engine devuelve sus candidatos habituales (fan-out + merge entre segmentos, sección 5) — el punto de entrada, aquí llamados **candidatos ancla**.
2. Por cada ancla, se expande consultando `relations`: chunks conectados en cualquier dirección (`from_chunk_id` o `to_chunk_id` igual al ancla), excluyendo filas con `relation_type IS NULL` (pares ya evaluados sin relación — no aportan nada aquí).
3. Los chunks expandidos se puntúan y se mezclan con los anclas en una única lista, ordenada y truncada al límite pedido.

**Por qué expansión y no solo reordenar lo que ya trajo el ANN**: el valor del grafo está en encontrar chunks que el ANN *no* trae — un chunk puede quedar lejos en el espacio de embeddings (vocabulario distinto) y aun así ser la continuación lógica de un candidato ancla, si un LLM ya estableció esa relación en el Graph Engine. Si el Ranking Engine solo reordenara el `top_k` del ANN, ese chunk nunca aparecería.

**Scoring**: Semanta no juzga qué tipo de relación es "más relevante" — `relation_type` sigue siendo vocabulario libre del usuario, mismo principio que en la sección 6. La puntuación de un chunk expandido es aritmética pura sobre datos que Semanta ya tiene:

```
score_expandido = score_ancla × confidence_relación × (hop_decay ^ n_hops)
```

- `score_ancla`: similitud del candidato ancla del que parte la expansión.
- `confidence_relación`: el valor de `confidence` guardado en `relations` (aportado por el usuario al persistir la relación, sección 6).
- `hop_decay`: factor configurable, penaliza expansiones de más de un salto.

Si un chunk es alcanzable por varias rutas (varios anclas, o varios saltos), se queda con el `score_expandido` más alto — no se duplica en el resultado ni se suman scores.

**Parámetros** (vía `settings`, salvo que se indique lo contrario en la llamada):

| Parámetro | Rol | Default |
|---|---|---|
| `max_hops` | profundidad de expansión desde cada ancla | 1 |
| `hop_decay` | penalización multiplicativa por salto | 0.5 |
| `expand_relation_types` | subconjunto de `relation_type` a seguir al expandir; vacío/null = todos los no-nulos | todos |

`max_hops = 1` por defecto no es casualidad: la expansión multi-salto crece combinatoriamente (candidatos × aristas por nodo, por cada salto adicional) y el coste deja de ser predecible. Se deja configurable para quien quiera asumirlo, pero el default prioriza que `semanta_search` siga siendo barato — mismo criterio que ya se aplicó a `SEGMENT_MAX_SIZE` en la sección 5.

**Resultado**: cada fila devuelta por `semanta_search` incluye `chunk_id`, `score`, `origin` (`ann` | `graph`) y, si `origin = graph`, `relation_type` y `hop_count` — para que quien consume el resultado sepa *por qué* apareció ese chunk, no solo que apareció.

------------------------------------------------------------------------

## 8. Configuración vía SQL

Tabla `settings` (clave-valor), editable con SQL normal (`UPDATE`/`INSERT`), sin funciones especiales que aprender. Parámetros cubiertos: `chars_per_token`, `chunk_size`, `chunk_overlap`, `M`, `ef_construction`, `ef_search`, `top_k`, `max_hops`, `hop_decay`, `expand_relation_types`.

**Nota**: el tope de tamaño de segmento (`SEGMENT_MAX_SIZE`, sección 5) queda deliberadamente fuera de `settings` — es un parámetro del core del motor, no una perilla de usuario.

Cambiar un valor en `settings` no tiene efecto retroactivo — solo afecta a operaciones futuras (nuevos chunks, nuevas búsquedas). Para aplicar retroactivamente un cambio estructural del grafo, existe la función explícita de reconstrucción mencionada en la sección 5.

------------------------------------------------------------------------

## 9. Bosquejo de superficie SQL

Aún no cerrado en detalle, pero se desprende de las decisiones anteriores:

- `semanta_add_document(name, content, metadata, tags)` → extrae y trocea, devuelve los chunks creados (sin embeddings todavía).
- `semanta_store_embedding(chunk_id, vector, model_name, dimension)` → guarda el vector, conecta el nodo estructuralmente dentro del segmento `appendable` actual, y devuelve como candidatos el resultado de una búsqueda global (fan-out + merge sobre todos los segmentos) con ese mismo vector — no solo las aristas locales de la inserción.
- `semanta_get_candidates(chunk_id, top_k?)` → vecinos actuales de un chunk, recalculados con la misma búsqueda global (fan-out + merge entre segmentos), por si se quieren re-consultar más adelante.
- `semanta_store_relation(from_chunk_id, to_chunk_id, relation_type, confidence)` → persiste (o marca como evaluado sin relación).
- `semanta_search(query_vector, top_k?, expand?)` → búsqueda semántica: recorre todos los segmentos y hace merge de candidatos por distancia (ANN Engine), y por defecto (`expand = true`) expande y reordena esos candidatos vía el grafo de relaciones (Ranking Engine, sección 7). `expand = false` da ANN puro, sin tocar `relations`.
- `semanta_rebuild_graph()` → reconstrucción manual del grafo bajo los `settings` actuales; borra todos los segmentos existentes y re-segmenta desde cero.
- Tabla `settings` → lectura/escritura directa por SQL.

------------------------------------------------------------------------

## 10. Fuera de alcance (anotado para el futuro)

- Extractores para formatos más allá de Markdown (PDF, etc.).
- Chunking consciente de la estructura del Markdown (usar encabezados como fronteras) — quedó descartado en favor de tamaño fijo por simplicidad, posible mejora futura.
- Re-evaluación de candidatos tras una reconstrucción del grafo, para pescar relaciones que antes no se detectaron por tener un `M` más pequeño. Implica coste de llamadas a LLM, no se diseña todavía.
- Umbral mínimo de similitud para filtrar candidatos antes de ofrecerlos al LLM — se descartó por ahora: `top_k` configurable + candidatos siempre ordenados por cercanía ya permite al usuario recortar la lista según lo que le quepa a su LLM.
- Paralelización de la búsqueda entre segmentos (un hilo por segmento) — el diseño base la recorre de forma secuencial; queda como optimización de rendimiento futura.
- Optimización/reconstrucción de un segmento `sealed` individual (mejorar su calidad tras sellarlo) — en esta versión sellar solo detiene nuevas inserciones, no reconstruye nada.
- Ajustar `SEGMENT_MAX_SIZE` en base a comportamiento real observado — el valor de 10k es un punto de partida razonable, no una cifra derivada de benchmarks.
- Ponderar `relation_type` por importancia semántica de forma automática (ej. aprender qué tipos de relación conviene priorizar al expandir) — se deja enteramente en manos del usuario vía `expand_relation_types`, sin heurística automática en Semanta, mismo principio que el vocabulario libre de la sección 6.
- Expansión multi-salto (`max_hops > 1`) como default — queda soportada por diseño pero sin activarla por defecto, ver sección 7.
