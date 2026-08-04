# Semanta

## ¿Qué es Semanta?

**Semanta** es un motor semántico embebido para SQLite que permite
almacenar documentos, indexarlos, relacionarlos y recuperarlos mediante
lenguaje natural.

Su objetivo no es ser simplemente una base de datos vectorial, sino
ofrecer un pipeline completo para aplicaciones RAG sin depender de
múltiples herramientas externas.

------------------------------------------------------------------------

# Objetivos

Semanta busca simplificar el desarrollo de aplicaciones que trabajan con
documentación técnica, manuales o bases de conocimiento.

En lugar de construir un pipeline compuesto por numerosas librerías:

Documento → Chunking → Embeddings → Base vectorial → Relaciones →
Búsqueda → LLM

el desarrollador únicamente necesita interactuar con Semanta.

------------------------------------------------------------------------

# Flujo de ingestión

Cuando se incorpora un documento, Semanta ejecuta automáticamente el
siguiente proceso:

1.  Extraer el texto del documento.
2.  Dividir el contenido en chunks.
3.  Generar un embedding para cada chunk.
4.  Buscar los vecinos semánticos más próximos mediante un índice ANN.
5.  Utilizar un LLM para validar únicamente esos candidatos y detectar
    relaciones semánticas.
6.  Guardar dichas relaciones en un grafo de conocimiento.
7.  Actualizar el índice vectorial.
8.  Persistir toda la información en SQLite.

Este enfoque evita comparar todos los chunks entre sí y permite escalar
de forma eficiente.

------------------------------------------------------------------------

# Información almacenada

Semanta mantiene diferentes tipos de datos:

## Documentos

-   Nombre
-   Hash
-   Fecha
-   Metadatos
-   Etiquetas

## Chunks

-   Texto
-   Posición dentro del documento
-   Documento padre

## Embeddings

Un embedding por cada chunk.

## Índice ANN

Inicialmente basado en HNSW para realizar búsquedas rápidas.

## Knowledge Graph

Relaciones semánticas entre chunks.

Ejemplos:

-   continúa
-   requiere
-   amplía
-   es ejemplo de
-   contradice

Cada relación puede almacenar un nivel de confianza.

## Payload

Metadatos indexables para realizar filtros durante las búsquedas.

## Configuración

Información sobre los modelos utilizados:

-   Modelo de embeddings
-   Dimensión
-   Fecha de generación
-   Parámetros

------------------------------------------------------------------------

# Flujo de búsqueda

Cuando el usuario realiza una consulta:

1.  Se genera el embedding de la consulta.
2.  Se consulta el índice ANN.
3.  Se recuperan los mejores candidatos.
4.  Se expanden utilizando el Knowledge Graph.
5.  Se reordenan los resultados.
6.  Se devuelve el contexto óptimo al LLM.

De esta forma, el índice vectorial encuentra los puntos de entrada
mientras que el grafo aporta contexto adicional.

------------------------------------------------------------------------

# Arquitectura

Semanta pretende funcionar como una capa semántica sobre SQLite.

El núcleo estará dividido en módulos independientes:

-   Document Engine
-   Chunk Engine
-   Embedding Engine
-   ANN Engine
-   Graph Engine
-   Ranking Engine
-   Storage Engine

Cada módulo podrá evolucionar sin afectar al resto del sistema.

------------------------------------------------------------------------

# Filosofía

Semanta no está ligado a una implementación concreta.

Los proveedores serán intercambiables.

## Embeddings

-   BGE
-   E5
-   Nomic
-   OpenAI
-   Gemma
-   Otros

## ANN

-   HNSW
-   IVF
-   Brute Force
-   DiskANN
-   Futuras implementaciones

## LLM

-   llama.cpp
-   Ollama
-   OpenAI
-   Gemini
-   Cualquier proveedor compatible

------------------------------------------------------------------------

# API deseada

La experiencia del desarrollador debe ser extremadamente sencilla.

``` csharp
var semanta = new Semanta(database);

semanta.AddDocument("manual.pdf");

var results = semanta.Search(
    "¿Cómo cambiar el aceite?"
);
```

Toda la complejidad del pipeline queda encapsulada dentro de Semanta.

------------------------------------------------------------------------

# Visión

Semanta no pretende ser únicamente una base de datos vectorial.

Su objetivo es convertirse en un motor semántico embebido que combine:

-   Almacenamiento documental.
-   Chunking.
-   Embeddings.
-   Índices ANN.
-   Grafo de conocimiento.
-   Recuperación híbrida.
-   Expansión automática del contexto.

Todo ello integrado sobre SQLite mediante una arquitectura modular y
extensible.
