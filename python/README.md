# semanta (Python loader)

Thin local package that bundles the compiled Semanta SQLite extension and a
small helper to load it — Semanta itself has no Python bindings, it's a
native SQLite extension consumed entirely through SQL.

This is currently a **local-only build**, not published to PyPI: the
compiled binary is platform-specific and isn't committed to the repo, so it
has to be built and installed per machine.

## Build and install

From the repo root:

```sh
cargo build --release
cp target/release/libsemanta.so python/semanta/   # .dylib on macOS, .dll on Windows
pip install -e python/
```

## Usage

```python
import sqlite3
import semanta

con = sqlite3.connect("my.db")
semanta.load(con)

doc_id = con.execute(
    "SELECT semanta_add_document(?, ?, NULL, NULL, ?)",
    ("manual.md", markdown_text, "manual-v1"),
).fetchone()[0]
```

See the main `README.md` for the full SQL surface.
