"""Load the Semanta SQLite extension into a Python sqlite3 connection.

Semanta itself is a native Rust/SQLite extension, not a Python library — this
package only bundles the precompiled binary and a small loader, the same
pattern used by sqlite-vec's Python distribution.
"""

import os
import platform
import sqlite3

_PLATFORM_LIBS = {
    ("Linux", "x86_64"): "libsemanta.so",
    ("Linux", "aarch64"): "libsemanta.so",
    ("Darwin", "x86_64"): "libsemanta.dylib",
    ("Darwin", "arm64"): "libsemanta.dylib",
    ("Windows", "AMD64"): "semanta.dll",
}


def loadable_path() -> str:
    """Absolute path to the compiled Semanta extension for the current platform."""
    key = (platform.system(), platform.machine())
    filename = _PLATFORM_LIBS.get(key)
    if filename is None:
        raise RuntimeError(f"Semanta has no prebuilt binary for {key[0]}/{key[1]} yet")

    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected {path} but it's missing. Build it with `cargo build --release` "
            "in the Semanta repo and copy target/release/libsemanta.* into python/semanta/ "
            "before installing this package."
        )
    return path


def load(connection: sqlite3.Connection) -> None:
    """Load the Semanta extension into an open sqlite3.Connection."""
    connection.enable_load_extension(True)
    try:
        connection.load_extension(loadable_path())
    finally:
        connection.enable_load_extension(False)
