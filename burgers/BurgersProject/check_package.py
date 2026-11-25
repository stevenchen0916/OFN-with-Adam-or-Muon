"""Verify the delivered archive before executing its campaign."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check():
    manifest = json.loads((HERE / "PACKAGE_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        path = HERE / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Missing or changed package file: " + name)
    print(f"Verified {len(manifest['files'])} delivered files. Formal Burgers results are not bundled.", flush=True)


if __name__ == "__main__":
    check()
