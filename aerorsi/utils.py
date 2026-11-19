"""Hashing and JSON helpers shared by every stage."""
import hashlib
import json
from pathlib import Path


def digest(value) -> str:
    """SHA-256 of a JSON value; used for identities and for seeded, order-independent shuffles."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seeded_order(items, seed, salt, key=lambda x: x):
    """Sort items by digest([seed, salt, key(item)]): a reproducible shuffle that does not depend on input order."""
    return sorted(items, key=lambda item: digest([seed, salt, key(item)]))


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value, overwrite=False):
    """Write JSON; by default refuse to replace an existing file, so that no stage silently overwrites results."""
    with Path(path).open("w" if overwrite else "x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return file_hash(path)
