"""Loading manifests (the JSON lists written by scripts/build_*_manifest.py)."""

from __future__ import annotations

import json
import os


def load_manifests(paths: str, limit: int = 0, require_emb: bool = True,
                   verbose: bool = True) -> list[dict]:
    """Load one or more manifests ("a.json,b.json").

    Each entry gets a "dataset" field (from the entry, else the file name) so
    results can be reported PER DATASET instead of silently pooled. `limit`
    applies per manifest (0 = all). Entries whose embedding file is missing
    are skipped when require_emb is True.
    """
    out = []
    for p in [x.strip() for x in paths.split(",") if x.strip()]:
        with open(p) as f:
            items = json.load(f)
        tag = os.path.splitext(os.path.basename(p))[0]
        kept, missing = 0, 0
        for it in items:
            if require_emb and not os.path.exists(it["emb"]):
                missing += 1
                continue
            it = dict(it)
            it.setdefault("dataset", tag)
            out.append(it)
            kept += 1
            if limit and kept >= limit:
                break
        if verbose:
            print(f"[manifest] {p}: using {kept} songs"
                  + (f" ({missing} skipped: embedding file missing)" if missing else ""))
    return out


def by_dataset(items: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for it in items:
        groups.setdefault(it.get("dataset", "?"), []).append(it)
    return groups
