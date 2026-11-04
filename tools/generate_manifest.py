#!/usr/bin/env python3
"""Generate manifest.json for the BridgeUpdate repository.

Run this AFTER committing the files that should be updated. The script reads
the Git blobs from the selected ref (HEAD by default), not the Windows working
tree, so CRLF conversion cannot corrupt the SHA-256 values.

Usage:
    python tools/generate_manifest.py --version 2026.11.04-4
    python tools/generate_manifest.py --version 2026.11.04-4 --ref origin/main

Then commit and push both the changed files and manifest.json.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT_PREFIXES = (
    "etc/synaptic-bridge/",
    "home/pico/synaptic-bridge/luckfox-device/",
    "home/pico/synaptic_cortex/",
)
MANIFEST_NAME = "manifest.json"


def git(repo: Path, *args: str, text: bool = False):
    cmd = ["git", "-C", str(repo), *args]
    return subprocess.check_output(cmd, text=text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version",
        default=datetime.now().strftime("%Y.%m.%d-%H%M"),
        help="Manifest version (default: current local date/time)",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Git ref to hash (default: HEAD)",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    names = git(repo, "ls-tree", "-r", "-z", "--name-only", args.ref, text=True)
    paths = [item for item in names.split("\0") if item]

    files = []
    current = set()
    for rel in sorted(paths):
        rel = rel.replace("\\", "/")
        if rel == MANIFEST_NAME or not rel.startswith(ROOT_PREFIXES):
            continue
        current.add(rel)
        data = git(repo, "show", f"{args.ref}:{rel}")
        files.append({
            "path": rel,
            "sha256": hashlib.sha256(data).hexdigest(),
        })

    previous = set()
    try:
        old = json.loads(git(repo, "show", f"{args.ref}:{MANIFEST_NAME}", text=True))
        previous = {
            item.get("path") for item in old.get("files", [])
            if isinstance(item, dict) and item.get("path")
        }
    except (subprocess.CalledProcessError, ValueError):
        pass

    payload = {
        "version": str(args.version),
        "bridge_api": 2,
        "min_gui": "2.3.0",
        "notes": "BridgeUpdate snapshot: bridge, plugins, cortex.",
        "allow_code": True,
        "restart": True,
        "files": files,
        "removed": sorted(previous - current),
    }
    out = repo / MANIFEST_NAME
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"{out}: {len(files)} files, version {payload['version']}")
    print("Next: git add manifest.json && git commit -m 'update manifest' && git push")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
