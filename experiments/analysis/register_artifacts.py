#!/usr/bin/env python3
"""Append content-addressed run records while retaining failures/invalidations."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digest(root):
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
    ]
    paths = subprocess.check_output(command, cwd=root, text=True).splitlines()
    excluded_prefixes = ("output/", ".CondaPkg/", "paper/figures/")
    excluded_names = {
        "paper/workshop_draft.pdf",
        "paper/iclr_workshop.pdf",
        "paper/iclr_workshop_source.zip",
    }
    digest = hashlib.sha256()
    retained = []
    for relative in sorted(paths):
        if relative in excluded_names or relative.startswith(excluded_prefixes):
            continue
        path = root / relative
        if not path.is_file():
            continue
        retained.append(relative)
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), retained


def status_counts(value, counts):
    if isinstance(value, dict):
        status = value.get("status")
        if isinstance(status, (str, int)):
            counts[str(status)] = counts.get(str(status), 0) + 1
        for item in value.values():
            status_counts(item, counts)
    elif isinstance(value, list):
        for item in value:
            status_counts(item, counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("output/runs"))
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--invalidations",
        type=Path,
        default=Path("experiments/invalidated_runs.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    invalidations = json.loads(args.invalidations.read_text())
    existing = set()
    if args.registry.exists():
        for line in args.registry.read_text().splitlines():
            if line.strip():
                existing.add(json.loads(line)["sha256"])
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    snapshot, source_paths = source_digest(root)
    args.registry.parent.mkdir(parents=True, exist_ok=True)
    appended = 0
    with args.registry.open("a") as registry:
        for path in sorted(args.run_dir.glob("*.json")):
            artifact_digest = digest_file(path)
            if artifact_digest in existing:
                continue
            try:
                payload = json.loads(path.read_text())
                parse_error = None
            except Exception as error:
                payload = None
                parse_error = str(error)
            counts = {}
            if payload is not None:
                status_counts(payload, counts)
            record = {
                "registry_schema_version": 1,
                "registered_at": datetime.now(timezone.utc).isoformat(),
                "artifact": str(path.resolve()),
                "artifact_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": artifact_digest,
                "json_parse_error": parse_error,
                "result_schema_version": (
                    payload.get("schema_version") if isinstance(payload, dict) else None
                ),
                "created_at": (
                    payload.get("created_at") if isinstance(payload, dict) else None
                ),
                "status_counts": counts,
                "invalidated": path.name in invalidations,
                "invalidation_reason": invalidations.get(path.name),
                "git_commit": commit,
                "source_snapshot_sha256": snapshot,
                "source_file_count": len(source_paths),
            }
            registry.write(json.dumps(record, sort_keys=True) + "\n")
            existing.add(artifact_digest)
            appended += 1
    print(
        json.dumps(
            {
                "registry": str(args.registry),
                "appended": appended,
                "total_unique_digests": len(existing),
                "source_snapshot_sha256": snapshot,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
