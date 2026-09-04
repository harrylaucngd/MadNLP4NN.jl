#!/usr/bin/env python3
"""Concurrent, resumable HTTP range downloader with response validation."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def request_range(url, start, stop, expected_total=None):
    last_error = None
    for attempt in range(10):
        try:
            if expected_total is not None:
                response = subprocess.run(
                    [
                        "curl",
                        "-sSL",
                        "--fail",
                        "--range",
                        f"{start}-{stop}",
                        "--write-out",
                        "%{http_code}",
                        url,
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    timeout=180,
                )
                data, status = response.stdout[:-3], response.stdout[-3:]
                if status != b"206":
                    last_error = RuntimeError(
                        f"Expected HTTP 206, received {status.decode(errors='replace')}"
                    )
                elif len(data) != stop - start + 1:
                    last_error = RuntimeError(
                        f"Range {start}-{stop} returned {len(data)} bytes"
                    )
                else:
                    return data, expected_total
            else:
                request = urllib.request.Request(
                    url,
                    headers={"Range": f"bytes={start}-{stop}"},
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status == 206 and content_range.startswith(
                        f"bytes {start}-{stop}/"
                    ):
                        data = response.read()
                        total = int(content_range.rsplit("/", 1)[1])
                        if len(data) == stop - start + 1:
                            return data, total
                    last_error = RuntimeError(
                        f"Unexpected ranged response: {response.status} {content_range}"
                    )
        except Exception as error:
            last_error = error
        time.sleep(min(15.0, 0.5 * 2**attempt))
    raise RuntimeError(f"Range {start}-{stop} failed after retries") from last_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--total-size", type=int)
    args = parser.parse_args()
    first, total = request_range(args.url, 0, 0, args.total_size)
    del first
    chunk_size = args.chunk_mib * 1024 * 1024
    chunks = [
        (index, start, min(total - 1, start + chunk_size - 1))
        for index, start in enumerate(range(0, total, chunk_size))
    ]
    state_path = args.output.with_suffix(args.output.suffix + ".ranges.json")
    completed = set()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("url") == args.url and state.get("total") == total:
            completed = set(state.get("completed", []))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_CREAT | os.O_RDWR, 0o664)
    os.ftruncate(descriptor, total)
    lock = threading.Lock()

    def persist_state():
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "url": args.url,
                    "total": total,
                    "chunk_size": chunk_size,
                    "completed": sorted(completed),
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(temporary, state_path)

    def download(chunk):
        index, start, stop = chunk
        if index in completed:
            return index, True
        data, response_total = request_range(args.url, start, stop, total)
        if response_total != total:
            raise RuntimeError("Remote size changed during download")
        os.pwrite(descriptor, data, start)
        with lock:
            completed.add(index)
            persist_state()
            print(
                f"chunk {len(completed)}/{len(chunks)} "
                f"({100 * len(completed) / len(chunks):.1f}%)",
                flush=True,
            )
        return index, False

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(download, chunks))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(f"Wrote {args.output} ({total} bytes)")


if __name__ == "__main__":
    main()
