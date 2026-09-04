#!/usr/bin/env python3
"""Run a command while sampling process-tree RSS and NVIDIA memory."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def process_tree_rss(root_pid):
    try:
        output = subprocess.check_output(
            ["ps", "-e", "-o", "pid=,ppid=,rss="], text=True
        )
    except Exception:
        return 0, []
    table = {}
    children = {}
    for line in output.splitlines():
        try:
            pid, parent, rss = map(int, line.split())
        except ValueError:
            continue
        table[pid] = rss
        children.setdefault(parent, []).append(pid)
    stack = [root_pid]
    descendants = []
    while stack:
        pid = stack.pop()
        if pid in descendants:
            continue
        descendants.append(pid)
        stack.extend(children.get(pid, []))
    return 1024 * sum(table.get(pid, 0) for pid in descendants), descendants


def gpu_memory_for_pids(pids):
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return 0
    wanted = set(pids)
    total_mib = 0
    for line in output.splitlines():
        try:
            pid_text, memory_text = line.split(",", 1)
            if int(pid_text.strip()) in wanted:
                total_mib += int(memory_text.strip())
        except ValueError:
            continue
    return total_mib * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("A command is required after --")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    process = subprocess.Popen(command, env=os.environ.copy())
    peak_rss = peak_gpu = 0
    samples = 0
    while process.poll() is None:
        rss, pids = process_tree_rss(process.pid)
        gpu = gpu_memory_for_pids(pids)
        peak_rss = max(peak_rss, rss)
        peak_gpu = max(peak_gpu, gpu)
        samples += 1
        time.sleep(args.interval)
    rss, pids = process_tree_rss(process.pid)
    peak_rss = max(peak_rss, rss)
    peak_gpu = max(peak_gpu, gpu_memory_for_pids(pids))
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "command": command,
        "exit_code": process.returncode,
        "elapsed_s": time.perf_counter() - started,
        "sampling_interval_s": args.interval,
        "samples": samples,
        "peak_process_tree_rss_bytes": peak_rss,
        "peak_gpu_memory_bytes": peak_gpu,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
