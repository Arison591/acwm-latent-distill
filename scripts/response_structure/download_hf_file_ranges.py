#!/usr/bin/env python3
"""Concurrent range downloader for a single Hugging Face Hub file.

This is intentionally small and dependency-light.  The HF CLI can parallelize
multi-file downloads, but a single large LFS file may still arrive as one slow
stream.  Here we resolve the Hub URL with the token kept in process memory, then
download byte ranges into resumable chunk files on the data disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


def _request(url: str, token: str | None, method: str = "GET", range_header: str | None = None):
    headers = {"User-Agent": "acwm-response-range-downloader/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if range_header:
        headers["Range"] = range_header
    req = urllib.request.Request(url, headers=headers, method=method)
    return urllib.request.urlopen(req, timeout=60)


def _resolve(repo_id: str, filename: str, revision: str, token: str | None) -> tuple[str, int, str | None]:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_name = urllib.parse.quote(filename, safe="/")
    url = f"https://huggingface.co/{quoted_repo}/resolve/{revision}/{quoted_name}"

    with _request(url, token, method="HEAD") as resp:
        final_url = resp.geturl()
        size = int(resp.headers["Content-Length"])
        etag = resp.headers.get("ETag")
    return final_url, size, etag


def _download_chunk(
    *,
    url: str,
    token: str | None,
    chunk_path: Path,
    start: int,
    end: int,
    retries: int,
) -> None:
    expected = end - start + 1
    if chunk_path.exists() and chunk_path.stat().st_size == expected:
        return

    tmp_path = chunk_path.with_suffix(chunk_path.suffix + ".tmp")
    existing = tmp_path.stat().st_size if tmp_path.exists() else 0
    if existing > expected:
        tmp_path.unlink()
        existing = 0

    for attempt in range(retries + 1):
        try:
            range_header = f"bytes={start + existing}-{end}"
            with _request(url, token, range_header=range_header) as resp:
                status = getattr(resp, "status", None)
                if status not in (200, 206):
                    raise RuntimeError(f"unexpected HTTP status {status} for {range_header}")
                mode = "ab" if existing else "wb"
                with tmp_path.open(mode) as f:
                    shutil.copyfileobj(resp, f, length=1024 * 1024)

            if tmp_path.stat().st_size != expected:
                existing = tmp_path.stat().st_size
                raise RuntimeError(f"incomplete chunk {chunk_path.name}: {existing}/{expected}")
            tmp_path.rename(chunk_path)
            return
        except Exception:
            if attempt >= retries:
                raise
            time.sleep(min(30, 2**attempt))
            existing = tmp_path.stat().st_size if tmp_path.exists() else 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-size-mb", type=int, default=64)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--token-env", default="HF_TOKEN")
    args = parser.parse_args()

    token = os.environ.get(args.token_env)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = args.output.with_suffix(args.output.suffix + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)

    final_url, total_size, etag = _resolve(args.repo_id, args.filename, args.revision, token)
    chunk_size = args.chunk_size_mb * 1024 * 1024
    ranges = []
    for start in range(0, total_size, chunk_size):
        end = min(start + chunk_size - 1, total_size - 1)
        index = len(ranges)
        ranges.append((index, start, end, chunk_dir / f"chunk_{index:05d}"))

    manifest = {
        "repo_id": args.repo_id,
        "filename": args.filename,
        "revision": args.revision,
        "output": str(args.output),
        "total_size": total_size,
        "etag": etag,
        "chunk_size": chunk_size,
        "chunks": len(ranges),
        "workers": args.workers,
    }
    (chunk_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(
        json.dumps(
            {
                "event": "start",
                "output": str(args.output),
                "total_size": total_size,
                "chunks": len(ranges),
                "workers": args.workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                _download_chunk,
                url=final_url,
                token=token,
                chunk_path=chunk_path,
                start=start,
                end=end,
                retries=args.retries,
            )
            for _, start, end, chunk_path in ranges
        ]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()
            done += 1
            if done == len(ranges) or done % 4 == 0:
                print(json.dumps({"event": "progress", "chunks_done": done, "chunks_total": len(ranges)}), flush=True)

    tmp_output = args.output.with_suffix(args.output.suffix + ".assembling")
    with tmp_output.open("wb") as out:
        for _, _, _, chunk_path in ranges:
            with chunk_path.open("rb") as chunk:
                shutil.copyfileobj(chunk, out, length=8 * 1024 * 1024)
    if tmp_output.stat().st_size != total_size:
        raise RuntimeError(f"assembled size mismatch: {tmp_output.stat().st_size}/{total_size}")
    tmp_output.rename(args.output)

    digest = _sha256(args.output)
    (args.output.with_suffix(args.output.suffix + ".sha256")).write_text(f"{digest}  {args.output.name}\n")
    print(json.dumps({"event": "complete", "output": str(args.output), "sha256": digest}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, OSError) as exc:
        print(json.dumps({"event": "error", "message": str(exc)}, sort_keys=True), file=sys.stderr, flush=True)
        raise
