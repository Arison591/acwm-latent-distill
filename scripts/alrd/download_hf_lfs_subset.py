#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


POINTER_RE = re.compile(r"oid sha256:([0-9a-f]{64})\nsize (\d+)\n?$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a checked subset from a Hugging Face Git-LFS checkout.")
    parser.add_argument("--repo", required=True, help="Hub repo id, e.g. t1an/ACWM-Phys")
    parser.add_argument("--pointer_root", type=Path, required=True)
    parser.add_argument("--destination_root", type=Path, required=True)
    parser.add_argument("--glob", action="append", required=True, dest="globs")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pointers = {}
    for pattern in args.globs:
        for path in args.pointer_root.glob(pattern):
            if not path.is_file():
                continue
            match = POINTER_RE.search(path.read_text(errors="ignore"))
            if match:
                pointers[path.relative_to(args.pointer_root)] = (match.group(1), int(match.group(2)))
    if not pointers:
        raise RuntimeError("no Git-LFS pointers matched")

    downloaded = existing = 0
    items = list(pointers.items())
    for offset in range(0, len(items), args.batch_size):
        batch = items[offset:offset + args.batch_size]
        actions = _batch_actions(args.repo, [value for _, value in batch])
        def download_one(item):
            relative, (oid, size) = item
            target = args.destination_root / relative
            if _valid(target, oid, size):
                return "existing"
            action = actions.get(oid)
            if not action:
                raise RuntimeError(f"no download action returned for {relative} ({oid})")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            request = urllib.request.Request(action["href"], headers=action.get("header", {}))
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if not _valid(temporary, oid, size):
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"checksum or size mismatch for {relative}")
            os.replace(temporary, target)
            return "downloaded"
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            statuses = list(executor.map(download_one, batch))
        downloaded += statuses.count("downloaded")
        existing += statuses.count("existing")
        print(f"verified {min(offset + len(batch), len(items))}/{len(items)}", flush=True)
    print(json.dumps({"matched": len(items), "downloaded": downloaded, "already_valid": existing}, sort_keys=True))


def _batch_actions(repo: str, objects: list[tuple[str, int]]) -> dict:
    payload = json.dumps({
        "operation": "download",
        "transfers": ["basic"],
        "objects": [{"oid": oid, "size": size} for oid, size in objects],
    }).encode()
    headers = {"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"https://huggingface.co/datasets/{repo}.git/info/lfs/objects/batch",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    errors = [item for item in result.get("objects", []) if "error" in item]
    if errors:
        raise RuntimeError(f"LFS batch errors: {errors[:3]}")
    return {item["oid"]: item.get("actions", {}).get("download") for item in result.get("objects", [])}


def _valid(path: Path, oid: str, size: int) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest() == oid


if __name__ == "__main__":
    main()
