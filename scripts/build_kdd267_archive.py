#!/usr/bin/env python3
"""Build a deterministic KDD267 source archive from the public tree."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "ehrdyn-icu-2.1.1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_files(root: Path) -> list[Path]:
    output = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return [
        root / relative
        for relative in sorted(set(output))
        if (root / relative).is_file()
    ]


def build(root: Path, output: Path) -> tuple[str, int]:
    files = _public_files(root)
    buffer = io.BytesIO()
    with tarfile.open(
        fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT
    ) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(f"{PREFIX}/{relative}")
            info.size = len(data)
            info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=handle, mtime=0
        ) as compressed:
            compressed.write(buffer.getvalue())
    return _sha256(output), len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first_hash, first_count = build(ROOT, args.output)
    second = args.output.with_name(args.output.name + ".replay")
    second_hash, second_count = build(ROOT, second)
    second.unlink()
    if (first_hash, first_count) != (second_hash, second_count):
        raise RuntimeError("KDD267 source archive is not deterministic")
    checksum = args.output.with_name(args.output.name + ".sha256")
    checksum.write_text(
        f"{first_hash}  {args.output.name}\n", encoding="utf-8"
    )
    print(
        {
            "archive_sha256": first_hash,
            "files": first_count,
            "bytes": args.output.stat().st_size,
            "deterministic": True,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
