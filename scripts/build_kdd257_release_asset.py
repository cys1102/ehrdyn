#!/usr/bin/env python3
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
PREFIX = "ehrdyn-icu-2.0.2"
EXCLUDED = {
    "MANIFEST.sha256",
}
EXCLUDED_PREFIXES = (
    "release/",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def public_files(root: Path) -> list[Path]:
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
    files = []
    for relative in sorted(set(output)):
        if relative in EXCLUDED or relative.startswith(EXCLUDED_PREFIXES):
            continue
        path = root / relative
        if path.is_file() and "__pycache__" not in path.parts:
            files.append(path)
    return files


def build_archive(root: Path, output: Path) -> tuple[str, int]:
    files = public_files(root)
    manifest = "\n".join(
        f"{sha256(path)}  {path.relative_to(root).as_posix()}" for path in files
    ) + "\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in files:
            relative = path.relative_to(root).as_posix()
            data = path.read_bytes()
            info = tarfile.TarInfo(f"{PREFIX}/{relative}")
            info.size = len(data)
            info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
        data = manifest.encode("utf-8")
        info = tarfile.TarInfo(f"{PREFIX}/MANIFEST.sha256")
        info.size = len(data)
        info.mode = 0o644
        info.mtime = 0
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        tar.addfile(info, io.BytesIO(data))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as archive:
            archive.write(buffer.getvalue())
    checksum = sha256(output)
    checksum_path = output.with_name(output.name + ".sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    return checksum, len(files) + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first_hash, files = build_archive(ROOT, args.output)
    with io.BytesIO() as _:
        second = args.output.with_name(args.output.name + ".second")
        second_hash, second_files = build_archive(ROOT, second)
        second.unlink()
        second.with_name(second.name + ".sha256").unlink()
    if first_hash != second_hash or files != second_files:
        raise RuntimeError("KDD257 release archive is not deterministic")
    print(
        f'{{"archive_sha256":"{first_hash}","files":{files},'
        f'"bytes":{args.output.stat().st_size},"deterministic":true}}'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
