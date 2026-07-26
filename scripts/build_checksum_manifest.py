#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

from kdd2027_benchmark.privacy import _ignored_build_file


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name == "MANIFEST.sha256"
            or _ignored_build_file(path)
        ):
            continue
        relative = path.relative_to(ROOT).as_posix()
        rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
