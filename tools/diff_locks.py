"""Compare two uv lock files by package version.

Used to diagnose why the CI lock freshness check fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERN = re.compile(r"^([A-Za-z0-9._-]+)==([^\s\\]+)")


def pkgs(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = PATTERN.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main() -> None:
    a = pkgs(Path(sys.argv[1]))
    b = pkgs(Path(sys.argv[2]))
    print(f"checked-in: {len(a)} packages   regenerated: {len(b)} packages")
    changed = 0
    for name in sorted(set(a) | set(b)):
        if a.get(name) == b.get(name):
            continue
        changed += 1
        if name not in a:
            tag = "  NEW"
        elif name not in b:
            tag = "  REMOVED"
        else:
            tag = ""
        print(f"  {name:28s} {str(a.get(name)):14s} -> {str(b.get(name)):14s}{tag}")
    print(f"\n{changed} package(s) differ.")


if __name__ == "__main__":
    main()
