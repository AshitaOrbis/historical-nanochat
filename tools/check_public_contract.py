#!/usr/bin/env python3
"""Fail when the README or build metadata names Python code that does not ship.

Run without arguments while developing, and with ``--git-ref HEAD`` to check
the exact regular-file set a clean clone of a committed revision will receive.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path


MODULE_COMMAND = re.compile(
    r"(?m)^\s*(?:python|python3)\s+-m\s+([A-Za-z_][A-Za-z0-9_.]*)"
)


@dataclass(frozen=True)
class CandidateTree:
    root: Path
    git_ref: str | None

    def _git(self, *args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(self.root), *args])

    def read_text(self, path: str) -> str:
        if self.git_ref is None:
            return (self.root / path).read_text(encoding="utf-8")
        return self._git("show", f"{self.git_ref}:{path}").decode("utf-8")

    def regular_files(self) -> set[str]:
        if self.git_ref is None:
            return {
                path.relative_to(self.root).as_posix()
                for path in self.root.rglob("*")
                if path.is_file() and not path.is_symlink() and ".git" not in path.parts
            }

        records = self._git("ls-tree", "-r", "-z", self.git_ref).split(b"\0")
        files: set[str] = set()
        for record in records:
            if not record:
                continue
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, _sha = metadata.split(b" ", 2)
            if mode == b"100644" or mode == b"100755":
                if object_type == b"blob":
                    files.add(raw_path.decode("utf-8"))
        return files


def module_candidates(module: str) -> tuple[str, str]:
    base = module.replace(".", "/")
    return f"{base}.py", f"{base}/__main__.py"


def check_contract(tree: CandidateTree) -> list[str]:
    files = tree.regular_files()
    errors: list[str] = []

    metadata = tomllib.loads(tree.read_text("pyproject.toml"))
    packages = (
        metadata.get("tool", {})
        .get("hatch", {})
        .get("build", {})
        .get("targets", {})
        .get("wheel", {})
        .get("packages", [])
    )
    package_roots = {package.strip("/").split("/")[-1] for package in packages}

    readme = tree.read_text("README.md")
    modules = sorted(
        module
        for module in set(MODULE_COMMAND.findall(readme))
        if module.split(".", 1)[0] in package_roots
    )
    for module in modules:
        candidates = module_candidates(module)
        if not any(candidate in files for candidate in candidates):
            errors.append(
                f"README command `python -m {module}` has no shipped module "
                f"({candidates[0]} or {candidates[1]})"
            )

    for package in packages:
        prefix = package.rstrip("/") + "/"
        if not any(path.startswith(prefix) and path.endswith(".py") for path in files):
            errors.append(
                f"pyproject wheel package {package!r} has no shipped Python sources"
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--git-ref",
        help="check the exact regular-file tree at this Git revision (for example HEAD)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    tree = CandidateTree(root=root, git_ref=args.git_ref)
    errors = check_contract(tree)
    if errors:
        for error in errors:
            print(f"PUBLIC CONTRACT ERROR: {error}", file=sys.stderr)
        print(f"PUBLIC CONTRACT: FAIL ({len(errors)} violation(s))", file=sys.stderr)
        return 1

    source = f"git ref {args.git_ref}" if args.git_ref else "working tree"
    print(f"PUBLIC CONTRACT: PASS ({source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
