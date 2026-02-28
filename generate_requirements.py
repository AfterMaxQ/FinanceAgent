"""
Utility to scan the repository for imported third-party packages and
write a `requirements.txt` with pinned versions based on the current
environment.
"""
from __future__ import annotations

import ast
import sys
from importlib import metadata, util
from pathlib import Path
from typing import Iterable, Set

ROOT = Path(__file__).resolve().parent
OUTPUT_FILE = ROOT / "requirements.txt"

# Directories to skip while scanning
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}


def iter_python_files(base: Path) -> Iterable[Path]:
    for path in base.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        yield path


def collect_imports(py_files: Iterable[Path]) -> Set[str]:
    imports: Set[str] = set()
    for file in py_files:
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split(".")[0]
                    imports.add(base)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base = node.module.split(".")[0]
                    imports.add(base)
    return imports


def is_stdlib(module: str) -> bool:
    if module in {"__future__", "typing"}:
        return True
    if hasattr(sys, "stdlib_module_names"):
        return module in sys.stdlib_module_names  # type: ignore[attr-defined]
    spec = util.find_spec(module)
    if spec is None or spec.origin is None:
        return False
    return "lib" in spec.origin or spec.origin == "built-in"


def is_local_module(module: str) -> bool:
    return (ROOT / module).is_dir() or (ROOT / f"{module}.py").exists()


def map_to_distributions(modules: Iterable[str]) -> dict[str, str]:
    dist_map = metadata.packages_distributions()
    resolved: dict[str, str] = {}
    for module in modules:
        dists = dist_map.get(module)
        if dists:
            resolved[module] = dists[0]
    return resolved


def build_requirements() -> list[str]:
    imports = collect_imports(iter_python_files(ROOT))
    third_party = {
        name
        for name in imports
        if not is_stdlib(name)
        and not is_local_module(name)
        and not name.startswith("_")
    }

    module_to_dist = map_to_distributions(third_party)
    requirements: list[str] = []

    for module in sorted(third_party):
        dist = module_to_dist.get(module)
        if not dist:
            # Skip modules we cannot resolve to a distribution; likely missing or vendored.
            continue
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
        requirements.append(f"{dist}=={version}")

    return sorted(set(requirements), key=str.lower)


def main() -> None:
    requirements = build_requirements()
    OUTPUT_FILE.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    print(f"Wrote {len(requirements)} dependencies to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

