"""Patch MLflow 3.14.x for Python 3.14 (Traversable import moved in stdlib)."""

from __future__ import annotations

import sys
from pathlib import Path


def patch_skill_installer(skill_installer: Path) -> bool:
    text = skill_installer.read_text(encoding="utf-8")
    old = "from importlib.abc import Traversable"
    new = """try:
    from importlib.resources.abc import Traversable
except ImportError:
    from importlib.abc import Traversable"""
    if old not in text:
        return False
    skill_installer.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def main() -> int:
    if sys.version_info < (3, 14):
        print("Python < 3.14: no MLflow patch needed.")
        return 0

    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed.")
        return 1

    skill_installer = Path(mlflow.__file__).resolve().parent / "assistant" / "skill_installer.py"
    if not skill_installer.exists():
        print(f"File not found: {skill_installer}")
        return 1

    if patch_skill_installer(skill_installer):
        print(f"Patched {skill_installer}")
    else:
        print("Patch already applied or MLflow version uses a different import.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
