"""Patch MLflow 3.14.x for Python 3.14 (Traversable import moved in stdlib)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PATCH_MARKER = "from importlib.resources.abc import Traversable"
ORIGINAL_IMPORT = "from importlib.abc import Traversable"
PATCHED_IMPORT = """try:
    from importlib.resources.abc import Traversable
except ImportError:
    from importlib.abc import Traversable"""


def _is_valid_python(source: str) -> bool:
    try:
        compile(source, "skill_installer.py", "exec")
        return True
    except SyntaxError:
        return False


def _restore_mlflow_package() -> None:
    import mlflow

    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-deps",
            f"mlflow=={mlflow.__version__}",
        ],
        check=True,
    )


def patch_skill_installer(skill_installer: Path) -> str:
    text = skill_installer.read_text(encoding="utf-8")

    if PATCH_MARKER in text and _is_valid_python(text):
        return "already_ok"

    if PATCH_MARKER in text and not _is_valid_python(text):
        _restore_mlflow_package()
        text = skill_installer.read_text(encoding="utf-8")

    if ORIGINAL_IMPORT not in text:
        if PATCH_MARKER in text and _is_valid_python(text):
            return "already_ok"
        raise RuntimeError(
            f"Unexpected {skill_installer.name} content; expected {ORIGINAL_IMPORT!r}."
        )

    patched = text.replace(ORIGINAL_IMPORT, PATCHED_IMPORT, 1)
    if not _is_valid_python(patched):
        raise RuntimeError(f"Patch would produce invalid Python in {skill_installer}.")

    skill_installer.write_text(patched, encoding="utf-8")
    return "patched"


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

    try:
        result = patch_skill_installer(skill_installer)
    except RuntimeError as exc:
        print(exc)
        return 1

    if result == "patched":
        print(f"Patched {skill_installer}")
    else:
        print("Patch already applied.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
