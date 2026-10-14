"""The exam app stays small and AI-free: its dependencies, its imports, and its environment."""

from __future__ import annotations

import ast
import tomllib
from importlib.metadata import distributions
from pathlib import Path

from studybuddy_exam import guard

PACKAGE = Path(__file__).parents[1] / "src" / "studybuddy_exam"
PYPROJECT = Path(__file__).parents[1] / "pyproject.toml"

ALLOWED_DEPENDENCIES = {"starlette", "uvicorn", "rapidfuzz", "jellyfish"}
# Distribution names that would mean an AI or ML library was installed in the exam environment.
FORBIDDEN_DISTRIBUTIONS = {
    "anthropic", "openai", "boto3", "botocore", "torch", "tensorflow", "jax", "mlx", "mlx-audio", "mlx-lm",
    "transformers", "sentence-transformers", "scikit-learn", "onnxruntime", "llama-cpp-python", "ocrmac",
    "pyobjc-framework-vision", "pyobjc-framework-coreml", "huggingface-hub", "tokenizers", "studybuddy",
    "google-genai", "google-generativeai", "cohere", "mistralai", "ollama", "litellm", "langchain",
}


def test_runtime_dependencies_are_the_approved_four() -> None:
    declared = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    names = {d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower() for d in declared}
    assert names == ALLOWED_DEPENDENCIES


def test_no_ai_distribution_is_installed_in_this_environment() -> None:
    installed = {(d.metadata["Name"] or "").lower().replace("_", "-") for d in distributions()}
    assert not installed & FORBIDDEN_DISTRIBUTIONS, sorted(installed & FORBIDDEN_DISTRIBUTIONS)


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return found


def test_source_never_imports_blocked_modules() -> None:
    offenders = {
        (path.name, module)
        for path in PACKAGE.rglob("*.py")
        for module in _imports(path)
        if guard.is_blocked(module)
    }
    assert not offenders


def test_guard_is_the_first_thing_main_does() -> None:
    tree = ast.parse((PACKAGE / "__main__.py").read_text(encoding="utf-8"))
    body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    first, second = body[0], body[1]
    assert isinstance(first, ast.ImportFrom) and first.module == "studybuddy_exam" and first.names[0].name == "guard"
    assert ast.unparse(second) == "guard.install()"


def test_package_init_imports_nothing() -> None:
    assert _imports(PACKAGE / "__init__.py") == set()


def test_lockfile_lists_only_the_runtime_stack() -> None:
    lock = Path(__file__).parents[1] / "requirements.lock"
    pinned = {line.split("==")[0].lower() for line in lock.read_text().splitlines() if "==" in line}
    assert ALLOWED_DEPENDENCIES <= pinned
    assert not pinned & FORBIDDEN_DISTRIBUTIONS
    assert pinned <= ALLOWED_DEPENDENCIES | {"anyio", "click", "h11", "idna", "typing-extensions", "typing_extensions"}
