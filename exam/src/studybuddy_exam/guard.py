"""The no-AI, no-network guard (plans/exam-search-plan.md §8, constraint C1).

:func:`install` adds a PEP 578 audit hook, which cannot be removed once added. From then on
the process refuses to:

* import any AI or machine-learning library, or the studybuddy builder (which contains AI);
* connect, send, resolve names or bind anywhere except this computer's loopback address.

The exam app calls it as its very first action, before importing anything else, and the
status page reports what it checked. Native code making raw system calls could bypass an
audit hook, so keep Wi-Fi off during the exam as well.
"""

from __future__ import annotations

import ipaddress
import sys
from typing import Any

# Top-level packages that run or call models. Anything else is ordinary code.
BLOCKED_MODULES = frozenset(
    {
        # model runtimes and ML frameworks
        "mlx", "mlx_audio", "mlx_lm", "torch", "torchaudio", "torchvision", "tensorflow", "keras",
        "jax", "flax", "onnxruntime", "llama_cpp", "ctranslate2", "faster_whisper", "whisper",
        "transformers", "sentence_transformers", "sklearn", "spacy", "gensim", "tokenizers",
        "huggingface_hub", "coremltools", "ocrmac", "Vision", "CoreML", "NaturalLanguage",
        # model provider clients
        "anthropic", "claude_agent_sdk", "openai", "boto3", "botocore", "aiobotocore",
        "google.generativeai", "google.genai", "vertexai", "cohere", "mistralai", "ollama",
        "groq", "litellm", "langchain", "langchain_core", "llama_index",
        # the builder: transcription and enrichment live there
        "studybuddy",
    }
)

_LOOPBACK_NAMES = frozenset({"localhost", "localhost.", "ip6-localhost"})
_installed = False


class GuardViolation(PermissionError):
    """Something tried to use AI or the network during the exam."""


def installed() -> bool:
    return _installed


def is_blocked(module: str) -> bool:
    """True for a blocked package or any of its submodules (``boto3.session``, ...)."""
    parts = module.split(".")
    return any(".".join(parts[: i + 1]) in BLOCKED_MODULES for i in range(len(parts)))


def loaded_blocked_modules() -> list[str]:
    """Blocked modules present in ``sys.modules`` (should always be empty)."""
    return sorted(name for name in list(sys.modules) if is_blocked(name))


def is_loopback(host: Any) -> bool:
    if host is None:
        return True  # getaddrinfo(None, port) and bind(("", port)) mean "this machine"
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    host = str(host).strip("[]").split("%")[0]
    if host in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _address_host(address: Any) -> Any:
    if isinstance(address, tuple) and address:
        return address[0]
    if isinstance(address, (str, bytes)):  # AF_UNIX path: local by definition
        return "localhost"
    return None


def _hook(event: str, args: tuple[Any, ...]) -> None:
    if event == "import":
        module = args[0]
        if isinstance(module, str) and is_blocked(module):
            raise GuardViolation(f"exam mode: importing {module!r} is not allowed (AI libraries are blocked)")
    elif event in ("socket.connect", "socket.sendto", "socket.bind"):
        host = _address_host(args[1]) if len(args) > 1 else None
        if host == "" and event == "socket.bind":
            raise GuardViolation("exam mode: listening on every network interface is not allowed")
        if not is_loopback(host):
            raise GuardViolation(f"exam mode: network access to {host!r} is blocked (loopback only)")
    elif event == "socket.getaddrinfo":
        if not is_loopback(args[0]):
            raise GuardViolation(f"exam mode: looking up {args[0]!r} is blocked (loopback only)")
    elif event == "socket.gethostbyname":
        if not is_loopback(args[0]):
            raise GuardViolation(f"exam mode: looking up {args[0]!r} is blocked (loopback only)")
    elif event == "urllib.Request":
        url = str(args[0])
        if not (url.startswith("http://127.0.0.1") or url.startswith("http://localhost")):
            raise GuardViolation(f"exam mode: fetching {url!r} is blocked (loopback only)")


def install() -> None:
    """Install the guard (once per process). Raises if a blocked module is already loaded."""
    global _installed
    if _installed:
        return
    if already := loaded_blocked_modules():
        raise GuardViolation(f"exam mode: blocked modules were loaded before the guard: {', '.join(already)}")
    sys.addaudithook(_hook)
    _installed = True
