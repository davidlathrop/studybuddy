"""WER, memory and machine-info helpers shared by the benchmark scripts."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import resource
import subprocess
import unicodedata
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import jiwer

# A few US/UK spelling pairs, mapped to US so a British voice or a model's spelling choice
# is not counted as an error. The generated text avoids most such words.
_SPELLING = {
    "vapour": "vapor", "colour": "color", "colours": "colors", "behaviour": "behavior",
    "favour": "favor", "favours": "favors", "labour": "labor", "organised": "organized",
    "realised": "realized", "realise": "realize", "fertiliser": "fertilizer",
    "signalling": "signaling", "tumour": "tumor", "centre": "center", "metre": "meter",
    "grey": "gray", "analyse": "analyze", "travelling": "traveling", "modelling": "modeling",
    "programme": "program", "defence": "defense", "neighbour": "neighbor",
}
_DASHES = re.compile(r"[-‐-―/]")
_NOT_WORD = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation (hyphens and slashes become spaces), collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    text = _NOT_WORD.sub("", _DASHES.sub(" ", text))
    return " ".join(_SPELLING.get(w, w) for w in text.split())


@dataclass
class WerResult:
    wer: float
    ref_words: int
    hyp_words: int
    substitutions: int
    deletions: int
    insertions: int
    buckets: list[dict[str, Any]] = field(default_factory=list)  # WER per stretch of the file

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def word_error_rate(
    reference_paragraphs: list[str],
    hypothesis: str,
    *,
    paragraph_starts: list[float] | None = None,
    bucket_s: float | None = None,
) -> WerResult:
    """WER of ``hypothesis`` against the paragraphs of the reference, after :func:`normalize`.

    With ``paragraph_starts`` and ``bucket_s``, errors are also split into time buckets by
    the reference word they fall on (insertions go to the next reference word), giving the
    WER of each stretch of the file, e.g. to see whether quality drops late in a long file.
    """
    ref_words: list[str] = []
    word_time: list[float] = []
    for i, para in enumerate(reference_paragraphs):
        words = normalize(para).split()
        ref_words += words
        if paragraph_starts is not None:
            word_time += [paragraph_starts[i]] * len(words)
    hyp = normalize(hypothesis)
    out = jiwer.process_words(" ".join(ref_words), hyp if hyp else "<empty>")
    result = WerResult(
        wer=out.wer, ref_words=len(ref_words), hyp_words=len(hyp.split()),
        substitutions=out.substitutions, deletions=out.deletions, insertions=out.insertions,
    )
    if paragraph_starts is None or not bucket_s or not ref_words:
        return result

    n_buckets = int(max(word_time) // bucket_s) + 1
    errors, words = [0] * n_buckets, [0] * n_buckets
    for t in word_time:
        words[int(t // bucket_s)] += 1
    for chunk in out.alignments[0]:
        if chunk.type in ("substitute", "delete"):
            for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                errors[int(word_time[i] // bucket_s)] += 1
        elif chunk.type == "insert":
            i = min(chunk.ref_start_idx, len(word_time) - 1)
            errors[int(word_time[i] // bucket_s)] += chunk.hyp_end_idx - chunk.hyp_start_idx
    result.buckets = [
        {"start_s": b * bucket_s, "end_s": (b + 1) * bucket_s, "ref_words": words[b],
         "errors": errors[b], "wer": errors[b] / words[b] if words[b] else None}
        for b in range(n_buckets)
    ]
    return result


def load_reference(media: Path) -> tuple[list[str], list[float]]:
    """(paragraph texts, paragraph start times) from the manifest next to ``media``."""
    manifest = json.loads(media.with_suffix(".json").read_text(encoding="utf-8"))
    paragraphs = manifest["paragraphs"]
    return [p["text"] for p in paragraphs], [p["start_s"] for p in paragraphs]


# --------------------------------------------------------------------------- memory


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
            "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time",
        )
    ]


_RUSAGE_INFO_V4 = 4


def peak_footprint_bytes(pid: int | None = None) -> int | None:
    """Lifetime peak "physical footprint" of a process (what Activity Monitor calls Memory).

    Unlike max RSS, this includes Metal (GPU) buffers, which is where MLX keeps the model
    weights and activations on Apple silicon.
    """
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        info = _RusageInfoV4()
        rc = libc.proc_pid_rusage(os.getpid() if pid is None else pid, _RUSAGE_INFO_V4, ctypes.byref(info))
        return int(info.ri_lifetime_max_phys_footprint) if rc == 0 else None
    except (OSError, AttributeError):
        return None


def max_rss_bytes() -> int:
    """Peak resident set size of this process (macOS reports ru_maxrss in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def max_rss_children_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss


def gb(n: float | None) -> float | None:
    return None if n is None else round(n / 1e9, 3)


# --------------------------------------------------------------------------- machine


def machine_info() -> dict[str, Any]:
    def sysctl(name: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""

    def pkg(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.split("\n")[0]
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "model": sysctl("hw.model"),
        "memory_gb": round(int(sysctl("hw.memsize") or 0) / 2**30),
        "cpu_cores": f"{sysctl('hw.perflevel0.physicalcpu')}P+{sysctl('hw.perflevel1.physicalcpu')}E",
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "mlx": pkg("mlx"),
        "mlx_audio": pkg("mlx-audio"),
        "numpy": pkg("numpy"),
        "ffmpeg": ffmpeg,
        "power": power.replace("Now drawing from ", "").strip("'"),
    }
