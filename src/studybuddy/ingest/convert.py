"""PowerPoint and Word files to PDF with the Mac's own Office apps, for exam-day viewing (plan §5.2).

Driven through ``osascript``. Office is sandboxed: files outside its container trigger a
"Grant File Access" dialog, so each conversion runs on a copy inside the app's container
and the PDF is moved out afterwards. macOS asks once for permission to control the app.

Use a :class:`Converter` as a context manager: apps it had to launch are quit at the end,
apps that were already open are left alone.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

log = logging.getLogger(__name__)

TIMEOUT_S = 300.0


class ConversionError(Exception):
    """A file that could not be converted to PDF; ingestion goes on without a view PDF."""


@dataclass(frozen=True)
class _App:
    name: str
    bundle_id: str
    language: str  # osascript -l
    script: str  # receives (input path, output path) as argv


_POWERPOINT = _App(
    "Microsoft PowerPoint",
    "com.microsoft.Powerpoint",
    "AppleScript",
    """
on run argv
    set inPath to item 1 of argv
    set outPath to item 2 of argv
    tell application "Microsoft PowerPoint"
        open (POSIX file inPath)
        set pres to active presentation
        save pres in (POSIX file outPath) as save as PDF
        close pres saving no
    end tell
end run
""",
)

_WORD = _App(
    "Microsoft Word",
    "com.microsoft.Word",
    "JavaScript",
    """
function run(argv) {
    const word = Application("Microsoft Word");
    word.open(Path(argv[0]));
    const doc = word.activeDocument;
    doc.saveAs({ fileName: argv[1], fileFormat: "format PDF" });
    doc.close({ saving: "no" });
}
""",
)

_APPS = {".pptx": _POWERPOINT, ".docx": _WORD}


def _osascript(language: str, script: str, *args: str, timeout_s: float = 30.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["osascript", "-l", language, "-e", script, *args],
        capture_output=True, text=True, timeout=timeout_s,
    )


def _installed(app: _App) -> bool:
    if shutil.which("osascript") is None:
        return False
    try:
        proc = _osascript("AppleScript", f'id of application "{app.name}"')
    except subprocess.TimeoutExpired:
        return False
    return proc.returncode == 0


def _running(app: _App) -> bool:
    proc = _osascript("AppleScript", f'application "{app.name}" is running')
    return proc.stdout.strip() == "true"


def _quit(app: _App) -> None:
    try:
        _osascript("AppleScript", f'tell application "{app.name}" to quit saving no', timeout_s=60.0)
    except subprocess.TimeoutExpired:
        log.warning("%s did not quit; close it yourself", app.name)


@dataclass
class Converter:
    timeout_s: float = TIMEOUT_S
    _launched: set[str] = field(default_factory=set)
    _checked: dict[str, bool] = field(default_factory=dict)

    def __enter__(self) -> Converter:
        return self

    def __exit__(self, kind: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None) -> None:
        for app in _APPS.values():
            if app.name in self._launched:
                _quit(app)
        self._launched.clear()

    def can_convert(self, path: Path) -> bool:
        app = _APPS.get(Path(path).suffix.lower())
        if app is None:
            return False
        if app.name not in self._checked:
            self._checked[app.name] = _installed(app)
            if not self._checked[app.name]:
                log.warning("%s is not installed; %s files get no PDF for viewing", app.name, Path(path).suffix)
        return self._checked[app.name]

    def to_pdf(self, src: Path, dst: Path) -> Path:
        """Convert ``src`` (.pptx or .docx) to ``dst`` (.pdf); raise ConversionError on failure."""
        src, dst = Path(src), Path(dst)
        app = _APPS.get(src.suffix.lower())
        if app is None:
            raise ConversionError(f"cannot convert {src.suffix} files")
        if not self.can_convert(src):
            raise ConversionError(f"{app.name} is not installed")
        if app.name not in self._launched and not _running(app):
            self._launched.add(app.name)

        work = _work_dir(app)
        try:
            staged_in = work / f"input{src.suffix.lower()}"
            staged_out = work / "output.pdf"
            shutil.copy2(src, staged_in)
            log.info("Converting %s to PDF with %s", src.name, app.name)
            try:
                proc = _osascript(app.language, app.script, str(staged_in), str(staged_out), timeout_s=self.timeout_s)
            except subprocess.TimeoutExpired:
                raise ConversionError(f"{app.name} took longer than {self.timeout_s:.0f}s to convert {src.name}") from None
            if proc.returncode != 0 or not staged_out.is_file() or staged_out.stat().st_size == 0:
                detail = (proc.stderr or proc.stdout).strip() or f"exit status {proc.returncode}"
                raise ConversionError(f"{app.name} could not convert {src.name}: {detail}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(staged_out, dst)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return dst


def _work_dir(app: _App) -> Path:
    """A fresh folder the app may write to without asking: inside its sandbox container if it has one."""
    container = Path.home() / "Library" / "Containers" / app.bundle_id / "Data"
    base = container / "studybuddy-convert" if container.is_dir() else Path.home() / ".cache" / "studybuddy-convert"
    work = base / uuid.uuid4().hex
    work.mkdir(parents=True)
    return work
