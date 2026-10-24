"""Console + file logging, stage timers and progress lines for every long-running command.

    from . import log as plog
    log = plog.setup("full_run")            # once, in a main(): console INFO + logs/<UTC>_full_run.log
    with plog.stage("decode the video"):
        for x in plog.track(items, len(items), "decoding", unit="frames"):
            ...

Library modules only call `get(__name__)` and never configure anything, so they stay quiet in the
tests and in any caller that did not ask for logs. The console shows INFO; the file also keeps
DEBUG lines, Python warnings, full tracebacks and -- through faulthandler -- the stack of a hard
crash (a GPU driver fault, say), which is what makes a failed overnight run debuggable.

Messages hold aggregates only: counts, timings, scores. Never log frame-level data. The log files
stay in logs/ (gitignored) all the same.
"""

from __future__ import annotations

import contextlib
import datetime
import faulthandler
import logging
import os
import platform
import subprocess
import sys
import time

from . import ROOT

ROOT_LOGGER = "possession"
_state = {"path": None, "fault_file": None}


def get(name=ROOT_LOGGER):
    """The logger for a module: get(__name__) inside the package."""
    if name != ROOT_LOGGER and not name.startswith(ROOT_LOGGER + "."):
        name = ROOT_LOGGER + "." + name
    return logging.getLogger(name)


def rel(path):
    """`path` relative to the repository when that is possible, else absolute -- os.path.relpath
    raises on Windows when the two are on different drives (data on E:, repo on D:)."""
    try:
        out = os.path.relpath(path, ROOT)
    except ValueError:
        return os.path.abspath(path)
    return out if not out.startswith("..") else os.path.abspath(path)


def log_dir():
    return os.environ.get("MTAH_LOG_DIR") or os.path.join(ROOT, "logs")


def setup(command, verbose=False):
    """Configure console + file logging once per process; later calls reuse the same file."""
    logger = get()
    if _state["path"]:
        return logger
    os.makedirs(log_dir(), exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(log_dir(), "{}_{}.log".format(stamp, command))

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
    to_file = logging.FileHandler(path, encoding="utf-8")
    to_file.setLevel(logging.DEBUG)
    to_file.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-28s %(message)s", "%Y-%m-%d %H:%M:%S"))

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers[:] = [console, to_file]
    logging.captureWarnings(True)
    warn = logging.getLogger("py.warnings")
    warn.handlers[:] = [console, to_file]
    warn.propagate = False

    _state["path"] = path
    _state["fault_file"] = open(path, "a", encoding="utf-8")
    faulthandler.enable(file=_state["fault_file"], all_threads=True)
    previous = sys.excepthook

    def excepthook(kind, value, tb):
        if not issubclass(kind, KeyboardInterrupt):
            logger.critical("unhandled %s: %s", kind.__name__, value, exc_info=(kind, value, tb))
        else:
            logger.warning("interrupted (Ctrl-C)")
        previous(kind, value, tb)
    sys.excepthook = excepthook

    logger.info("log file: %s", rel(path))
    for key, value in environment().items():
        logger.debug("env %-14s %s", key, value)
    logger.debug("argv %s", " ".join(sys.argv))
    return logger


def log_path():
    return _state["path"]


def environment():
    """Interpreter, libraries, git state and hardware -- written at the top of every log file."""
    env = {"python": sys.version.split()[0], "platform": platform.platform(terse=True),
           "cpu": platform.processor(), "cwd": os.getcwd()}
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10).stdout
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
        env["git"] = head + (" (uncommitted changes)" if out.strip() else "")
    except (OSError, subprocess.SubprocessError):
        env["git"] = "unknown"
    try:
        from .config import DEFAULT
        env["config"] = DEFAULT.digest()
    except Exception:       # noqa: BLE001 -- never let the log header fail a run
        pass
    for name in ("numpy", "pandas", "cv2"):
        try:
            env[name] = getattr(__import__(name), "__version__", "?")
        except ImportError:
            pass
    torch = sys.modules.get("torch")        # only if the command already uses it: importing is slow
    if torch is not None:
        env["torch"] = torch.__version__
    if torch is not None:
        env["torch_hip"] = getattr(torch.version, "hip", None)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            env["gpu"] = "{} ({}, {:.1f} GB)".format(torch.cuda.get_device_name(0),
                                                     getattr(p, "gcnArchName", "?"), p.total_memory / 2 ** 30)
        else:
            env["gpu"] = "none visible to torch"
    try:
        import psutil
        env["ram_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except ImportError:
        pass
    return env


def memory():
    """' | RAM 2.1 GB, GPU peak 0.8 GB' -- whatever can be measured, or ''."""
    parts = []
    try:
        import psutil
        parts.append("RAM {:.1f} GB".format(psutil.Process().memory_info().rss / 2 ** 30))
    except ImportError:
        pass
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        parts.append("GPU peak {:.1f} GB".format(torch.cuda.max_memory_allocated() / 2 ** 30))
    return " | " + ", ".join(parts) if parts else ""


def duration(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return "{}:{:02d}:{:02d}".format(h, m, s) if h else "{}:{:02d}".format(m, s)


@contextlib.contextmanager
def stage(name, logger=None):
    """'==> name' on entry, '<== name  0:42 | RAM ...' on exit, the traceback in the file on failure."""
    logger = logger or get()
    logger.info("==> %s", name)
    t0 = time.perf_counter()
    try:
        yield
    except KeyboardInterrupt:
        logger.warning("<== %s INTERRUPTED after %s", name, duration(time.perf_counter() - t0))
        raise
    except BaseException as exc:
        logger.error("<== %s FAILED after %s: %s: %s", name, duration(time.perf_counter() - t0),
                     type(exc).__name__, exc)
        logger.debug("traceback of the failure in '%s'", name, exc_info=True)
        raise
    logger.info("<== %s  %s%s", name, duration(time.perf_counter() - t0), memory())


class Progress:
    """Progress lines with rate and ETA, at most one every `every` seconds plus a final one.

    Plain log lines rather than a redrawn bar, so the file log reads the same as the console.
    """

    def __init__(self, total, desc, unit="", every=10.0, logger=None):
        self.total, self.desc, self.unit, self.every = max(int(total), 0), desc, unit, every
        self.logger = logger or get()
        self.done = 0
        self.t0 = self.last = time.perf_counter()
        self.closed = False

    def update(self, n=1):
        self.set(self.done + n)

    def set(self, done):
        self.done = done
        now = time.perf_counter()
        if now - self.last >= self.every:
            self.last = now
            self._emit(now, final=False)

    def close(self):
        """A final line -- unless the work was too quick to have been worth watching."""
        if not self.closed:
            self.closed = True
            now = time.perf_counter()
            if self.last > self.t0 or now - self.t0 >= 2.0:
                self._emit(now, final=True)

    def _emit(self, now, final):
        elapsed = max(now - self.t0, 1e-9)
        rate = self.done / elapsed
        unit = " " + self.unit if self.unit else ""
        if final:
            self.logger.info("    %s: %d%s in %s (%.0f%s/s)", self.desc, self.done, unit, duration(elapsed),
                             rate, unit)
            return
        if self.total:
            eta = (self.total - self.done) / rate if rate > 0 else float("nan")
            self.logger.info("    %s: %3.0f%%  %d/%d%s  %.0f%s/s  ETA %s", self.desc,
                             100.0 * self.done / self.total, self.done, self.total, unit, rate, unit,
                             duration(eta) if eta == eta else "?")
        else:
            self.logger.info("    %s: %d%s  %.0f%s/s", self.desc, self.done, unit, rate, unit)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, tb):
        if kind is None:
            self.close()


def track(iterable, total, desc, unit="", every=10.0, logger=None):
    """Iterate with progress lines (see Progress)."""
    with Progress(total, desc, unit, every, logger) as p:
        for item in iterable:
            yield item
            p.update()
