"""Structured logging system for flow cytometry pipeline execution and debugging.

Provides dual-destination logging:
- Detailed debug trace persisted to 'pipeline.log' in the run output directory.
- Informative status updates streamed to console/stdout.
- Step timing, diagnostic metrics, and automated exception trace capture for easy debugging.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

DEFAULT_LOGGER_NAME = "flow_cytometry"

# When set, overrides the console handler level in every ``setup_run_logger`` call.
# Used by the terminal progress UI to silence line-by-line console output while the
# full DEBUG trace keeps streaming to the run's ``pipeline.log`` file.
_CONSOLE_LEVEL_OVERRIDE: Optional[int] = None


def set_console_level_override(level: Optional[int]) -> Optional[int]:
    """Force the console handler level for all subsequent/active run loggers.

    Args:
        level: Logging level to force (e.g. ``logging.CRITICAL + 10`` to silence the
            console), or ``None`` to remove the override.

    Returns:
        The previous override value so callers can restore it.
    """
    global _CONSOLE_LEVEL_OVERRIDE
    previous = _CONSOLE_LEVEL_OVERRIDE
    _CONSOLE_LEVEL_OVERRIDE = level
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            if level is not None:
                h.setLevel(level)
    return previous


def format_duration(seconds: float) -> str:
    """Format duration in seconds into a clean, human-readable string.

    Args:
        seconds: Elapsed duration in seconds.

    Returns:
        Human-readable string representation (e.g., '0.45s', '14.85s', '1m 24.35s', '2h 05m 12.00s').
    """
    if seconds < 0:
        return "0.00s"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    if minutes < 60:
        return f"{minutes}m {rem_seconds:05.2f}s"
    hours = int(minutes // 60)
    rem_minutes = int(minutes % 60)
    return f"{hours}h {rem_minutes:02d}m {rem_seconds:05.2f}s"


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Retrieve or create the flow cytometry pipeline logger."""
    return logging.getLogger(name)


def setup_run_logger(
    run_dir: Union[str, Path],
    log_filename: str = "pipeline.log",
    console_level: Union[str, int] = "INFO",
    file_level: Union[str, int] = "DEBUG",
    logger_name: str = DEFAULT_LOGGER_NAME,
    overwrite: bool = True,
) -> Tuple[logging.Logger, Path]:
    """Configure logger to stream to console and write detailed debug logs to run output folder.

    Args:
        run_dir: Directory where the log file will be stored.
        log_filename: Filename for the log file (default: 'pipeline.log').
        console_level: Log level for console output (default: 'INFO').
        file_level: Log level for file output (default: 'DEBUG').
        logger_name: Name of the logger (default: 'flow_cytometry').
        overwrite: Whether to overwrite an existing log file at the target path (default: True).
            When False and a FileHandler is already attached to the same file, the existing
            handler is preserved without truncating earlier pipeline step logs.

    Returns:
        Tuple of (configured logging.Logger instance, resolved Path to log file).
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    log_file_path = (run_path / log_filename).resolve()

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Capture all events; individual handlers filter levels

    # 1. File Handler: Comprehensive debug format with millisecond timestamps and caller info
    file_fmt = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d | %(levelname)-7s | [%(name)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if isinstance(file_level, str):
        resolved_file_level = getattr(logging, file_level.upper(), logging.DEBUG)
    else:
        resolved_file_level = file_level

    # Clean up any existing FileHandlers to prevent duplicate logs across runs,
    # unless overwrite=False and an active FileHandler already targets log_file_path.
    reused_existing_file_handler = False
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            existing_path = Path(getattr(h, "baseFilename", "")).resolve() if getattr(h, "baseFilename", None) else None
            if not overwrite and existing_path == log_file_path:
                h.setLevel(resolved_file_level)
                h.setFormatter(file_fmt)
                reused_existing_file_handler = True
                continue
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)

    if not reused_existing_file_handler:
        file_mode = "w" if overwrite else "a"
        file_handler = logging.FileHandler(str(log_file_path), mode=file_mode, encoding="utf-8")
        file_handler.setLevel(resolved_file_level)
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    # 2. Console Handler: Clean format with timestamps
    console_fmt = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if isinstance(console_level, str):
        resolved_console_level = getattr(logging, console_level.upper(), logging.INFO)
    else:
        resolved_console_level = console_level
    if _CONSOLE_LEVEL_OVERRIDE is not None:
        resolved_console_level = _CONSOLE_LEVEL_OVERRIDE

    # Check if a non-file StreamHandler is already present
    has_stream_handler = False
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            has_stream_handler = True
            h.setLevel(resolved_console_level)
            h.setFormatter(console_fmt)
            break

    if not has_stream_handler:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(resolved_console_level)
        stream_handler.setFormatter(console_fmt)
        logger.addHandler(stream_handler)

    logger.propagate = False
    return logger, log_file_path


def close_run_logger(logger: logging.Logger) -> None:
    """Flush and close all FileHandlers on the logger to release resources."""
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                h.flush()
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)


@contextmanager
def log_step(logger: logging.Logger, step_title: str) -> Generator[None, None, None]:
    """Context manager to measure and log execution duration and errors for a pipeline step."""
    logger.info(f">>> Starting {step_title}...")
    t0 = time.perf_counter()
    try:
        yield
        elapsed = time.perf_counter() - t0
        logger.info(f"<<< Completed {step_title} (duration: {elapsed:.2f}s)")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.error(f"!!! Failed {step_title} after {elapsed:.2f}s: {exc}", exc_info=True)
        raise
