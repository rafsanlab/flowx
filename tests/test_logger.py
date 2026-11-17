"""Unit tests for structured logging module (src.io.logger)."""

from pathlib import Path
import logging
import pytest

from flowx.io.logger import (
    DEFAULT_LOGGER_NAME,
    close_run_logger,
    format_duration,
    get_logger,
    log_step,
    setup_run_logger,
)


def test_setup_run_logger(tmp_path: Path):
    """Verify setup_run_logger creates log file, attaches handlers, and records entries."""
    logger, log_file = setup_run_logger(
        run_dir=tmp_path,
        log_filename="test_run.log",
        console_level="INFO",
        file_level="DEBUG",
        logger_name="test_flow_logger",
    )

    assert log_file.is_file()
    assert logger.name == "test_flow_logger"

    logger.debug("Debug diagnostic message")
    logger.info("Informational progress message")

    close_run_logger(logger)

    content = log_file.read_text(encoding="utf-8")
    assert "DEBUG" in content
    assert "Debug diagnostic message" in content
    assert "INFO" in content
    assert "Informational progress message" in content
    assert "test_flow_logger" in content


def test_log_step_timing(tmp_path: Path):
    """Verify log_step measures elapsed time and logs completion markers."""
    logger, log_file = setup_run_logger(
        run_dir=tmp_path,
        log_filename="step_test.log",
        logger_name="test_step_logger",
    )

    with log_step(logger, "Custom Test Step"):
        logger.info("Executing inner task...")

    close_run_logger(logger)

    content = log_file.read_text(encoding="utf-8")
    assert ">>> Starting Custom Test Step..." in content
    assert "<<< Completed Custom Test Step (duration:" in content
    assert "Executing inner task..." in content


def test_log_step_exception_capture(tmp_path: Path):
    """Verify log_step catches, logs exception tracebacks with millisecond precision, and re-raises."""
    logger, log_file = setup_run_logger(
        run_dir=tmp_path,
        log_filename="error_test.log",
        logger_name="test_error_logger",
    )

    with pytest.raises(RuntimeError, match="Simulated fatal error"):
        with log_step(logger, "Failing Step"):
            raise RuntimeError("Simulated fatal error")

    close_run_logger(logger)

    content = log_file.read_text(encoding="utf-8")
    assert ">>> Starting Failing Step..." in content
    assert "!!! Failed Failing Step after" in content
    assert "RuntimeError: Simulated fatal error" in content


def test_format_duration():
    """Verify format_duration converts seconds into clean human-readable duration strings."""
    # Under a minute
    assert format_duration(0.0) == "0.00s"
    assert format_duration(0.423) == "0.42s"
    assert format_duration(14.85) == "14.85s"
    assert format_duration(59.99) == "59.99s"

    # Minutes
    assert format_duration(60.0) == "1m 00.00s"
    assert format_duration(75.25) == "1m 15.25s"
    assert format_duration(125.0) == "2m 05.00s"

    # Hours
    assert format_duration(3600.0) == "1h 00m 00.00s"
    assert format_duration(3665.5) == "1h 01m 05.50s"

    # Negative / boundary
    assert format_duration(-5.0) == "0.00s"
