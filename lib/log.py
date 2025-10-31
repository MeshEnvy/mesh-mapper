"""Logging system for mesh mapper."""

import sys

_debug_enabled = False
_silent_enabled = False


def set_logging(debug: bool = False, silent: bool = False):
    """Set global logging flags."""
    global _debug_enabled, _silent_enabled
    _debug_enabled = debug
    _silent_enabled = silent


def log_info(message: str):
    """Log an INFO message."""
    if not _silent_enabled:
        print(f'[INFO] {message}', file=sys.stdout)


def log_debug(message: str):
    """Log a DEBUG message."""
    if not _silent_enabled and _debug_enabled:
        print(f'[DEBUG] {message}', file=sys.stdout)


def log_error(message: str):
    """Log an ERROR message."""
    if not _silent_enabled:
        print(f'[ERROR] {message}', file=sys.stderr)


def log_warn(message: str):
    """Log a WARN message."""
    if not _silent_enabled:
        print(f'[WARN] {message}', file=sys.stdout)

