"""
Project progress logger: distinct from system/library logs.
All messages use the [CALL_ANALYSIS] prefix so you can see pipeline progress and filter easily.
"""

import logging
import sys

_PROJECT_LOGGER_NAME = "call_analysis"
_SETUP_DONE = False


def setup_project_logging(level=logging.INFO):
    """
    Configure the call_analysis logger so all pipeline and preprocessing logs
    use the [CALL_ANALYSIS] prefix and go to stdout. Call this at the start of
    run_full_analysis, run_diarization_only, or transcribe_audio.
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    _SETUP_DONE = True

    log = logging.getLogger(_PROJECT_LOGGER_NAME)
    log.setLevel(level)
    log.propagate = False  # Only our handler; avoid duplicate output from root

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("[CALL_ANALYSIS] %(levelname)s | %(message)s")
    )
    log.addHandler(handler)


def get_progress_logger():
    """Return the project logger for progress messages (use after setup_project_logging)."""
    return logging.getLogger(_PROJECT_LOGGER_NAME)
