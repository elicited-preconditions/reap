# Adapted from the plangolin code base (Aichmüller et al.).
# MIT License, Copyright (c) 2025 Michael Aichmüller; see LICENSES/plangolin-MIT.txt.

import datetime
import logging
import logging.handlers
import re
from functools import cache
from io import StringIO

try:
    from colorama import Fore, Style, init
except ImportError:  # pragma: no cover - fallback when colorama unavailable

    class _NoColor:
        def __getattr__(self, item):
            return ""

    Fore = _NoColor()
    Fore.GREEN = Fore.BLUE = Fore.YELLOW = Fore.RED = Fore.MAGENTA = ""
    Style = _NoColor()
    Style.RESET_ALL = ""

    def init(**kwargs):
        return None


from tqdm.auto import tqdm as tqdm_

# init colorama for cross-platform color support
init(autoreset=True)

# Matches ANSI CSI (e.g., "\x1b[K", "\x1b[?25l") and OSC ("\x1b]...\x07") sequences
ANSI_ESCAPE_RE = re.compile(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07]*\x07)")

LOG_COLORS = {
    "INFO": Fore.GREEN,
    "DEBUG": Fore.BLUE,
    "WARNING": Fore.YELLOW,
    "ERROR": Fore.RED,
    "CRITICAL": Fore.MAGENTA,
}

null_logger = logging.getLogger("null_logger")
# make absolutely sure nothing ever propagates out
null_logger.propagate = False
# attach a NullHandler so that it never errors for “no handlers”
null_logger.addHandler(logging.NullHandler())


class SPDLOGFormatter(logging.Formatter):
    """
    Custom log formatter with color support and custom log to match SPDLOG's format
    """

    spdlog_format_string = (
        "[%(asctime)s] [%(name)s:%(lineno)d] [%(levelname)s] %(message)s"
    )

    def __init__(self):
        super().__init__(fmt=self.spdlog_format_string, datefmt="%Y-%m-%d %H:%M:%S")

    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created)
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def format(self, record):
        log_color = LOG_COLORS.get(record.levelname, "")
        record.levelname = f"{log_color}{record.levelname.lower()}{Style.RESET_ALL}"
        return super().format(record)


class TqdmLogFormatter(SPDLOGFormatter):
    def format(self, record):
        # Get the original message.
        original_msg = record.getMessage()
        cr_prefix = ""
        if original_msg.startswith("\r"):
            # If there are leading '\r' characters, preserve them.
            # (might want to preserve just one or all; here we capture all leading '\r's.)
            num_cr = len(original_msg) - len(original_msg.lstrip("\r"))
            cr_prefix = "\r" * num_cr
            # Remove the leading carriage returns from the message attribute for formatting.
            record.msg = original_msg.lstrip("\r")

        # Format the message as usual.
        formatted = super().format(record)
        # Prepend the captured carriage return(s) to the final formatted string.
        if cr_prefix:
            formatted = cr_prefix + formatted

        return formatted


class TqdmLogGuard:
    def __init__(self, logger):
        self._logger = logger

    def __enter__(self):
        self._handlers = []
        for handler in self._logger.handlers:
            state = (
                handler,
                handler.formatter,
                getattr(handler, "terminator", "\n"),
                handler.emit,
            )
            self._handlers.append(state)
            handler.setFormatter(TqdmLogFormatter())

            def _patched_emit(
                record,
                *,
                _handler=handler,
                _orig_terminator=state[2],
            ):
                progress_update = getattr(record, "progress_update", False)
                progress_complete = getattr(record, "progress_complete", False)
                try:
                    if (
                        progress_update
                        and hasattr(_handler.stream, "isatty")
                        and _handler.stream.isatty()
                    ):
                        msg = _handler.format(record)
                        if not msg.startswith("\r"):
                            msg = "\r" + msg
                        msg = msg.rstrip("\n")
                        _handler.stream.write(msg + "\x1b[K")
                        _handler.flush()
                        if progress_complete:
                            _handler.stream.write("\n")
                            _handler.flush()
                    else:
                        msg = _handler.format(record)
                        _handler.stream.write(msg + _orig_terminator)
                        _handler.flush()
                except Exception:
                    _handler.handleError(record)

            handler.emit = _patched_emit
        return self._logger

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for handler, formatter, terminator, original_emit in self._handlers:
            handler.setFormatter(formatter)
            handler.terminator = terminator
            handler.emit = original_emit
        self._handlers.clear()


class TqdmLogger(StringIO):
    def __init__(self, logger):
        super().__init__()
        self._logger = logger
        self._last_progress: str | None = None

    def write(self, buffer):
        if buffer is None:
            return
        clean = ANSI_ESCAPE_RE.sub("", buffer)

        if "%|" in clean:
            progress_text = clean.lstrip("\r").rstrip("\n")
            if not progress_text.strip():
                return
            if progress_text != self._last_progress:
                self._last_progress = progress_text
                with TqdmLogGuard(self._logger) as logger:
                    logger.info(
                        progress_text,
                        extra={
                            "progress_update": True,
                            "progress_complete": "100%" in progress_text,
                        },
                    )
            return

        if clean in ("\n", "\r\n"):
            return

        stripped = clean.rstrip("\n")
        if not stripped or not stripped.strip():
            return
        with TqdmLogGuard(self._logger) as logger:
            logger.info(stripped)

    def flush(self):
        # tqdm flushes aggressively; progress entries are logged when they arrive,
        # so there's nothing left to flush explicitly.
        return


def setup_logger(name="root"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Don't let this logger bubble messages up to parents
    logger.propagate = False

    # Only add a StreamHandler once
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(SPDLOGFormatter())
        logger.addHandler(console_handler)

    return logger


@cache
def get_logger(name):
    logger = logging.getLogger(name)
    setup_logger(name)
    return logger


def tqdm(
    x=None, logger: logging.Logger | None = None, ascii_: bool = False, **log_kwargs
):
    logger = logger or get_logger("tqdm")
    return tqdm_(
        x,
        file=TqdmLogger(logger),
        ascii=ascii_,
        **log_kwargs,
    )
