"""structlog, file only.

No console handler: the user reads rich's output, and mixing log lines into it
would make both unreadable. When a colleague says "it doesn't work", this file is
what they get asked to attach — which is why device, compute type and GPU name
end up in it, being the first thing to check on hardware we cannot inspect.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

import structlog


log = structlog.get_logger()


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(log_dir / "sbobina.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.dev.ConsoleRenderer(colors=False, exception_formatter=structlog.dev.plain_traceback),
        )
    )
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(pid=os.getpid())
