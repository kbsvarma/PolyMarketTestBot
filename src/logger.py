from __future__ import annotations

import logging
from pathlib import Path

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    class _FallbackLogger:
        def __init__(self) -> None:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
            self._logger = logging.getLogger("polymarket_bot")

        def remove(self) -> None:
            return None

        def add(self, sink, **_: object) -> None:
            if callable(sink):
                return None
            handler = logging.FileHandler(str(sink))
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self._logger.addHandler(handler)

        def warning(self, msg: str, *args: object) -> None:
            self._logger.warning(msg.format(*args))

        def info(self, msg: str, *args: object) -> None:
            self._logger.info(msg.format(*args))

    logger = _FallbackLogger()


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(log_dir / "system.log", rotation="5 MB", retention=5, serialize=True)
    logger.add(log_dir / "errors.log", rotation="5 MB", retention=5, level="ERROR", serialize=True)
    logger.add(lambda message: print(message, end=""), level="INFO")
