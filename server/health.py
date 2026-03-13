from dataclasses import dataclass
from threading import Lock


@dataclass
class HealthSnapshot:
    live: bool
    ready: bool
    message: str


class HealthState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._live = True
        self._ready = False
        self._message = "starting"

    def set(self, *, live: bool = None, ready: bool = None, message: str = None) -> None:
        with self._lock:
            if live is not None:
                self._live = live
            if ready is not None:
                self._ready = ready
            if message is not None:
                self._message = message

    def snapshot(self) -> HealthSnapshot:
        with self._lock:
            return HealthSnapshot(self._live, self._ready, self._message)
