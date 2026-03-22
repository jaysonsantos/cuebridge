from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CancellationToken:
    """Cooperative cancellation primitive shared across translation calls."""

    cancelled: bool = False
    reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        self.cancelled = True
        if reason is not None:
            self.reason = reason
