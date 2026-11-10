"""Recoverable JMAP EventSource state event subscriptions.

This module provides a state event subscription which keeps a checkpoint of
the last delivered EventSource event id, reconnects from that checkpoint when
the stream closes or a retryable network error occurs, and applies bounded
exponential backoff between reconnection attempts.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from threading import Event as ThreadingEvent

import requests
import requests.auth
import sseclient

from .logging import log
from .models import Event

__all__ = [
    "RECONNECT_REASON_HTTP_ERROR",
    "RECONNECT_REASON_NETWORK_ERROR",
    "RECONNECT_REASON_STREAM_CLOSED",
    "ReconnectRecord",
    "StateEventSubscription",
    "StateEventsAuthError",
    "StateEventsConfig",
    "StateEventsError",
    "StateEventsInvalidEventError",
    "StateEventsRetryLimitError",
]

RECONNECT_REASON_STREAM_CLOSED = "stream_closed"
RECONNECT_REASON_NETWORK_ERROR = "network_error"
RECONNECT_REASON_HTTP_ERROR = "http_error"

_HTTP_AUTH = "auth"
_HTTP_RETRYABLE = "retryable"
_HTTP_FATAL = "fatal"

_STOP_POLL_INTERVAL = 0.1

RequestsAuth = (
    requests.auth.AuthBase
    | tuple[str, str]
    | Callable[[requests.PreparedRequest], requests.PreparedRequest]
)


class StateEventsError(RuntimeError):
    """Base class for state event subscription failures."""


class StateEventsAuthError(StateEventsError):
    """A non-retryable authentication or authorization failure."""

    def __init__(self, message: str, *, status_code: int | None) -> None:
        super().__init__(message)
        self.status_code = status_code


class StateEventsInvalidEventError(StateEventsError):
    """A delivered state event could not be parsed."""


class StateEventsRetryLimitError(StateEventsError):
    """Reconnection attempts were exhausted without recovery."""

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_event_id: str | None,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.last_event_id = last_event_id
        self.last_error = error


@dataclass(frozen=True)
class ReconnectRecord:
    """Observability record for a single reconnection attempt."""

    attempt: int
    last_event_id: str | None
    delay: float
    reason: str
    error: BaseException | None = None


@dataclass
class StateEventsConfig:
    """Configuration for a recoverable state event subscription.

    :param types: EventSource ``types`` URL parameter.
    :param ping: EventSource ``ping`` interval in seconds.
    :param max_retries: Maximum number of reconnection attempts.
    :param initial_delay: Initial exponential backoff delay in seconds.
    :param max_delay: Upper bound for the backoff delay in seconds.
    :param stop_event: Signal used to stop the subscription.
    :param on_reconnect: Callback invoked with each reconnection record.
    :param last_event_id: Initial EventSource checkpoint.
    """

    types: str = "*"
    ping: int = 0
    max_retries: int = 5
    initial_delay: float = 1.0
    max_delay: float = 30.0
    stop_event: ThreadingEvent | None = None
    on_reconnect: Callable[[ReconnectRecord], None] | None = None
    last_event_id: str | None = None

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be greater than or equal to 0")
        if self.initial_delay <= 0:
            raise ValueError("initial_delay must be greater than 0")
        if self.max_delay <= 0:
            raise ValueError("max_delay must be greater than 0")
        if self.max_delay < self.initial_delay:
            raise ValueError(
                "max_delay must be greater than or equal to initial_delay"
            )
        if self.ping < 0:
            raise ValueError("ping must be greater than or equal to 0")


class StateEventSubscription:
    """Iterable recoverable subscription delivering JMAP state events."""

    def __init__(
        self,
        url: str,
        auth: RequestsAuth | None,
        config: StateEventsConfig,
        last_event_id: str | None = None,
    ) -> None:
        self._url = url
        self._auth = auth
        self._config = config
        self._checkpoint = last_event_id
        self._attempt = 0
        self.reconnects: list[ReconnectRecord] = []

    @property
    def checkpoint(self) -> str | None:
        """Last saved EventSource event id."""
        return self._checkpoint

    def __iter__(self) -> Iterator[Event]:
        return self.events()

    def events(self) -> Generator[Event, None, None]:
        """Yield state events, reconnecting from the last valid event id."""
        seen_ids: set[str] = set()
        while True:
            if self._is_stopped():
                return
            try:
                sse = sseclient.SSEClient(
                    self._url,
                    auth=self._auth,
                    last_id=self._checkpoint,
                )
            except requests.HTTPError as exc:
                self._handle_http_error(exc)
                continue
            except requests.RequestException as exc:
                if not self._handle_request_exception(exc):
                    return
                continue
            stream_closed = yield from self._iter_connection(sse, seen_ids)
            if stream_closed and (
                self._is_stopped()
                or not self._schedule_reconnect(
                    RECONNECT_REASON_STREAM_CLOSED, None
                )
            ):
                return

    def _iter_connection(
        self, sse: sseclient.SSEClient, seen_ids: set[str]
    ) -> Generator[Event, None, bool]:
        try:
            for raw_event in sse:
                event = self._parse_event(raw_event)
                if event is not None:
                    if event.id is not None and event.id in seen_ids:
                        log.debug(
                            "Skipping duplicate state event id %s",
                            event.id,
                        )
                    else:
                        if event.id is not None:
                            seen_ids.add(event.id)
                            self._checkpoint = event.id
                        yield event
                if self._is_stopped():
                    return False
        except requests.HTTPError as exc:
            self._handle_http_error(exc)
            return False
        except requests.RequestException as exc:
            # Raises on retry exhaustion; otherwise reconnect is scheduled
            self._handle_request_exception(exc)
            return False
        else:
            return True

    def _parse_event(self, raw_event: sseclient.Event | None) -> Event | None:
        if raw_event is None or raw_event.event != "state":
            return None
        if not raw_event.data:
            log.debug("Ignoring empty state event id=%r", raw_event.id)
            return None
        try:
            return Event.load_from_sseclient_event(raw_event)
        except (ValueError, KeyError, TypeError) as exc:
            raise StateEventsInvalidEventError(
                f"Invalid state event with id {raw_event.id!r}: {exc}"
            ) from exc

    def _handle_request_exception(
        self, exc: requests.RequestException
    ) -> bool:
        if self._is_stopped():
            return False
        if self._schedule_reconnect(RECONNECT_REASON_NETWORK_ERROR, exc):
            return True
        if self._is_stopped():
            return False
        raise self._retry_limit_error(exc) from exc

    def _handle_http_error(self, exc: requests.HTTPError) -> None:
        kind = self._classify_http_error(exc)
        if kind == _HTTP_AUTH:
            status = (
                exc.response.status_code if exc.response is not None else None
            )
            raise StateEventsAuthError(
                f"EventSource authentication failed (HTTP {status})",
                status_code=status,
            ) from exc
        if kind == _HTTP_FATAL:
            raise StateEventsError(
                "EventSource request failed with a non-retryable HTTP error"
            ) from exc
        if self._is_stopped() or not self._schedule_reconnect(
            RECONNECT_REASON_HTTP_ERROR, exc
        ):
            if self._is_stopped():
                return
            raise self._retry_limit_error(exc) from exc

    @staticmethod
    def _classify_http_error(exc: requests.HTTPError) -> str:
        # Response.__bool__ returns ``ok``, so error responses are falsy:
        # compare against None explicitly.
        response = exc.response
        status = response.status_code if response is not None else None
        if status in (401, 403):
            return _HTTP_AUTH
        if status is None or status == 408 or status == 429 or status >= 500:
            return _HTTP_RETRYABLE
        return _HTTP_FATAL

    def _schedule_reconnect(
        self, reason: str, error: BaseException | None
    ) -> bool:
        if self._is_stopped():
            return False
        self._attempt += 1
        if self._attempt > self._config.max_retries:
            log.warning(
                "State event subscription retries exhausted after %s "
                "reconnection attempt(s)",
                self._attempt - 1,
            )
            return False
        delay = min(
            self._config.initial_delay * (2 ** (self._attempt - 1)),
            self._config.max_delay,
        )
        record = ReconnectRecord(
            attempt=self._attempt,
            last_event_id=self._checkpoint,
            delay=delay,
            reason=reason,
            error=error,
        )
        self.reconnects.append(record)
        log.info(
            "Reconnecting state event subscription: attempt=%s reason=%s "
            "last_event_id=%s delay=%.2fs",
            record.attempt,
            record.reason,
            record.last_event_id,
            record.delay,
        )
        if self._config.on_reconnect is not None:
            self._config.on_reconnect(record)
        self._sleep(delay)
        return not self._is_stopped()

    def _retry_limit_error(
        self, error: BaseException | None
    ) -> StateEventsRetryLimitError:
        return StateEventsRetryLimitError(
            f"State event subscription gave up after {self._attempt - 1} "
            "reconnection attempt(s)",
            attempts=self._attempt - 1,
            last_event_id=self._checkpoint,
            error=error,
        )

    def _sleep(self, delay: float) -> None:
        stop_event = self._config.stop_event
        if stop_event is None:
            time.sleep(delay)
            return
        deadline = time.monotonic() + delay
        while not stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(remaining, _STOP_POLL_INTERVAL))

    def _is_stopped(self) -> bool:
        stop_event = self._config.stop_event
        return stop_event is not None and stop_event.is_set()
