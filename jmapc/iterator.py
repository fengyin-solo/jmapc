from __future__ import annotations

import dataclasses
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

from .methods.base import Query, QueryResponse

if TYPE_CHECKING:
    from .client import Client

__all__ = [
    "DEFAULT_QUERY_PAGE_SIZE",
    "QueryDuplicateIdError",
    "QueryIterator",
    "QueryIteratorError",
    "QueryIteratorItem",
    "QueryStalledPositionError",
    "QueryStateChangedError",
]

DEFAULT_QUERY_PAGE_SIZE = 256


class QueryIteratorError(RuntimeError):
    """Base class for errors raised while iterating query result pages."""


class QueryStateChangedError(QueryIteratorError):
    """Raised when the server reports a different query state mid-iteration."""

    def __init__(
        self,
        *,
        page: int,
        previous_state: str,
        new_state: str,
    ) -> None:
        self.page = page
        self.previous_state = previous_state
        self.new_state = new_state
        super().__init__(
            f"Query state changed on page {page}: "
            f"{previous_state!r} -> {new_state!r}"
        )


class QueryStalledPositionError(QueryIteratorError):
    """Raised when a query response does not advance the result position."""

    def __init__(
        self,
        *,
        page: int,
        requested_position: int,
        response_position: int,
    ) -> None:
        self.page = page
        self.requested_position = requested_position
        self.response_position = response_position
        super().__init__(
            f"Query position stalled on page {page}: requested "
            f"{requested_position}, server returned {response_position}"
        )


class QueryDuplicateIdError(QueryIteratorError):
    """Raised when the server returns an id that was already yielded."""

    def __init__(
        self,
        *,
        page: int,
        position: int,
        result_id: str,
    ) -> None:
        self.page = page
        self.position = position
        self.result_id = result_id
        super().__init__(
            f"Duplicate id {result_id!r} returned on page {page} at "
            f"position {position}"
        )


@dataclass(frozen=True)
class QueryIteratorItem:
    """A single query result id with its pagination context."""

    page: int
    position: int
    id: str
    response: QueryResponse


class QueryIterator:
    """Lazily paginate any JMAP ``query`` method in original sort order.

    The supplied query method is never modified. Each page is requested
    using a copy, so the original limit, position, sort, filter and collapse
    settings are retained on the caller's instance.

    Iteration ends safely when the server returns an empty page, a short
    page, all results known via ``total``, or the query's own limit is
    reached. Protocol anomalies raise a distinct ``QueryIteratorError``
    subclass.
    """

    def __init__(
        self,
        client: Client,
        method: Query,
        page_size: int | None = None,
    ) -> None:
        if not isinstance(method, Query):
            raise TypeError(
                "Query pagination requires a Query method instance, got "
                f"{type(method).__name__}"
            )
        self._client = client
        self._method = method
        self._limit_cap = method.limit
        if page_size is None:
            page_size = method.limit or DEFAULT_QUERY_PAGE_SIZE
        if page_size <= 0:
            raise ValueError(
                f"page_size must be a positive integer, got {page_size}"
            )
        if self._limit_cap is not None and self._limit_cap <= 0:
            raise ValueError(
                "Query limit must be a positive integer, got "
                f"{self._limit_cap}"
            )
        self._page_size = page_size
        self._next_position = method.position or 0
        self._yielded = 0
        self._seen_ids: set[str] = set()
        self._buffer: deque[QueryIteratorItem] = deque()
        self._responses: list[QueryResponse] = []
        self._exhausted = False
        self._query_state: str | None = None
        self._total: int | None = None

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> QueryIteratorItem:
        if not self._buffer and not self._exhausted:
            self._fetch_page()
        if not self._buffer:
            raise StopIteration
        return self._buffer.popleft()

    @property
    def page(self) -> int:
        """Number of query pages fetched so far."""
        return len(self._responses)

    @property
    def position(self) -> int:
        """Absolute position of the next result to request."""
        return self._next_position

    @property
    def query_state(self) -> str | None:
        """Query state from the most recently fetched page."""
        return self._query_state

    @property
    def total(self) -> int | None:
        """Total result count reported by the server, if known."""
        return self._total

    @property
    def responses(self) -> tuple[QueryResponse, ...]:
        """All query responses fetched so far, in request order."""
        return tuple(self._responses)

    def _fetch_page(self) -> None:
        page = len(self._responses)
        request_limit = self._page_size
        if self._limit_cap is not None:
            request_limit = min(request_limit, self._limit_cap - self._yielded)
        changes: dict[str, Any] = {"limit": request_limit}
        if page > 0:
            # Subsequent pages use absolute positions. Any anchor from the
            # original query only applies to the first request.
            changes["position"] = self._next_position
            changes["anchor"] = None
            changes["anchor_offset"] = None
        page_query = dataclasses.replace(self._method, **changes)
        response = cast(
            "QueryResponse",
            self._client.request(page_query, raise_errors=True),
        )
        self._handle_response(response, page=page, request_limit=request_limit)

    def _handle_response(
        self,
        response: QueryResponse,
        *,
        page: int,
        request_limit: int,
    ) -> None:
        result_ids = response.ids
        if not isinstance(result_ids, list):
            raise TypeError(
                "Query response ids must be a list, got "
                f"{type(result_ids).__name__}"
            )
        if (
            self._query_state is not None
            and response.query_state != self._query_state
        ):
            raise QueryStateChangedError(
                page=page,
                previous_state=self._query_state,
                new_state=response.query_state,
            )
        self._query_state = response.query_state
        if response.total is not None:
            self._total = response.total
        self._responses.append(response)
        if not result_ids:
            # An empty page means there are no further results.
            self._exhausted = True
            return
        page_end = response.position + len(result_ids)
        if page_end <= self._next_position:
            raise QueryStalledPositionError(
                page=page,
                requested_position=self._next_position,
                response_position=response.position,
            )
        self._buffer_ids(result_ids, response=response, page=page)
        self._next_position = page_end
        self._yielded += len(result_ids)
        total_reached = self._total is not None and page_end >= self._total
        cap_reached = (
            self._limit_cap is not None and self._yielded >= self._limit_cap
        )
        if len(result_ids) < request_limit or total_reached or cap_reached:
            # A short page, a reached total, or a reached query limit
            # guarantees no further results need to be requested.
            self._exhausted = True

    def _buffer_ids(
        self,
        result_ids: list[str],
        *,
        response: QueryResponse,
        page: int,
    ) -> None:
        for offset, result_id in enumerate(result_ids):
            if result_id in self._seen_ids:
                raise QueryDuplicateIdError(
                    page=page,
                    position=response.position + offset,
                    result_id=result_id,
                )
            self._seen_ids.add(result_id)
            self._buffer.append(
                QueryIteratorItem(
                    page=page,
                    position=response.position + offset,
                    id=result_id,
                    response=response,
                )
            )
