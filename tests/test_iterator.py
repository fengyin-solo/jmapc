from __future__ import annotations

import json
from collections import deque
from typing import TYPE_CHECKING, Any

import pytest
import responses

from jmapc import (
    Client,
    ClientError,
    Comparator,
    EmailQueryFilterCondition,
    QueryDuplicateIdError,
    QueryIterator,
    QueryIteratorError,
    QueryIteratorItem,
    QueryStalledPositionError,
    QueryStateChangedError,
)
from jmapc.constants import (
    JMAP_URN_CORE,
    JMAP_URN_MAIL,
    JMAP_URN_SUBMISSION,
)
from jmapc.methods import (
    CoreEcho,
    EmailQuery,
    EmailQueryResponse,
    EmailSubmissionQuery,
    EmailSubmissionQueryResponse,
    MailboxQuery,
    MailboxQueryResponse,
)
from jmapc.methods.base import Query

if TYPE_CHECKING:
    import requests


class _TotalUnset:
    pass


_TOTAL_UNSET = _TotalUnset()


def queue_jmap_calls(
    http_responses: responses.RequestsMock,
    calls: list[tuple[dict[str, Any], dict[str, Any]]],
) -> deque[tuple[dict[str, Any], dict[str, Any]]]:
    """Serve queued JMAP request/response pairs, asserting request bodies."""
    queue = deque(calls)

    def _callback(
        request: requests.PreparedRequest,
    ) -> tuple[int, dict[str, str], str]:
        try:
            expected_request, response = queue.popleft()
        except IndexError:  # pragma: no cover - asserted by tests
            raise AssertionError("Unexpected extra JMAP request") from None
        assert request.headers["Content-Type"] == "application/json"
        assert json.loads(str(request.body) or "{}") == expected_request
        response.setdefault("sessionState", "test;session;state")
        return (200, {}, json.dumps(response))

    http_responses.add_callback(
        method=responses.POST,
        url="https://jmap-api.localhost/api",
        callback=_callback,
    )
    return queue


NAMESPACE_URNS = {
    "Email": {JMAP_URN_MAIL},
    "Mailbox": {JMAP_URN_MAIL},
    "EmailSubmission": {JMAP_URN_SUBMISSION},
}


def expected_query_request(
    namespace: str,
    *,
    position: int | None = None,
    limit: int | None = None,
    arguments: dict[str, Any] | None = None,
    urns: set[str] | None = None,
) -> dict[str, Any]:
    method_arguments: dict[str, Any] = {"accountId": "u1138"}
    if position is not None:
        method_arguments["position"] = position
    if limit is not None:
        method_arguments["limit"] = limit
    if arguments:
        method_arguments.update(arguments)
    expected_urns = urns or NAMESPACE_URNS.get(namespace, set())
    return {
        "methodCalls": [
            [
                f"{namespace}/query",
                method_arguments,
                f"single.{namespace}/query",
            ]
        ],
        "using": sorted({JMAP_URN_CORE, *expected_urns}),
    }


def query_response(
    namespace: str,
    ids: list[str],
    *,
    position: int = 0,
    query_state: str = "query:state:1",
    total: int | _TotalUnset | None = _TOTAL_UNSET,
    limit: int | None = None,
) -> dict[str, Any]:
    method_response: dict[str, Any] = {
        "accountId": "u1138",
        "queryState": query_state,
        "canCalculateChanges": True,
        "position": position,
        "ids": ids,
    }
    if not isinstance(total, _TotalUnset):
        method_response["total"] = total
    if limit is not None:
        method_response["limit"] = limit
    return {
        "methodResponses": [
            [
                f"{namespace}/query",
                method_response,
                f"single.{namespace}/query",
            ]
        ]
    }


def test_query_iterator_with_total(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    calls = [
        (
            expected_query_request("Email", limit=2),
            query_response("Email", ["M1", "M2"], total=5),
        ),
        (
            expected_query_request("Email", position=2, limit=2),
            query_response("Email", ["M3", "M4"], position=2, total=5),
        ),
        (
            expected_query_request("Email", position=4, limit=2),
            query_response("Email", ["M5"], position=4, total=5),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    iterator = client.query(EmailQuery(), page_size=2)
    assert isinstance(iterator, QueryIterator)
    items = list(iterator)
    assert [(i.page, i.position, i.id) for i in items] == [
        (0, 0, "M1"),
        (0, 1, "M2"),
        (1, 2, "M3"),
        (1, 3, "M4"),
        (2, 4, "M5"),
    ]
    assert all(isinstance(i.response, EmailQueryResponse) for i in items)
    assert all(isinstance(i, QueryIteratorItem) for i in items)
    assert iterator.total == 5
    assert iterator.query_state == "query:state:1"
    assert iterator.position == 5
    assert iterator.page == 3
    assert len(iterator.responses) == 3
    assert not queue


def test_query_iterator_without_total(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    calls = [
        (
            expected_query_request("Email", limit=2),
            query_response("Email", ["M1", "M2"]),
        ),
        (
            expected_query_request("Email", position=2, limit=2),
            query_response("Email", ["M3", "M4"], position=2),
        ),
        (
            expected_query_request("Email", position=4, limit=2),
            query_response("Email", [], position=4),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    iterator = client.query(EmailQuery(), page_size=2)
    items = list(iterator)
    assert [i.id for i in items] == ["M1", "M2", "M3", "M4"]
    assert [i.position for i in items] == [0, 1, 2, 3]
    assert iterator.total is None
    assert iterator.query_state == "query:state:1"
    assert iterator.position == 4
    assert iterator.page == 3
    assert not queue


def test_query_iterator_exact_full_page_with_total(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    calls = [
        (
            expected_query_request("Email", limit=2),
            query_response("Email", ["M1", "M2"], total=4),
        ),
        (
            expected_query_request("Email", position=2, limit=2),
            query_response("Email", ["M3", "M4"], position=2, total=4),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    iterator = client.query(EmailQuery(), page_size=2)
    assert [i.id for i in iterator] == ["M1", "M2", "M3", "M4"]
    # Total reached: no probe request for another page
    assert not queue
    assert iterator.total == 4
    assert iterator.page == 2
    assert iterator.position == 4


def test_query_iterator_tail_page(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    calls = [
        (
            expected_query_request("Email", limit=3),
            query_response("Email", ["M1", "M2", "M3"], total=7),
        ),
        (
            expected_query_request("Email", position=3, limit=3),
            query_response("Email", ["M4", "M5", "M6"], position=3, total=7),
        ),
        (
            expected_query_request("Email", position=6, limit=3),
            query_response("Email", ["M7"], position=6, total=7),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    iterator = client.query(EmailQuery(), page_size=3)
    items = list(iterator)
    assert [i.id for i in items] == [f"M{index}" for index in range(1, 8)]
    assert [i.page for i in items] == [0, 0, 0, 1, 1, 1, 2]
    assert [i.position for i in items] == list(range(7))
    assert not queue
    assert iterator.page == 3
    assert iterator.position == 7


@pytest.mark.parametrize("total", [0, None])
def test_query_iterator_empty_result(
    client: Client,
    http_responses: responses.RequestsMock,
    *,
    total: int | None,
) -> None:
    calls = [
        (
            expected_query_request("Email", limit=2),
            query_response("Email", [], total=total),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    iterator = client.query(EmailQuery(), page_size=2)
    assert list(iterator) == []
    assert iterator.query_state == "query:state:1"
    assert iterator.total == total
    assert iterator.position == 0
    assert iterator.page == 1
    assert not queue


def test_query_iterator_custom_limit_and_preserved_query(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    query = EmailQuery(
        limit=5,
        sort=[Comparator(property="receivedAt", is_ascending=False)],
        filter=EmailQueryFilterCondition(in_mailbox="MBX1"),
        collapse_threads=True,
        calculate_total=True,
    )
    extra_arguments = {
        "collapseThreads": True,
        "calculateTotal": True,
        "filter": {"inMailbox": "MBX1"},
        "sort": [
            {
                "anchorOffset": 0,
                "calculateTotal": False,
                "isAscending": False,
                "position": 0,
                "property": "receivedAt",
            }
        ],
    }
    calls = [
        (
            expected_query_request(
                "Email", limit=2, arguments=extra_arguments
            ),
            query_response("Email", ["M1", "M2"], total=5),
        ),
        (
            expected_query_request(
                "Email",
                position=2,
                limit=2,
                arguments=extra_arguments,
            ),
            query_response("Email", ["M3", "M4"], position=2, total=5),
        ),
        (
            expected_query_request(
                "Email",
                position=4,
                limit=1,
                arguments=extra_arguments,
            ),
            query_response("Email", ["M5"], position=4, total=5),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    iterator = client.query(query, page_size=2)
    assert [i.id for i in iterator] == [
        "M1",
        "M2",
        "M3",
        "M4",
        "M5",
    ]
    assert not queue
    # The caller's query object must remain unmodified
    assert query.limit == 5
    assert query.position is None
    assert query.anchor is None
    assert query.anchor_offset is None
    assert query.collapse_threads is True
    assert query.calculate_total is True
    assert query.filter == EmailQueryFilterCondition(in_mailbox="MBX1")
    assert query.sort == [
        Comparator(property="receivedAt", is_ascending=False)
    ]
    assert query.account_id is None
    assert iterator.total == 5
    assert iterator.position == 5


def test_query_iterator_page_size_defaults_to_query_limit(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    query = EmailQuery(limit=3)
    calls = [
        (
            expected_query_request("Email", limit=3),
            query_response("Email", ["M1", "M2", "M3"], total=3),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    iterator = client.query(query)
    assert [i.id for i in iterator] == ["M1", "M2", "M3"]
    assert not queue
    assert query.limit == 3
    assert query.position is None
    assert query.account_id is None
    assert iterator.page == 1


def test_query_iterator_limit_cap_without_total(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    calls = [
        (
            expected_query_request("Email", limit=2),
            query_response("Email", ["M1", "M2"]),
        ),
        (
            expected_query_request("Email", position=2, limit=2),
            query_response("Email", ["M3", "M4"], position=2),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    query = EmailQuery(limit=4)
    iterator = client.query(query, page_size=2)
    assert [i.id for i in iterator] == ["M1", "M2", "M3", "M4"]
    # Query limit cap reached on full pages: no probe request is sent
    assert not queue
    assert iterator.total is None
    assert iterator.position == 4
    assert iterator.page == 2
    assert query.limit == 4
    assert query.position is None


def test_query_iterator_starting_position(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    calls = [
        (
            expected_query_request("Email", position=10, limit=5),
            query_response("Email", ["M1", "M2"], position=10, total=12),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    iterator = client.query(EmailQuery(position=10), page_size=5)
    items = list(iterator)
    assert [(i.page, i.position, i.id) for i in items] == [
        (0, 10, "M1"),
        (0, 11, "M2"),
    ]
    # Short page and reached total: no probe request is sent
    assert not queue
    assert iterator.page == 1
    assert iterator.position == 12
    assert iterator.total == 12


def test_query_iterator_duplicate_id(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    calls = [
        (
            expected_query_request("Email", limit=2),
            query_response("Email", ["M1", "M2"]),
        ),
        (
            expected_query_request("Email", position=2, limit=2),
            query_response("Email", ["M2", "M3"], position=2),
        ),
    ]
    queue_jmap_calls(http_responses, calls)
    iterator = client.query(EmailQuery(), page_size=2)
    with pytest.raises(QueryDuplicateIdError) as exc_info:
        list(iterator)
    assert exc_info.value.result_id == "M2"
    assert exc_info.value.page == 1
    assert exc_info.value.position == 2
    assert isinstance(exc_info.value, QueryIteratorError)
    assert iterator.page == 2
    assert iterator.query_state == "query:state:1"


def test_query_iterator_stalled_position(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    calls = [
        (
            expected_query_request("Email", limit=2),
            query_response("Email", ["M1", "M2"]),
        ),
        (
            expected_query_request("Email", position=2, limit=2),
            query_response("Email", ["M3", "M4"], position=0),
        ),
    ]
    queue_jmap_calls(http_responses, calls)
    iterator = client.query(EmailQuery(), page_size=2)
    with pytest.raises(QueryStalledPositionError) as exc_info:
        list(iterator)
    assert exc_info.value.requested_position == 2
    assert exc_info.value.response_position == 0
    assert exc_info.value.page == 1
    assert isinstance(exc_info.value, QueryIteratorError)


def test_query_iterator_state_change(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    calls = [
        (
            expected_query_request("Email", limit=2),
            query_response("Email", ["M1", "M2"], query_state="query:state:a"),
        ),
        (
            expected_query_request("Email", position=2, limit=2),
            query_response(
                "Email",
                ["M3", "M4"],
                position=2,
                query_state="query:state:b",
            ),
        ),
    ]
    queue_jmap_calls(http_responses, calls)
    iterator = client.query(EmailQuery(), page_size=2)
    with pytest.raises(QueryStateChangedError) as exc_info:
        list(iterator)
    assert exc_info.value.previous_state == "query:state:a"
    assert exc_info.value.new_state == "query:state:b"
    assert exc_info.value.page == 1
    assert isinstance(exc_info.value, QueryIteratorError)


@pytest.mark.parametrize(
    (
        "namespace",
        "method",
        "response_type",
        "urns",
        "arguments",
    ),
    [
        (
            "Email",
            EmailQuery(),
            EmailQueryResponse,
            {JMAP_URN_MAIL},
            None,
        ),
        (
            "Mailbox",
            MailboxQuery(),
            MailboxQueryResponse,
            {JMAP_URN_MAIL},
            {"sortAsTree": False, "filterAsTree": False},
        ),
        (
            "EmailSubmission",
            EmailSubmissionQuery(),
            EmailSubmissionQueryResponse,
            {JMAP_URN_SUBMISSION},
            None,
        ),
    ],
)
def test_query_iterator_query_subclasses(
    client: Client,
    http_responses: responses.RequestsMock,
    *,
    namespace: str,
    method: Query,
    response_type: type[EmailQueryResponse],
    urns: set[str],
    arguments: dict[str, Any] | None,
) -> None:
    calls = [
        (
            expected_query_request(
                namespace,
                limit=2,
                urns=urns,
                arguments=arguments,
            ),
            query_response(namespace, ["id1", "id2"], total=2),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    iterator = client.query(method, page_size=2)
    items = list(iterator)
    assert [i.id for i in items] == ["id1", "id2"]
    assert all(isinstance(i.response, response_type) for i in items)
    assert iterator.total == 2
    assert not queue


def test_query_iterator_cancellation(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    calls = [
        (
            expected_query_request("Email", limit=2),
            query_response("Email", ["M1", "M2"], total=5),
        ),
        (
            expected_query_request("Email", position=2, limit=2),
            query_response("Email", ["M3", "M4"], position=2, total=5),
        ),
    ]
    queue = queue_jmap_calls(http_responses, calls)
    query = EmailQuery()
    iterator = client.query(query, page_size=2)
    consumed: list[QueryIteratorItem] = []
    for item in iterator:
        consumed.append(item)
        if len(consumed) == 2:
            break
    assert [i.id for i in consumed] == ["M1", "M2"]
    # Only the first page was requested
    assert len(queue) == 1
    assert iterator.page == 1
    assert iterator.position == 2
    assert iterator.total == 5
    assert iterator.query_state == "query:state:1"
    assert query.position is None
    assert query.account_id is None


def test_query_iterator_invalid_arguments(client: Client) -> None:
    with pytest.raises(TypeError, match="Query method instance"):
        client.query(
            CoreEcho(  # ty: ignore[invalid-argument-type]
                data={},
            ),
            page_size=2,
        )
    with pytest.raises(ValueError, match="page_size"):
        client.query(EmailQuery(), page_size=0)
    with pytest.raises(ValueError, match="page_size"):
        client.query(EmailQuery(), page_size=-3)
    with pytest.raises(ValueError, match="Query limit"):
        client.query(EmailQuery(limit=0))


def test_query_iterator_non_list_ids(client: Client) -> None:
    iterator = client.query(EmailQuery(), page_size=2)
    malformed_response = EmailQueryResponse(
        account_id="u1138",
        query_state="query:state:1",
        can_calculate_changes=True,
        position=0,
        ids="not-a-list",  # ty: ignore[invalid-argument-type]
    )
    with pytest.raises(TypeError, match="ids must be a list"):
        iterator._handle_response(  # noqa: SLF001
            malformed_response, page=0, request_limit=2
        )


def test_query_iterator_server_error(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    response = {
        "methodResponses": [
            [
                "error",
                {"type": "serverFail", "description": "boom"},
                "single.Email/query",
            ]
        ]
    }
    queue_jmap_calls(
        http_responses,
        [(expected_query_request("Email", limit=2), response)],
    )
    with pytest.raises(ClientError):
        list(client.query(EmailQuery(), page_size=2))
    assert not issubclass(ClientError, QueryIteratorError)


def test_single_request_remains_supported(
    client: Client, http_responses: responses.RequestsMock
) -> None:
    query = EmailQuery(limit=10)
    calls = [
        (
            expected_query_request("Email", limit=10),
            query_response("Email", ["M1"], total=1, limit=10),
        ),
    ]
    queue_jmap_calls(http_responses, calls)
    result = client.request(query)
    assert result == EmailQueryResponse(
        account_id="u1138",
        ids=["M1"],
        query_state="query:state:1",
        can_calculate_changes=True,
        position=0,
        total=1,
        limit=10,
    )
    assert query.limit == 10
    assert query.position is None
