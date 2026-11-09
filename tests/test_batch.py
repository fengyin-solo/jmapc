from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
import requests
import responses

from jmapc import (
    BatchShardingError,
    Client,
    ClientError,
    Ref,
    ResultReference,
)
from jmapc.api import APIRequest
from jmapc.batch import _iter_result_of_ids, plan_request_shards
from jmapc.errors import ServerFail
from jmapc.methods import (
    CoreEcho,
    CoreEchoResponse,
    InvocationResponseOrError,
    MailboxGet,
    MailboxGetResponse,
)

from .data import make_session_response

ACCOUNT_ID = "u1138"
API_URL = "https://jmap-api.localhost/api"
SESSION_URL = "https://jmap-example.localhost/.well-known/jmap"


def make_limited_session_response(
    *,
    max_calls_in_request: int = 16,
    max_size_request: int = 10_000_000,
    state: str = "test;session;state",
    username: str = "ness@onett.example.net",
) -> dict[str, Any]:
    session_response = make_session_response()
    session_response["state"] = state
    session_response["username"] = username
    session_response["capabilities"]["urn:ietf:params:jmap:core"][
        "maxCallsInRequest"
    ] = max_calls_in_request
    session_response["capabilities"]["urn:ietf:params:jmap:core"][
        "maxSizeRequest"
    ] = max_size_request
    return session_response


def register_session(
    http_responses: responses.RequestsMock,
    *,
    max_calls_in_request: int = 16,
    max_size_request: int = 10_000_000,
    state: str = "test;session;state",
    username: str = "ness@onett.example.net",
) -> None:
    http_responses.add(
        method=responses.GET,
        url=SESSION_URL,
        body=json.dumps(
            make_limited_session_response(
                max_calls_in_request=max_calls_in_request,
                max_size_request=max_size_request,
                state=state,
                username=username,
            )
        ),
    )


def echo_calls(count: int) -> list[CoreEcho]:
    return [CoreEcho(data={"index": i}) for i in range(count)]


def echo_method_responses(
    method_calls: list[list[Any]],
) -> list[list[Any]]:
    return [[name, args, method_id] for name, args, method_id in method_calls]


def register_echo_shard(
    http_responses: responses.RequestsMock,
    expected_method_calls: list[list[Any]],
    *,
    using: list[str] | None = None,
    status: int = 200,
    session_state: str = "test;session;state",
    created_ids: dict[str, str] | None = None,
    method_response_override: (
        Callable[[list[list[Any]]], list[list[Any]]] | None
    ) = None,
) -> None:
    expected_request = {
        "methodCalls": expected_method_calls,
        "using": using or ["urn:ietf:params:jmap:core"],
    }

    def _callback(
        request: requests.PreparedRequest,
    ) -> tuple[int, dict[str, str], str]:
        body = json.loads(str(request.body) or "{}")
        assert body == expected_request
        method_responses = (
            method_response_override(expected_method_calls)
            if method_response_override
            else echo_method_responses(expected_method_calls)
        )
        response: dict[str, Any] = {
            "methodResponses": method_responses,
            "sessionState": session_state,
        }
        if created_ids is not None:
            response["createdIds"] = created_ids
        return status, {}, json.dumps(response)

    http_responses.add_callback(
        method=responses.POST,
        url=API_URL,
        callback=_callback,
    )


def expected_echo_call(i: int) -> list[Any]:
    return ["Core/echo", {"index": i}, f"{i}.Core/echo"]


def expected_echo_responses(
    count: int,
) -> list[InvocationResponseOrError]:
    return [
        InvocationResponseOrError(
            id=f"{i}.Core/echo",
            response=CoreEchoResponse(data={"index": i}),
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Planner tests (no HTTP)
# ---------------------------------------------------------------------------


def test_plan_zero_calls() -> None:
    assert (
        plan_request_shards(
            account_id=ACCOUNT_ID,
            invocations=[],
            max_calls_in_request=16,
            max_size_request=10_000_000,
        )
        == []
    )


def test_plan_single_call() -> None:
    invocations = APIRequest.invocations_from_calls(echo_calls(1))
    shards = plan_request_shards(
        account_id=ACCOUNT_ID,
        invocations=invocations,
        max_calls_in_request=16,
        max_size_request=10_000_000,
    )
    assert len(shards) == 1
    assert [call[2] for call in shards[0].method_calls] == ["single.Core/echo"]


def test_plan_max_calls_in_request() -> None:
    invocations = APIRequest.invocations_from_calls(echo_calls(5))
    shards = plan_request_shards(
        account_id=ACCOUNT_ID,
        invocations=invocations,
        max_calls_in_request=2,
        max_size_request=10_000_000,
    )
    shard_ids = [[call[2] for call in shard.method_calls] for shard in shards]
    assert shard_ids == [
        ["0.Core/echo", "1.Core/echo"],
        ["2.Core/echo", "3.Core/echo"],
        ["4.Core/echo"],
    ]
    for shard in shards:
        assert len(shard.method_calls) <= 2


def test_plan_max_size_request() -> None:
    invocations = APIRequest.invocations_from_calls(
        [CoreEcho(data={"payload": "x" * 200}) for _ in range(5)]
    )
    two_call_size = len(
        APIRequest.from_invocations(ACCOUNT_ID, invocations[:2])
        .to_json()
        .encode("utf-8")
    )
    shards = plan_request_shards(
        account_id=ACCOUNT_ID,
        invocations=invocations,
        max_calls_in_request=16,
        max_size_request=two_call_size,
    )
    assert [[call[2] for call in shard.method_calls] for shard in shards] == [
        ["0.Core/echo", "1.Core/echo"],
        ["2.Core/echo", "3.Core/echo"],
        ["4.Core/echo"],
    ]
    for shard in shards:
        assert len(shard.to_json().encode("utf-8")) <= two_call_size


def test_plan_oversized_single_call() -> None:
    invocations = APIRequest.invocations_from_calls(
        [
            CoreEcho(data={"payload": "x" * 200}),
            CoreEcho(data={"index": 1}),
        ]
    )
    singleton_size = len(
        APIRequest.from_invocations(ACCOUNT_ID, [invocations[0]])
        .to_json()
        .encode("utf-8")
    )
    with pytest.raises(
        BatchShardingError,
        match=(
            r"Invocation '0\.Core/echo' serialized request size \d+ bytes "
            "exceeds server max_size_request"
        ),
    ):
        plan_request_shards(
            account_id=ACCOUNT_ID,
            invocations=invocations,
            max_calls_in_request=16,
            max_size_request=singleton_size - 1,
        )


def test_plan_reference_chains_split_per_chain() -> None:
    calls = [
        CoreEcho(data={"index": 0}),
        MailboxGet(ids=Ref(path="/list", method=-1)),
        CoreEcho(data={"index": 2}),
        MailboxGet(ids=Ref(path="/list", method=-1)),
    ]
    invocations = APIRequest.invocations_from_calls(calls)
    shards = plan_request_shards(
        account_id=ACCOUNT_ID,
        invocations=invocations,
        max_calls_in_request=2,
        max_size_request=10_000_000,
    )
    assert [[call[2] for call in shard.method_calls] for shard in shards] == [
        ["0.Core/echo", "1.Mailbox/get"],
        ["2.Core/echo", "3.Mailbox/get"],
    ]
    # Each reference resolves within its own shard
    for shard, expected_result_of in zip(
        shards, ["0.Core/echo", "2.Core/echo"], strict=True
    ):
        mailbox_args = shard.method_calls[1][1]
        assert mailbox_args["#ids"]["resultOf"] == expected_result_of


def test_plan_reference_chain_grouped_with_independent_call() -> None:
    calls = [
        CoreEcho(data={"index": 0}),
        CoreEcho(data={"index": 1}),
        MailboxGet(
            ids=Ref(path="/list", method="0.Core/echo"),
        ),
    ]
    invocations = APIRequest.invocations_from_calls(calls)
    # The reference from call 2 to call 0 forbids shard boundaries before
    # call 2, so all three calls must share a request when they fit.
    shards = plan_request_shards(
        account_id=ACCOUNT_ID,
        invocations=invocations,
        max_calls_in_request=3,
        max_size_request=10_000_000,
    )
    assert len(shards) == 1
    assert [call[2] for call in shards[0].method_calls] == [
        "0.Core/echo",
        "1.Core/echo",
        "2.Mailbox/get",
    ]


def test_plan_reference_group_exceeding_limits_fails() -> None:
    calls = [
        CoreEcho(data={"index": 0}),
        CoreEcho(data={"index": 1}),
        MailboxGet(
            ids=Ref(path="/list", method="0.Core/echo"),
        ),
    ]
    invocations = APIRequest.invocations_from_calls(calls)
    with pytest.raises(
        BatchShardingError,
        match=r"exceeds server limits .* cannot be split",
    ):
        plan_request_shards(
            account_id=ACCOUNT_ID,
            invocations=invocations,
            max_calls_in_request=2,
            max_size_request=10_000_000,
        )


def test_plan_long_reference_chain_exceeding_limits_fails() -> None:
    calls = [
        CoreEcho(data={"index": 0}),
        MailboxGet(
            ids=Ref(path="/list", method="0.Core/echo"),
        ),
        MailboxGet(
            ids=Ref(path="/list", method="1.Mailbox/get"),
        ),
    ]
    invocations = APIRequest.invocations_from_calls(calls)
    with pytest.raises(BatchShardingError, match="exceeds server limits"):
        plan_request_shards(
            account_id=ACCOUNT_ID,
            invocations=invocations,
            max_calls_in_request=2,
            max_size_request=10_000_000,
        )


def test_plan_max_calls_in_request_zero_fails() -> None:
    invocations = APIRequest.invocations_from_calls(echo_calls(1))
    with pytest.raises(
        BatchShardingError, match="fewer than one call per request"
    ):
        plan_request_shards(
            account_id=ACCOUNT_ID,
            invocations=invocations,
            max_calls_in_request=0,
            max_size_request=10_000_000,
        )


def test_plan_unknown_result_reference_fails() -> None:
    calls = [
        CoreEcho(data={"index": 0}),
        MailboxGet(
            ids=ResultReference(
                name="Core/echo",
                path="/list",
                result_of="9.does_not_exist",
            )
        ),
    ]
    invocations = APIRequest.invocations_from_calls(calls)
    with pytest.raises(
        BatchShardingError,
        match=(
            r"Invocation '1\.Mailbox/get' references result of unknown or "
            r"later call '9\.does_not_exist'"
        ),
    ):
        plan_request_shards(
            account_id=ACCOUNT_ID,
            invocations=invocations,
            max_calls_in_request=16,
            max_size_request=10_000_000,
        )


def test_iter_result_of_ids_nested() -> None:
    method_args = {
        "accountId": ACCOUNT_ID,
        "list": [
            {
                "#ref": {
                    "name": "Core/echo",
                    "path": "/a",
                    "resultOf": "0.Core/echo",
                }
            },
            {"plain": "value"},
        ],
        "other": {
            "deep": [
                {
                    "#ref": {
                        "name": "Core/echo",
                        "path": "/b",
                        "resultOf": "1.Core/echo",
                    }
                }
            ]
        },
    }
    assert list(_iter_result_of_ids(method_args)) == [
        "0.Core/echo",
        "1.Core/echo",
    ]
    assert list(_iter_result_of_ids("plain")) == []


def test_api_request_from_empty_invocations() -> None:
    api_request = APIRequest.from_invocations(ACCOUNT_ID, [])
    assert api_request.method_calls == []


# ---------------------------------------------------------------------------
# Client request_batch HTTP tests
# ---------------------------------------------------------------------------


def test_request_batch_zero_calls(
    http_responses_base: responses.RequestsMock,
) -> None:
    # No session and no API request are made for an empty batch
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    result = client.request_batch([])
    assert result.ok
    assert result.succeeded == []
    assert result.failed is None
    assert result.unsent == []
    assert result.method_responses == []
    assert result.created_ids == {}


def test_request_batch_single_call(
    http_responses_base: responses.RequestsMock,
) -> None:
    register_session(http_responses_base)
    register_echo_shard(
        http_responses_base,
        [["Core/echo", {"index": 0}, "single.Core/echo"]],
    )
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    result = client.request_batch(echo_calls(1))
    assert result.ok
    assert len(result.succeeded) == 1
    assert result.failed is None
    assert result.unsent == []
    assert result.method_responses == [
        InvocationResponseOrError(
            id="single.Core/echo",
            response=CoreEchoResponse(data={"index": 0}),
        )
    ]


def test_request_batch_max_calls_in_request(
    http_responses_base: responses.RequestsMock,
) -> None:
    register_session(http_responses_base, max_calls_in_request=2)
    for start in range(0, 5, 2):
        register_echo_shard(
            http_responses_base,
            [expected_echo_call(i) for i in range(start, min(start + 2, 5))],
        )
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    result = client.request_batch(echo_calls(5))
    assert result.ok
    assert result.failed is None
    assert result.unsent == []
    assert len(result.succeeded) == 3
    assert [shard.index for shard in result.succeeded] == [0, 1, 2]
    assert result.method_responses == expected_echo_responses(5)
    assert result.created_ids == {}


def test_request_batch_max_size_request(
    http_responses_base: responses.RequestsMock,
) -> None:
    max_size_request = 600
    register_session(
        http_responses_base,
        max_calls_in_request=16,
        max_size_request=max_size_request,
    )
    calls = [CoreEcho(data={"payload": "x" * 200}) for _ in range(4)]
    invocations = APIRequest.invocations_from_calls(calls)

    def _register_expected_shards() -> list[list[list[Any]]]:
        shards = plan_request_shards(
            account_id=ACCOUNT_ID,
            invocations=invocations,
            max_calls_in_request=16,
            max_size_request=max_size_request,
        )
        shard_calls: list[list[list[Any]]] = []
        for shard in shards:
            calls_out = [
                [name, args, method_id]
                for name, args, method_id in shard.method_calls
            ]
            register_echo_shard(http_responses_base, calls_out)
            shard_calls.append(calls_out)
        return shard_calls

    shard_calls = _register_expected_shards()
    assert len(shard_calls) >= 2
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    result = client.request_batch(calls)
    assert result.ok
    assert len(result.succeeded) == len(shard_calls)
    # Response order follows invocation order
    assert [response.id for response in result.method_responses] == [
        f"{i}.Core/echo" for i in range(4)
    ]


def test_request_batch_oversized_single_call_raises_before_send(
    http_responses_base: responses.RequestsMock,
) -> None:
    register_session(http_responses_base, max_size_request=100)
    # No POST callback is registered: sending the oversized call would raise
    # an unmatched-request error instead of BatchShardingError.
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    with pytest.raises(BatchShardingError, match="exceeds server max_size"):
        client.request_batch([CoreEcho(data={"payload": "x" * 500})])


def test_request_batch_reference_chain(
    http_responses_base: responses.RequestsMock,
) -> None:
    register_session(http_responses_base, max_calls_in_request=2)
    calls = [
        CoreEcho(data={"index": 0}),
        MailboxGet(ids=Ref(path="/list", method=-1)),
        CoreEcho(data={"index": 2}),
        MailboxGet(ids=Ref(path="/list", method=-1)),
    ]

    def _mailbox_response(
        method_calls: list[list[Any]],
    ) -> list[list[Any]]:
        return [
            [
                name,
                {
                    "accountId": ACCOUNT_ID,
                    "list": [],
                    "notFound": [],
                    "state": "1000",
                }
                if name == "Mailbox/get"
                else args,
                method_id,
            ]
            for name, args, method_id in method_calls
        ]

    register_echo_shard(
        http_responses_base,
        [
            ["Core/echo", {"index": 0}, "0.Core/echo"],
            [
                "Mailbox/get",
                {
                    "accountId": ACCOUNT_ID,
                    "#ids": {
                        "name": "Core/echo",
                        "path": "/list",
                        "resultOf": "0.Core/echo",
                    },
                },
                "1.Mailbox/get",
            ],
        ],
        using=[
            "urn:ietf:params:jmap:core",
            "urn:ietf:params:jmap:mail",
        ],
        method_response_override=_mailbox_response,
    )
    register_echo_shard(
        http_responses_base,
        [
            ["Core/echo", {"index": 2}, "2.Core/echo"],
            [
                "Mailbox/get",
                {
                    "accountId": ACCOUNT_ID,
                    "#ids": {
                        "name": "Core/echo",
                        "path": "/list",
                        "resultOf": "2.Core/echo",
                    },
                },
                "3.Mailbox/get",
            ],
        ],
        using=[
            "urn:ietf:params:jmap:core",
            "urn:ietf:params:jmap:mail",
        ],
        method_response_override=_mailbox_response,
    )
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    result = client.request_batch(calls)
    assert result.ok
    assert [response.id for response in result.method_responses] == [
        "0.Core/echo",
        "1.Mailbox/get",
        "2.Core/echo",
        "3.Mailbox/get",
    ]
    assert isinstance(result.method_responses[1].response, MailboxGetResponse)


def test_request_batch_session_refresh(
    http_responses_base: responses.RequestsMock,
) -> None:
    register_session(http_responses_base, max_calls_in_request=1)
    register_echo_shard(
        http_responses_base,
        [expected_echo_call(0)],
        session_state="updated;state;value",
    )
    # The refreshed session is fetched before the second shard is sent
    register_session(
        http_responses_base,
        max_calls_in_request=1,
        state="updated;state;value",
        username="paula@twoson.example.net",
    )
    register_echo_shard(
        http_responses_base,
        [expected_echo_call(1)],
        session_state="updated;state;value",
    )
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    assert client.jmap_session.username == "ness@onett.example.net"
    result = client.request_batch(echo_calls(2))
    assert result.ok
    assert len(result.succeeded) == 2
    assert client.jmap_session.username == "paula@twoson.example.net"


def test_request_batch_http_shard_failure(
    http_responses_base: responses.RequestsMock,
) -> None:
    register_session(http_responses_base, max_calls_in_request=1)
    register_echo_shard(http_responses_base, [expected_echo_call(0)])
    register_echo_shard(
        http_responses_base,
        [expected_echo_call(1)],
        status=500,
    )
    # No callback is registered for call 2: sending it would raise an
    # unmatched-request error, proving it remains unsent.
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    result = client.request_batch(echo_calls(3))
    assert not result.ok
    assert len(result.succeeded) == 1
    assert result.failed is not None
    assert result.failed.index == 1
    assert isinstance(result.failed.error, requests.exceptions.HTTPError)
    assert result.failed.error.response is not None
    assert result.failed.error.response.status_code == 500
    assert [call[2] for call in result.failed.request.method_calls] == [
        "1.Core/echo"
    ]
    assert len(result.unsent) == 1
    assert [call[2] for call in result.unsent[0].method_calls] == [
        "2.Core/echo"
    ]
    assert [response.id for response in result.method_responses] == [
        "0.Core/echo"
    ]


def test_request_batch_method_level_failure(
    http_responses_base: responses.RequestsMock,
) -> None:
    register_session(http_responses_base, max_calls_in_request=1)
    register_echo_shard(http_responses_base, [expected_echo_call(0)])

    def _error_response(
        _method_calls: list[list[Any]],
    ) -> list[list[Any]]:
        return [["error", {"type": "serverFail"}, "1.Core/echo"]]

    register_echo_shard(
        http_responses_base,
        [expected_echo_call(1)],
        method_response_override=_error_response,
    )
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    result = client.request_batch(echo_calls(3), raise_errors=True)
    assert not result.ok
    assert len(result.succeeded) == 1
    assert result.failed is not None
    assert result.failed.index == 1
    assert isinstance(result.failed.error, ClientError)
    assert len(result.failed.method_responses) == 1
    assert isinstance(result.failed.method_responses[0].response, ServerFail)
    assert len(result.unsent) == 1
    assert [response.id for response in result.method_responses] == [
        "0.Core/echo"
    ]


def test_request_batch_method_level_failure_not_raised(
    http_responses_base: responses.RequestsMock,
) -> None:
    register_session(http_responses_base, max_calls_in_request=1)
    register_echo_shard(http_responses_base, [expected_echo_call(0)])

    def _error_response(
        _method_calls: list[list[Any]],
    ) -> list[list[Any]]:
        return [["error", {"type": "serverFail"}, "1.Core/echo"]]

    register_echo_shard(
        http_responses_base,
        [expected_echo_call(1)],
        method_response_override=_error_response,
    )
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    result = client.request_batch(echo_calls(2), raise_errors=False)
    assert result.ok
    assert result.failed is None
    assert isinstance(result.method_responses[1].response, ServerFail)


def test_request_batch_created_ids_merged(
    http_responses_base: responses.RequestsMock,
) -> None:
    register_session(http_responses_base, max_calls_in_request=2)
    register_echo_shard(
        http_responses_base,
        [expected_echo_call(0), expected_echo_call(1)],
        created_ids={"c0": "id0", "c1": "id1"},
    )
    register_echo_shard(
        http_responses_base,
        [expected_echo_call(2), expected_echo_call(3)],
        created_ids={"c2": "id2", "c3": "id3"},
    )
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    result = client.request_batch(echo_calls(4))
    assert result.ok
    assert result.created_ids == {
        "c0": "id0",
        "c1": "id1",
        "c2": "id2",
        "c3": "id3",
    }
    assert [shard.response.created_ids for shard in result.succeeded] == [
        {"c0": "id0", "c1": "id1"},
        {"c2": "id2", "c3": "id3"},
    ]


def test_request_still_sends_single_batch(
    http_responses_base: responses.RequestsMock,
) -> None:
    # Existing Client.request keeps its single-request behavior even when
    # the call count exceeds the server's maxCallsInRequest.
    register_session(http_responses_base, max_calls_in_request=2)
    register_echo_shard(
        http_responses_base,
        [expected_echo_call(i) for i in range(5)],
    )
    client = Client(host="jmap-example.localhost", auth=("ness", "pk_fire"))
    result = client.request(echo_calls(5))
    assert result == expected_echo_responses(5)
