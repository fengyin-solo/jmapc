from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .api import APIRequest, APIResponse
from .methods import InvocationResponseOrError

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from .methods import Invocation


class BatchShardingError(ValueError):
    """Calls cannot be split into requests allowed by the server."""


@dataclass
class ShardSuccess:
    index: int
    request: APIRequest
    response: APIResponse


@dataclass
class ShardFailure:
    index: int
    request: APIRequest
    error: Exception
    method_responses: list[InvocationResponseOrError] = field(
        default_factory=list
    )


@dataclass
class BatchResult:
    succeeded: list[ShardSuccess]
    failed: ShardFailure | None
    unsent: list[APIRequest]

    @property
    def ok(self) -> bool:
        return self.failed is None

    @property
    def method_responses(self) -> list[InvocationResponseOrError]:
        return [
            method_response
            for shard in self.succeeded
            for method_response in shard.response.method_responses
        ]

    @property
    def created_ids(self) -> dict[str, str]:
        created_ids: dict[str, str] = {}
        for shard in self.succeeded:
            created_ids.update(shard.response.created_ids)
        return created_ids


def _iter_result_of_ids(value: Any) -> Iterator[str]:
    """Yield invocation ids referenced by result references in method args.

    Serialized result references use a ``#``-prefixed argument name whose
    value is an object containing ``resultOf``.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key.startswith("#")
                and isinstance(item, dict)
                and isinstance(item.get("resultOf"), str)
            ):
                yield item["resultOf"]
            else:
                yield from _iter_result_of_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_result_of_ids(item)


def _request_size(request: APIRequest) -> int:
    # The JSON string is transmitted verbatim as the HTTP request body, so
    # its UTF-8 encoded length is its on-the-wire size in octets.
    return len(request.to_json().encode("utf-8"))


def _check_singleton_sizes(
    account_id: str,
    invocations: Sequence[Invocation],
    full_request: APIRequest,
    max_size_request: int,
) -> None:
    # A single call which exceeds the size limit can never be sent, so
    # fail before any HTTP request is made. Each call is measured with its
    # fully resolved arguments from the full batch serialization.
    for invocation, method_call in zip(
        invocations, full_request.method_calls, strict=True
    ):
        singleton_request = APIRequest(
            account_id=account_id, method_calls=[method_call]
        )
        singleton_request.using |= set(invocation.method.using)
        singleton_size = _request_size(singleton_request)
        if singleton_size > max_size_request:
            raise BatchShardingError(
                f"Invocation {invocation.id!r} serialized request size "
                f"{singleton_size} bytes exceeds server max_size_request "
                f"{max_size_request} bytes"
            )


def _find_dependencies(
    invocations: Sequence[Invocation],
    full_request: APIRequest,
) -> list[set[int]]:
    id_to_index = {
        invocation.id: i for i, invocation in enumerate(invocations)
    }
    dependencies: list[set[int]] = [set() for _ in invocations]
    for i, (_name, method_args, _id) in enumerate(full_request.method_calls):
        for result_of in _iter_result_of_ids(method_args):
            target = id_to_index.get(result_of)
            if target is None or target >= i:
                raise BatchShardingError(
                    f"Invocation {invocations[i].id!r} references result "
                    f"of unknown or later call {result_of!r}"
                )
            dependencies[i].add(target)
    return dependencies


def _crossed_boundaries(
    dependencies: Sequence[set[int]], call_count: int
) -> list[bool]:
    # A boundary before call index b is crossed when a call at index j >= b
    # references a call at index t < b. Crossed boundaries cannot separate
    # shards: both sides of a result reference must share a request.
    crossed = [False] * (call_count + 1)
    for j, targets in enumerate(dependencies):
        for t in targets:
            for boundary in range(t + 1, j + 1):
                crossed[boundary] = True
    return crossed


def plan_request_shards(
    account_id: str,
    invocations: Sequence[Invocation],
    max_calls_in_request: int,
    max_size_request: int,
) -> list[APIRequest]:
    """Split invocations into server-legal API requests.

    Invocation order and ids are preserved. Calls connected by result
    references are kept in the same shard.

    All shards are fully planned and validated before any HTTP request is
    sent.
    """
    call_count = len(invocations)
    if call_count == 0:
        return []
    if max_calls_in_request < 1:
        raise BatchShardingError(
            "Server allows fewer than one call per request "
            f"(max_calls_in_request={max_calls_in_request})"
        )
    # Serialize the full batch once to discover result reference targets
    # using the same resolution as a single-request batch.
    full_request = APIRequest.from_invocations(account_id, invocations)
    _check_singleton_sizes(
        account_id, invocations, full_request, max_size_request
    )
    dependencies = _find_dependencies(invocations, full_request)
    crossed = _crossed_boundaries(dependencies, call_count)

    def _build_shard(indices: Sequence[int]) -> APIRequest:
        return APIRequest.from_invocations(
            account_id, [invocations[i] for i in indices]
        )

    shards: list[APIRequest] = []
    start = 0
    current: list[int] = []
    i = 0
    while i < call_count:
        candidate = [*current, i]
        candidate_request = _build_shard(candidate)
        fits = (
            len(candidate) <= max_calls_in_request
            and _request_size(candidate_request) <= max_size_request
        )
        if fits:
            current = candidate
            i += 1
            continue
        # Adding call i would exceed a server limit. Seal the current shard
        # at the latest valid boundary before i.
        boundary = i
        while boundary > start and crossed[boundary]:
            boundary -= 1
        if boundary == start:
            raise BatchShardingError(
                "Result reference group starting at invocation "
                f"{invocations[start].id!r} exceeds server limits "
                f"(max_calls_in_request={max_calls_in_request}, "
                f"max_size_request={max_size_request} bytes) and cannot be "
                "split without breaking a result reference"
            )
        shards.append(_build_shard(list(range(start, boundary))))
        start = boundary
        current = []
    shards.append(_build_shard(current))
    return shards
