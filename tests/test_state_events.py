import json
import threading
from collections.abc import Iterable
from typing import Any
from unittest import mock

import pytest
import requests
import responses
import sseclient

from jmapc import (
    RECONNECT_REASON_HTTP_ERROR,
    RECONNECT_REASON_STREAM_CLOSED,
    Client,
    Event,
    ReconnectRecord,
    StateChange,
    StateEventsAuthError,
    StateEventsConfig,
    StateEventsError,
    StateEventsInvalidEventError,
    StateEventsRetryLimitError,
    TypeState,
)

EVENT_SOURCE_URL = "https://jmap-api.localhost/events/*/no/0"
AUTH = ("ness", "pk_fire")


@pytest.fixture
def mock_sseclient() -> Iterable[mock.MagicMock]:
    mock_client = mock.MagicMock(spec=sseclient.SSEClient)
    mock_client.return_value.__iter__.side_effect = lambda: (
        mock_client.return_value
    )
    with mock.patch.object(sseclient, "SSEClient", mock_client):
        yield mock_client


def state_event(
    event_id: str | None, email_state: str = "1001"
) -> sseclient.Event:
    return sseclient.Event(
        id=event_id,
        event="state",
        data=json.dumps({"changed": {"u1138": {"Email": email_state}}}),
    )


def expected_event(event_id: str | None, email_state: str = "1001") -> Event:
    return Event(
        id=event_id,
        data=StateChange(changed={"u1138": TypeState(email=email_state)}),
    )


def http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(response=response)


def set_next_side_effect(
    mock_client: mock.MagicMock, events: list[object]
) -> None:
    mock_client.return_value.__next__.side_effect = events


def last_ids(mock_client: mock.MagicMock) -> list[str | None]:
    return [call.kwargs["last_id"] for call in mock_client.call_args_list]


def test_state_events_first_connection(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(mock_sseclient, [state_event("8001"), StopIteration])
    subscription = client.state_events(
        StateEventsConfig(ping=0, max_retries=0)
    )
    assert list(subscription) == [expected_event("8001")]
    assert subscription.checkpoint == "8001"
    mock_sseclient.assert_called_once_with(
        EVENT_SOURCE_URL, auth=AUTH, last_id=None
    )


def test_state_events_consecutive_state_events(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(
        mock_sseclient,
        [
            state_event("1"),
            state_event("2"),
            state_event("3"),
            StopIteration,
        ],
    )
    events = list(client.state_events(StateEventsConfig(max_retries=0)))
    assert events == [
        expected_event("1"),
        expected_event("2"),
        expected_event("3"),
    ]
    mock_sseclient.assert_called_once()


def test_state_events_ignores_ping_and_empty_events(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(
        mock_sseclient,
        [
            state_event("1"),
            sseclient.Event(id="p1", event="ping", data="ignore-me"),
            sseclient.Event(id="", event="ping", data="also-ignore-me"),
            sseclient.Event(event="state", data=""),
            None,
            state_event("2"),
            StopIteration,
        ],
    )
    events = list(client.state_events(StateEventsConfig(max_retries=0)))
    assert events == [expected_event("1"), expected_event("2")]
    mock_sseclient.assert_called_once()


def test_state_events_reconnect_on_close_with_checkpoint(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(
        mock_sseclient,
        [
            state_event("1"),
            StopIteration,
            state_event("2"),
            StopIteration,
            state_event("3"),
            StopIteration,
        ],
    )
    records: list[ReconnectRecord] = []
    subscription = client.state_events(
        StateEventsConfig(
            max_retries=2,
            initial_delay=0.01,
            max_delay=0.05,
            on_reconnect=records.append,
        )
    )
    events = list(subscription)
    assert events == [
        expected_event("1"),
        expected_event("2"),
        expected_event("3"),
    ]
    assert mock_sseclient.call_count == 3
    assert last_ids(mock_sseclient) == [None, "1", "2"]
    assert subscription.checkpoint == "3"
    assert [r.attempt for r in records] == [1, 2]
    assert [r.last_event_id for r in records] == ["1", "2"]
    assert [r.delay for r in records] == [0.01, 0.02]
    assert all(r.reason == RECONNECT_REASON_STREAM_CLOSED for r in records)
    assert all(r.error is None for r in records)
    assert records == subscription.reconnects


def test_state_events_initial_checkpoint(
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    client = Client(
        host="jmap-example.localhost",
        auth=AUTH,
        last_event_id="9000",
    )
    set_next_side_effect(
        mock_sseclient,
        [
            state_event("9001"),
            StopIteration,
            state_event("9002"),
            StopIteration,
        ],
    )
    subscription = client.state_events(
        StateEventsConfig(max_retries=1, initial_delay=0.01)
    )
    assert [event.id for event in subscription] == ["9001", "9002"]
    assert last_ids(mock_sseclient) == ["9000", "9001"]
    assert subscription.checkpoint == "9002"


def test_state_events_config_checkpoint_overrides_client(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(mock_sseclient, [state_event("8"), StopIteration])
    subscription = client.state_events(
        StateEventsConfig(max_retries=0, last_event_id="7")
    )
    assert [event.id for event in subscription] == ["8"]
    mock_sseclient.assert_called_once_with(
        EVENT_SOURCE_URL, auth=AUTH, last_id="7"
    )


def test_state_events_ping_interval_in_url(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(mock_sseclient, [state_event("1"), StopIteration])
    list(client.state_events(StateEventsConfig(ping=37, max_retries=0)))
    mock_sseclient.assert_called_once_with(
        "https://jmap-api.localhost/events/*/no/37",
        auth=AUTH,
        last_id=None,
    )


def test_state_events_backoff_is_exponential_and_capped(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(
        mock_sseclient,
        [
            requests.ConnectionError(),
            requests.ConnectionError(),
            requests.ConnectionError(),
            requests.ConnectionError(),
            requests.ConnectionError(),
        ],
    )
    subscription = client.state_events(
        StateEventsConfig(max_retries=4, initial_delay=1.0, max_delay=5.0)
    )
    with (
        mock.patch("jmapc.event_subscription.time.sleep") as sleep_mock,
        pytest.raises(StateEventsRetryLimitError),
    ):
        list(subscription)
    assert [call.args[0] for call in sleep_mock.call_args_list] == [
        1.0,
        2.0,
        4.0,
        5.0,
    ]
    assert mock_sseclient.call_count == 5


def test_state_events_stop_before_start(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    stop_event = threading.Event()
    stop_event.set()
    subscription = client.state_events(
        StateEventsConfig(stop_event=stop_event)
    )
    assert list(subscription) == []
    mock_sseclient.assert_not_called()


def test_state_events_stop_during_reconnect(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    stop_event = threading.Event()
    set_next_side_effect(
        mock_sseclient,
        [state_event("1"), StopIteration, state_event("2"), StopIteration],
    )

    def stop_on_reconnect(record: ReconnectRecord) -> None:
        stop_event.set()

    subscription = client.state_events(
        StateEventsConfig(
            max_retries=3,
            initial_delay=0.01,
            stop_event=stop_event,
            on_reconnect=stop_on_reconnect,
        )
    )
    assert [event.id for event in subscription] == ["1"]
    assert mock_sseclient.call_count == 1
    assert len(subscription.reconnects) == 1


def test_state_events_stop_during_iteration(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    stop_event = threading.Event()
    delivered = False

    def side_effect(*_args: object) -> sseclient.Event:
        nonlocal delivered
        if delivered:
            raise StopIteration
        delivered = True
        stop_event.set()
        return state_event("1")

    mock_sseclient.return_value.__next__.side_effect = side_effect
    subscription = client.state_events(
        StateEventsConfig(stop_event=stop_event)
    )
    assert [event.id for event in subscription] == ["1"]
    assert mock_sseclient.call_count == 1


def test_state_events_retry_exhausted_on_network_error(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(
        mock_sseclient,
        [
            requests.ConnectionError(),
            requests.ConnectionError(),
            requests.ConnectionError(),
        ],
    )
    subscription = client.state_events(
        StateEventsConfig(max_retries=2, initial_delay=0.01)
    )
    with pytest.raises(StateEventsRetryLimitError) as exc_info:
        list(subscription)
    error = exc_info.value
    assert error.attempts == 2
    assert error.last_event_id is None
    assert isinstance(error.last_error, requests.ConnectionError)
    assert mock_sseclient.call_count == 3
    assert len(subscription.reconnects) == 2


def test_state_events_retry_exhausted_on_connection_failure(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    mock_sseclient.side_effect = requests.ConnectionError()
    subscription = client.state_events(
        StateEventsConfig(max_retries=1, initial_delay=0.01)
    )
    with pytest.raises(StateEventsRetryLimitError) as exc_info:
        list(subscription)
    assert exc_info.value.attempts == 1
    assert mock_sseclient.call_count == 2


@pytest.mark.parametrize("status_code", [401, 403])
def test_state_events_auth_error_terminates_immediately(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
    status_code: int,
) -> None:
    mock_sseclient.side_effect = http_error(status_code)
    with pytest.raises(StateEventsAuthError) as exc_info:
        list(client.state_events(StateEventsConfig(max_retries=5)))
    assert exc_info.value.status_code == status_code
    mock_sseclient.assert_called_once()


def test_state_events_other_client_error_terminates_immediately(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    mock_sseclient.side_effect = http_error(404)
    with pytest.raises(StateEventsError) as exc_info:
        list(client.state_events(StateEventsConfig(max_retries=5)))
    assert not isinstance(exc_info.value, StateEventsAuthError)
    mock_sseclient.assert_called_once()


def test_state_events_retryable_http_error_reconnects(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(
        mock_sseclient,
        [state_event("1"), http_error(500), state_event("2"), StopIteration],
    )
    subscription = client.state_events(
        StateEventsConfig(max_retries=1, initial_delay=0.01)
    )
    with mock.patch("jmapc.event_subscription.time.sleep"):
        events = list(subscription)
    assert events == [expected_event("1"), expected_event("2")]
    assert last_ids(mock_sseclient) == [None, "1"]
    assert len(subscription.reconnects) == 1
    record = subscription.reconnects[0]
    assert record.reason == RECONNECT_REASON_HTTP_ERROR
    assert isinstance(record.error, requests.HTTPError)


@pytest.mark.parametrize("status_code", [408, 429, 503])
def test_state_events_retryable_status_on_connect_reconnects(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
    status_code: int,
) -> None:
    mock_sseclient.side_effect = [
        http_error(status_code),
        mock_sseclient.return_value,
    ]
    set_next_side_effect(mock_sseclient, [state_event("1"), StopIteration])
    subscription = client.state_events(
        StateEventsConfig(max_retries=1, initial_delay=0.01)
    )
    with mock.patch("jmapc.event_subscription.time.sleep"):
        events = list(subscription)
    assert events == [expected_event("1")]
    assert mock_sseclient.call_count == 2
    assert subscription.reconnects[0].reason == (RECONNECT_REASON_HTTP_ERROR)


def test_state_events_retryable_http_error_retry_exhausted(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    mock_sseclient.side_effect = [http_error(500), http_error(503)]
    subscription = client.state_events(
        StateEventsConfig(max_retries=1, initial_delay=0.01)
    )
    with pytest.raises(StateEventsRetryLimitError) as exc_info:
        list(subscription)
    assert exc_info.value.attempts == 1
    assert isinstance(exc_info.value.last_error, requests.HTTPError)
    assert mock_sseclient.call_count == 2


def test_state_events_stop_during_backoff(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    stop_event = threading.Event()
    set_next_side_effect(mock_sseclient, [requests.ConnectionError()])

    def stop_on_sleep(_seconds: float) -> None:
        stop_event.set()

    subscription = client.state_events(
        StateEventsConfig(
            max_retries=3,
            initial_delay=1.0,
            stop_event=stop_event,
        )
    )
    with mock.patch(
        "jmapc.event_subscription.time.sleep", side_effect=stop_on_sleep
    ):
        assert list(subscription) == []
    assert mock_sseclient.call_count == 1
    assert len(subscription.reconnects) == 1


def test_state_events_delivers_events_without_id(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(
        mock_sseclient,
        [state_event(None), state_event("2"), StopIteration],
    )
    subscription = client.state_events(StateEventsConfig(max_retries=0))
    assert list(subscription) == [
        expected_event(None),
        expected_event("2"),
    ]
    assert subscription.checkpoint == "2"


def test_state_events_duplicate_ids_not_delivered_twice(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(
        mock_sseclient,
        [
            state_event("1"),
            StopIteration,
            state_event("1"),
            state_event("2"),
            StopIteration,
        ],
    )
    subscription = client.state_events(
        StateEventsConfig(max_retries=1, initial_delay=0.01)
    )
    assert [event.id for event in subscription] == ["1", "2"]
    assert last_ids(mock_sseclient) == [None, "1"]
    assert subscription.checkpoint == "2"


def test_state_events_invalid_event_terminates_immediately(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    invalid_event = sseclient.Event(
        id="broken", event="state", data="{not valid json"
    )
    set_next_side_effect(
        mock_sseclient, [invalid_event, state_event("never-delivered")]
    )
    with (
        mock.patch("jmapc.event_subscription.time.sleep") as sleep_mock,
        pytest.raises(StateEventsInvalidEventError),
    ):
        list(client.state_events(StateEventsConfig(max_retries=5)))
    mock_sseclient.assert_called_once()
    sleep_mock.assert_not_called()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_retries": -1}, "max_retries"),
        ({"initial_delay": 0}, "initial_delay"),
        ({"max_delay": 0}, "max_delay"),
        (
            {"max_delay": 0.5, "initial_delay": 1.0},
            "max_delay must be greater than or equal",
        ),
        ({"ping": -1}, "ping"),
    ],
)
def test_state_events_config_validation(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        StateEventsConfig(**kwargs)


def test_client_events_remains_compatible(
    client: Client,
    mock_sseclient: mock.MagicMock,
    http_responses: responses.RequestsMock,
) -> None:
    set_next_side_effect(
        mock_sseclient,
        [
            state_event("8001"),
            sseclient.Event(id="p1", event="ping", data="ignore-me"),
            state_event("8002"),
            StopIteration,
        ],
    )
    assert next(client.events) == expected_event("8001")
    assert next(client.events) == expected_event("8002")
    with pytest.raises(StopIteration):
        next(client.events)
    mock_sseclient.assert_called_once_with(
        EVENT_SOURCE_URL, auth=AUTH, last_id=None
    )
