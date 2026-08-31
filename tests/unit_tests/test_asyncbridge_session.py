from __future__ import annotations

import httpx
import pytest

import ui.main as main
import ui.utils as utils


class DummyBridge:
    def __init__(self) -> None:
        self.started = False
        self.closed = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True


def test_ensure_async_bridge_reuses_bridge_for_same_session(monkeypatch) -> None:
    created: list[DummyBridge] = []

    def factory() -> DummyBridge:
        bridge = DummyBridge()
        created.append(bridge)
        return bridge

    monkeypatch.setattr(main, "AsyncBridge", factory)
    session_state: dict[str, object] = {}

    bridge_1 = main.ensure_async_bridge(session_state, session_id="session-1")
    bridge_2 = main.ensure_async_bridge(session_state, session_id="session-1")

    assert bridge_1 is bridge_2
    assert len(created) == 1
    assert bridge_1.started is True
    assert session_state["async_bridge"] is bridge_1


def test_ensure_async_bridge_recreates_after_browser_refresh(monkeypatch) -> None:
    created: list[DummyBridge] = []

    def factory() -> DummyBridge:
        bridge = DummyBridge()
        created.append(bridge)
        return bridge

    monkeypatch.setattr(main, "AsyncBridge", factory)
    session_state: dict[str, object] = {}

    bridge_1 = main.ensure_async_bridge(session_state, session_id="session-1")
    bridge_2 = main.ensure_async_bridge(session_state, session_id="session-2")

    assert bridge_1 is not bridge_2
    assert len(created) == 2
    assert bridge_1.closed is True
    assert bridge_2.started is True
    assert session_state["async_bridge_session_id"] == "session-2"


def test_cleanup_async_bridge_removes_and_closes_bridge() -> None:
    bridge = DummyBridge()
    session_state: dict[str, object] = {"async_bridge": bridge, "async_bridge_session_id": "session-1"}

    main.cleanup_async_bridge(session_state)

    assert bridge.closed is True
    assert "async_bridge" not in session_state
    assert "async_bridge_session_id" not in session_state


def test_ensure_async_bridge_creates_even_when_handshake_has_not_succeeded(monkeypatch) -> None:
    created: list[DummyBridge] = []

    def factory() -> DummyBridge:
        bridge = DummyBridge()
        created.append(bridge)
        return bridge

    monkeypatch.setattr(main, "AsyncBridge", factory)
    session_state: dict[str, object] = {}

    bridge = main.ensure_async_bridge(session_state, session_id="session-1")

    assert bridge is not None
    assert len(created) == 1
    assert bridge.started is True
    assert session_state["async_bridge"] is bridge


def test_async_bridge_request_requires_initialized_loop() -> None:
    bridge = main.AsyncBridge()

    with pytest.raises(RuntimeError, match="AsyncBridge loop is not initialized"):
        bridge.request("GET", "https://example.com")


def test_make_handshake_uses_bridge_with_retry(monkeypatch) -> None:
    calls = {"count": 0}

    class DummyBridge:
        def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
            calls["count"] += 1
            if calls["count"] == 1:
                raise httpx.ConnectError("temporary failure", request=httpx.Request(method, url))
            return httpx.Response(200, request=httpx.Request(method, url))

    bridge = DummyBridge()
    response = utils.make_handshake(bridge)

    assert response.status_code == 200
    assert calls["count"] == 2
