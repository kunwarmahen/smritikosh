"""Windows selector event-loop policy."""

import asyncio

from smritikosh.platform import install_selector_event_loop


class _FakeSelectorPolicy(asyncio.DefaultEventLoopPolicy):
    """Stand-in for asyncio.WindowsSelectorEventLoopPolicy."""


def test_noop_off_windows(monkeypatch):
    monkeypatch.setattr("smritikosh.platform.sys.platform", "darwin")
    monkeypatch.setattr(
        asyncio, "WindowsSelectorEventLoopPolicy", _FakeSelectorPolicy, raising=False
    )
    original = asyncio.get_event_loop_policy()
    install_selector_event_loop()
    assert asyncio.get_event_loop_policy() is original


def test_installs_policy_on_windows(monkeypatch):
    monkeypatch.setattr("smritikosh.platform.sys.platform", "win32")
    monkeypatch.setattr(
        asyncio, "WindowsSelectorEventLoopPolicy", _FakeSelectorPolicy, raising=False
    )
    original = asyncio.get_event_loop_policy()
    try:
        install_selector_event_loop()
        assert isinstance(asyncio.get_event_loop_policy(), _FakeSelectorPolicy)
    finally:
        asyncio.set_event_loop_policy(original)


def test_noop_when_policy_class_is_missing(monkeypatch):
    monkeypatch.setattr("smritikosh.platform.sys.platform", "win32")
    monkeypatch.delattr(asyncio, "WindowsSelectorEventLoopPolicy", raising=False)
    original = asyncio.get_event_loop_policy()
    install_selector_event_loop()
    assert asyncio.get_event_loop_policy() is original
