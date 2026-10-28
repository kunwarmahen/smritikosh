"""Windows runtime compatibility for asyncio entry points.

asyncpg registers sockets with ``loop.add_reader``. The default Windows event
loop (``ProactorEventLoop``) does not implement that method, so ``alembic
upgrade head``, the worker, and any other ``asyncio.run`` entry point raise
``NotImplementedError``. Installing the selector policy before the loop is
created avoids that.

``uvicorn`` already selects the selector loop for its own process. Callers
that use ``asyncio.run`` directly still need this.
"""

from __future__ import annotations

import asyncio
import sys


def install_selector_event_loop() -> None:
    """Use a selector event loop on Windows so asyncpg can register sockets.

    No-op on macOS and Linux. No-op if this Python has no
    ``WindowsSelectorEventLoopPolicy`` (removed in a future release).
    Must run before the event loop is created.
    """
    if sys.platform != "win32":
        return
    policy_cls = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_cls is None:
        return
    asyncio.set_event_loop_policy(policy_cls())
