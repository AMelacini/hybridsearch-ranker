import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Generator

import httpx

CHATBOT_URL = "https://example.com"


class AsyncBridge:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.client: httpx.AsyncClient | None = None

    def start(self) -> None:
        """Start the dedicated event loop and initialize the shared async HTTP client."""
        if self.client is not None:
            return
        self.executor.submit(self._run_loop)
        self.client = asyncio.run_coroutine_threadsafe(self._init_client(), self.loop).result()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _init_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient()

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send an HTTP request via the dedicated async loop and return the synchronous HTTPX response."""
        future = asyncio.run_coroutine_threadsafe(self._request(method, url, **kwargs), self.loop)
        return future.result()

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self.client is None:
            raise RuntimeError("AsyncBridge client is not initialized")
        return await self.client.request(method, url, **kwargs)

    def is_valid(self) -> bool:
        return self.client is not None and self.loop is not None and not self.loop.is_closed()

    def close(self) -> None:
        """Close the HTTP client and shut down the background loop without crashing on partial initialization."""
        if self.client is not None:
            try:
                asyncio.run_coroutine_threadsafe(self.client.aclose(), self.loop).result(timeout=5)
            except Exception:
                pass
            self.client = None

        if self.loop is not None and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)


@contextmanager
def managed_async_bridge() -> Generator[AsyncBridge, Any, Any]:
    bridge = AsyncBridge()
    bridge.start()
    try:
        yield bridge
    finally:
        bridge.close()
