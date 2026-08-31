import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Generator

import httpx

CHATBOT_URL = "https://example.com"


class AsyncBridge:
    def __init__(self) -> None:
        self.loop: asyncio.AbstractEventLoop | None = None
        self.executor: ThreadPoolExecutor | None = None
        self.client: httpx.AsyncClient | None = None

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        if self.loop is None:
            raise RuntimeError("AsyncBridge loop is not initialized")
        return self.loop

    def start(self) -> None:
        """Start the dedicated event loop and initialize the shared async HTTP client."""
        if self.client is not None:
            return
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.executor.submit(self._run_loop)
        self.client = asyncio.run_coroutine_threadsafe(self._init_client(), self._require_loop()).result()

    def _run_loop(self) -> None:
        loop = self._require_loop()
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _init_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient()

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send an HTTP request via the dedicated async loop and return the synchronous HTTPX response."""
        loop = self._require_loop()
        future = asyncio.run_coroutine_threadsafe(self._request(method, url, **kwargs), loop)
        return future.result()

    async def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if self.client is None:
            raise RuntimeError("AsyncBridge client is not initialized")
        return await self.client.request(method, url, **kwargs)

    def is_valid(self) -> bool:
        return self.client is not None and self.loop is not None and not self.loop.is_closed()

    def close(self) -> None:
        """Close the HTTP client and shut down the background loop without crashing on partial initialization."""
        loop = self.loop
        if self.client is not None:
            try:
                if loop is not None:
                    asyncio.run_coroutine_threadsafe(self.client.aclose(), loop).result(timeout=5)
            except Exception:
                pass
            self.client = None

        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)

        if self.executor is not None:
            try:
                self.executor.shutdown(wait=True, cancel_futures=True)
            finally:
                self.executor = None

        self.loop = None


@contextmanager
def managed_async_bridge() -> Generator[AsyncBridge, Any, Any]:
    bridge = AsyncBridge()
    bridge.start()
    try:
        yield bridge
    finally:
        bridge.close()
