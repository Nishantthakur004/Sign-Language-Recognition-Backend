from __future__ import annotations

import asyncio
import math
import time
from collections import OrderedDict, deque


class SlidingWindowRateLimiter:
    def __init__(self, limit: int, window_seconds: float, max_keys: int = 10_000):
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def hit(self, key: str) -> int:
        """Record an attempt and return a Retry-After value when limited."""
        now = time.monotonic()
        async with self._lock:
            attempts = self._requests.setdefault(key, deque())
            while attempts and now - attempts[0] >= self.window_seconds:
                attempts.popleft()
            if len(attempts) >= self.limit:
                self._requests.move_to_end(key)
                return max(1, math.ceil(self.window_seconds - (now - attempts[0])))
            attempts.append(now)
            self._requests.move_to_end(key)
            while len(self._requests) > self.max_keys:
                self._requests.popitem(last=False)
            return 0
