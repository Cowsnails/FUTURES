"""
Rate Limiting and Pacing Manager for IB Historical Data Requests

IB Gateway enforces strict pacing rules:
1. 15-second minimum between identical requests
2. Max 6 requests per 2 seconds per contract/exchange/tick type
3. Max 60 requests per 10-minute rolling window
4. BID_ASK requests count double
"""

import asyncio
import time
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
from dataclasses import dataclass
from ib_insync import Contract

logger = logging.getLogger(__name__)


@dataclass
class HistoricalRequest:
    """Represents a historical data request for pacing tracking"""
    contract_id: int
    end_datetime: str
    duration: str
    bar_size: str
    what_to_show: str
    timestamp: float

    def get_request_hash(self) -> str:
        """Generate unique hash for identical request detection"""
        return f"{self.contract_id}:{self.end_datetime}:{self.bar_size}:{self.what_to_show}"

    def get_contract_key(self) -> str:
        """Generate key for per-contract rate limiting"""
        return f"{self.contract_id}:{self.what_to_show}"


class PacingManager:
    """
    Manages rate limiting for IB historical data requests.

    Enforces:
    - Rule 1: 15-second minimum between identical requests
    - Rule 2: Max 6 requests per 2 seconds per contract
    - Rule 3: Max 60 requests per 10-minute global window
    """

    def __init__(
        self,
        identical_request_delay: int = 15,
        contract_window_seconds: int = 2,
        contract_max_requests: int = 6,
        global_window_seconds: int = 600,
        global_max_requests: int = 60
    ):
        """
        Initialize pacing manager.

        Args:
            identical_request_delay: Seconds between identical requests (default: 15)
            contract_window_seconds: Window for per-contract limit (default: 2)
            contract_max_requests: Max requests per contract window (default: 6)
            global_window_seconds: Global rolling window (default: 600 = 10 min)
            global_max_requests: Max requests in global window (default: 60)
        """
        self.identical_request_delay = identical_request_delay
        self.contract_window_seconds = contract_window_seconds
        self.contract_max_requests = contract_max_requests
        self.global_window_seconds = global_window_seconds
        self.global_max_requests = global_max_requests

        # Track identical requests by hash
        self.last_request_times: Dict[str, float] = {}

        # Track per-contract requests
        self.contract_requests: Dict[str, deque] = defaultdict(lambda: deque())

        # Track all requests globally
        self.all_requests: deque = deque()

        # Statistics
        self.total_requests = 0
        self.total_delays = 0
        self.total_delay_time = 0.0

    async def wait_if_needed(self, request: HistoricalRequest) -> float:
        """
        Wait if necessary to comply with pacing rules.

        Args:
            request: Historical data request to check

        Returns:
            Time waited in seconds (0 if no wait needed)
        """
        now = time.time()
        wait_time = self._calculate_required_wait(request, now)

        if wait_time > 0:
            self.total_delays += 1
            self.total_delay_time += wait_time
            logger.debug(f"Pacing: waiting {wait_time:.1f}s before request")
            await asyncio.sleep(wait_time)

        # Record this request
        self._record_request(request, now + wait_time)
        self.total_requests += 1

        return wait_time

    def _calculate_required_wait(self, request: HistoricalRequest, now: float) -> float:
        """Calculate how long to wait before this request can be made"""
        wait_times = []

        # Rule 1: Check identical request timing
        request_hash = request.get_request_hash()
        if request_hash in self.last_request_times:
            last_time = self.last_request_times[request_hash]
            elapsed = now - last_time
            if elapsed < self.identical_request_delay:
                wait_times.append(self.identical_request_delay - elapsed)

        # Rule 2: Check per-contract rate limit
        contract_key = request.get_contract_key()
        contract_times = self.contract_requests[contract_key]

        # Count requests in the window
        window_start = now - self.contract_window_seconds
        recent_requests = [t for t in contract_times if t > window_start]

        if len(recent_requests) >= self.contract_max_requests:
            # Need to wait until oldest request falls out of window
            oldest = min(recent_requests)
            wait_times.append(oldest + self.contract_window_seconds - now)

        # Rule 3: Check global rate limit
        global_window_start = now - self.global_window_seconds
        recent_global = [t for t in self.all_requests if t > global_window_start]

        if len(recent_global) >= self.global_max_requests:
            # Need to wait until oldest request falls out of window
            oldest = min(recent_global)
            wait_times.append(oldest + self.global_window_seconds - now)

        # Return maximum wait time needed
        return max(wait_times) if wait_times else 0

    def _record_request(self, request: HistoricalRequest, timestamp: float):
        """Record that a request was made"""
        # Record for identical request tracking
        request_hash = request.get_request_hash()
        self.last_request_times[request_hash] = timestamp

        # Record for per-contract tracking
        contract_key = request.get_contract_key()
        self.contract_requests[contract_key].append(timestamp)

        # Clean up old entries (keep only those within window)
        window_start = timestamp - self.contract_window_seconds - 1
        while (self.contract_requests[contract_key] and
               self.contract_requests[contract_key][0] < window_start):
            self.contract_requests[contract_key].popleft()

        # Record for global tracking
        self.all_requests.append(timestamp)

        # Clean up old global entries
        global_window_start = timestamp - self.global_window_seconds - 1
        while self.all_requests and self.all_requests[0] < global_window_start:
            self.all_requests.popleft()

    def can_send_now(self, request: HistoricalRequest) -> Tuple[bool, float]:
        """
        Check if a request can be sent now without waiting.

        Args:
            request: Historical data request to check

        Returns:
            Tuple of (can_send, wait_time_if_not)
        """
        now = time.time()
        wait_time = self._calculate_required_wait(request, now)
        return (wait_time == 0, wait_time)

    def get_statistics(self) -> Dict:
        """Get pacing statistics"""
        return {
            'total_requests': self.total_requests,
            'total_delays': self.total_delays,
            'total_delay_time_seconds': self.total_delay_time,
            'average_delay_seconds': (
                self.total_delay_time / self.total_delays
                if self.total_delays > 0 else 0
            ),
            'current_global_requests': len(self.all_requests),
            'global_capacity_remaining': (
                self.global_max_requests - len(self.all_requests)
            )
        }

    def reset_statistics(self):
        """Reset statistics counters"""
        self.total_requests = 0
        self.total_delays = 0
        self.total_delay_time = 0.0


class RequestQueue:
    """
    Queue for historical data requests with automatic pacing.

    Processes requests sequentially with automatic pacing delays.
    """

    def __init__(self, pacing_manager: Optional[PacingManager] = None):
        """
        Initialize request queue.

        Args:
            pacing_manager: PacingManager instance (creates new if None)
        """
        self.pacing_manager = pacing_manager or PacingManager()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.processing = False
        self.results: Dict[str, any] = {}

    async def add_request(
        self,
        request: HistoricalRequest,
        request_func: callable,
        *args,
        **kwargs
    ) -> str:
        """
        Add a request to the queue.

        Args:
            request: HistoricalRequest for pacing tracking
            request_func: Async function to call
            *args, **kwargs: Arguments for request_func

        Returns:
            Request ID for retrieving results
        """
        request_id = request.get_request_hash()

        await self.queue.put({
            'request': request,
            'func': request_func,
            'args': args,
            'kwargs': kwargs,
            'id': request_id
        })

        return request_id

    async def process_queue(self):
        """Process queued requests with automatic pacing"""
        self.processing = True

        logger.info("Request queue processing started")

        while self.processing:
            try:
                # Get next request (with timeout to allow checking processing flag)
                try:
                    item = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                request = item['request']
                func = item['func']
                args = item['args']
                kwargs = item['kwargs']
                request_id = item['id']

                # Wait for pacing if needed
                wait_time = await self.pacing_manager.wait_if_needed(request)

                if wait_time > 0:
                    logger.info(f"Paced request by {wait_time:.1f}s")

                # Execute request
                try:
                    result = await func(*args, **kwargs)
                    self.results[request_id] = {'success': True, 'data': result}
                    logger.debug(f"Request {request_id[:20]}... completed successfully")

                except Exception as e:
                    self.results[request_id] = {'success': False, 'error': str(e)}
                    logger.error(f"Request {request_id[:20]}... failed: {e}")

                self.queue.task_done()

            except Exception as e:
                logger.error(f"Error processing queue: {e}")

        logger.info("Request queue processing stopped")

    def stop_processing(self):
        """Stop processing the queue"""
        self.processing = False

    async def wait_for_completion(self):
        """Wait for all queued requests to complete"""
        await self.queue.join()

    def get_result(self, request_id: str) -> Optional[Dict]:
        """Get result for a request"""
        return self.results.get(request_id)


if __name__ == '__main__':
    # Example usage and testing
    async def test_pacing():
        logging.basicConfig(level=logging.INFO)

        manager = PacingManager()

        # Simulate multiple requests
        print("Testing pacing manager...")

        for i in range(10):
            request = HistoricalRequest(
                contract_id=123456,
                end_datetime='',
                duration='1 D',
                bar_size='1 min',
                what_to_show='TRADES',
                timestamp=time.time()
            )

            wait_time = await manager.wait_if_needed(request)
            print(f"Request {i+1}: waited {wait_time:.2f}s")

        stats = manager.get_statistics()
        print(f"\nStatistics: {stats}")

    asyncio.run(test_pacing())
