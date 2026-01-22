"""
IB Gateway Connection Service

Handles connection management, error recovery, and connection state tracking
for Interactive Brokers Gateway.
"""

import asyncio
import logging
import random
import sys
import time
from enum import Enum
from typing import Optional, Callable, Set, Any
from ib_insync import IB, Contract
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection state machine states"""
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2        # Socket connected
    READY = 3            # Data farms OK
    SUBSCRIBED = 4       # Active market data
    IB_DISCONNECTED = 5  # 1100 received (IB server issue)


# Error code classifications
RETRYABLE_ERRORS = {162, 366, 1100, 2103, 2105}  # Pacing, connection issues
FATAL_ERRORS = {200, 354, 502}  # Bad contract, no subscription, can't connect
INFORMATIONAL_CODES = {2104, 2106, 2158}  # Farm connection OK - not errors


class IBConnectionManager:
    """
    Manages connection to IB Gateway with error handling and automatic recovery.

    Features:
    - Automatic reconnection with exponential backoff
    - Connection state tracking
    - Error code classification and handling
    - Re-subscription after connection loss
    - Health monitoring
    - Windows compatibility with event loop handling
    """

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 4002,
        client_id: int = 1,
        timeout: int = 10
    ):
        """
        Initialize connection manager.

        Args:
            host: IB Gateway host (default: 127.0.0.1)
            port: IB Gateway port (4002=paper, 4001=live)
            client_id: Unique client ID (1-32)
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout

        self.ib = IB()
        self.state = ConnectionState.DISCONNECTED
        self.ib_server_connected = True

        # Track subscriptions for re-subscription after 1101
        self.active_subscriptions: Set[Contract] = set()
        self.subscription_callbacks = {}

        # Connection metrics
        self.connection_attempts = 0
        self.last_connection_time: Optional[datetime] = None
        self.last_error_time: Optional[datetime] = None
        self.last_tick_time: Optional[datetime] = None
        self.reconnection_count = 0

        # Setup error handlers
        self.ib.errorEvent += self._on_error
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.connectedEvent += self._on_connected

    async def connect(self) -> bool:
        """
        Connect to IB Gateway with retry logic.

        Returns:
            True if connected successfully, False otherwise
        """
        if self.ib.isConnected():
            logger.info("Already connected to IB Gateway")
            return True

        max_retries = 5
        base_delay = 2.0

        for attempt in range(max_retries):
            try:
                self.state = ConnectionState.CONNECTING
                self.connection_attempts += 1

                logger.info(
                    f"Connecting to IB Gateway at {self.host}:{self.port} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )

                # Windows: Use synchronous connect to avoid event loop issues
                if sys.platform == 'win32':
                    # Run synchronous connect in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: self.ib.connect(
                            self.host,
                            self.port,
                            clientId=self.client_id,
                            timeout=self.timeout
                        )
                    )
                else:
                    # Linux/Mac: Use async connect
                    await self.ib.connectAsync(
                        host=self.host,
                        port=self.port,
                        clientId=self.client_id,
                        timeout=self.timeout
                    )

                if self.ib.isConnected():
                    self.state = ConnectionState.CONNECTED
                    self.last_connection_time = datetime.now()

                    # Safely get server version
                    try:
                        if hasattr(self.ib, 'client') and hasattr(self.ib.client, 'serverVersion'):
                            version = self.ib.client.serverVersion
                            logger.info(f"✓ Connected to IB Gateway! Server version: {version}")
                        else:
                            logger.info("✓ Connected to IB Gateway successfully!")
                    except Exception as ve:
                        logger.info("✓ Connected to IB Gateway successfully!")

                    return True

            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt), 60.0)
                    jitter = delay * (0.75 + random.random() * 0.5)
                    logger.info(f"Retrying in {jitter:.1f} seconds...")
                    await asyncio.sleep(jitter)

        self.state = ConnectionState.DISCONNECTED
        logger.error("Failed to connect after all retry attempts")
        return False

    def disconnect(self):
        """Disconnect from IB Gateway"""
        if self.ib.isConnected():
            logger.info("Disconnecting from IB Gateway")
            self.ib.disconnect()
            self.state = ConnectionState.DISCONNECTED
            self.active_subscriptions.clear()

    def is_connected(self) -> bool:
        """Check if connected to IB Gateway"""
        return self.ib.isConnected() and self.ib_server_connected

    def get_state(self) -> ConnectionState:
        """Get current connection state"""
        return self.state

    def _on_connected(self):
        """Handle connection established event"""
        logger.info("Connection established")
        self.state = ConnectionState.CONNECTED
        self.last_connection_time = datetime.now()

    def _on_disconnected(self):
        """Handle disconnection event"""
        logger.warning("Disconnected from IB Gateway")
        self.state = ConnectionState.DISCONNECTED
        self.ib_server_connected = False

        # Schedule reconnection
        asyncio.create_task(self._reconnect())

    async def _reconnect(self):
        """Automatic reconnection logic"""
        logger.info("Attempting automatic reconnection...")
        self.reconnection_count += 1

        # Wait before reconnecting
        await asyncio.sleep(5)

        success = await self.connect()

        if success and self.active_subscriptions:
            logger.info("Reconnected successfully, re-subscribing to market data...")
            await self._resubscribe_all()

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Any):
        """
        Handle IB error messages.

        Critical error codes:
        - 1100: Connection lost
        - 1101: Connection restored, data lost (must re-subscribe)
        - 1102: Connection restored, data maintained
        - 2103/2105: Farm disconnected
        - 162: Pacing violation
        - 354: Not subscribed to market data
        """
        self.last_error_time = datetime.now()

        # Informational codes - not errors
        if errorCode in INFORMATIONAL_CODES:
            logger.debug(f"Info {errorCode}: {errorString}")
            return

        # Connection lost
        if errorCode == 1100:
            logger.error(f"ERROR 1100: IB server connection lost - {errorString}")
            self.ib_server_connected = False
            self.state = ConnectionState.IB_DISCONNECTED

        # Connection restored, data lost - must re-subscribe
        elif errorCode == 1101:
            logger.warning(f"ERROR 1101: Connection restored, data lost - {errorString}")
            self.ib_server_connected = True
            self.state = ConnectionState.CONNECTED

            # Critical: Re-subscribe to all market data
            asyncio.create_task(self._resubscribe_all())

        # Connection restored, data maintained
        elif errorCode == 1102:
            logger.info(f"ERROR 1102: Connection restored, data maintained - {errorString}")
            self.ib_server_connected = True

            if self.state == ConnectionState.IB_DISCONNECTED:
                self.state = ConnectionState.SUBSCRIBED

        # Farm disconnections (usually temporary)
        elif errorCode in {2103, 2105}:
            logger.warning(f"ERROR {errorCode}: Data farm disconnected - {errorString}")
            # Usually resolves automatically

        # Pacing violation
        elif errorCode == 162:
            logger.error(f"ERROR 162: Pacing violation - {errorString}")
            # Caller should implement retry with delay

        # Not subscribed to market data
        elif errorCode == 354:
            logger.error(
                f"ERROR 354: Not subscribed to market data - {errorString}\n"
                f"  Action required:\n"
                f"  1. Subscribe to CME Real-Time data in Client Portal\n"
                f"  2. Complete API Acknowledgement form\n"
                f"  3. Enable paper account data sharing (24h wait required)"
            )

        # Other errors
        elif errorCode in RETRYABLE_ERRORS:
            logger.warning(f"Retryable error {errorCode}: {errorString}")
        elif errorCode in FATAL_ERRORS:
            logger.error(f"Fatal error {errorCode}: {errorString}")
        else:
            logger.error(f"Error {errorCode}: {errorString}")

    def track_subscription(self, contract: Contract, callback: Optional[Callable] = None):
        """
        Track a market data subscription for re-subscription after disconnect.

        Args:
            contract: Contract being subscribed to
            callback: Optional callback to invoke after re-subscription
        """
        self.active_subscriptions.add(contract)
        if callback:
            self.subscription_callbacks[contract] = callback

        logger.debug(f"Tracking subscription for {contract.symbol}")

    def untrack_subscription(self, contract: Contract):
        """Stop tracking a market data subscription"""
        self.active_subscriptions.discard(contract)
        self.subscription_callbacks.pop(contract, None)
        logger.debug(f"Untracking subscription for {contract.symbol}")

    async def _resubscribe_all(self):
        """Re-subscribe to all tracked market data after connection loss"""
        if not self.active_subscriptions:
            logger.info("No active subscriptions to restore")
            return

        logger.info(f"Re-subscribing to {len(self.active_subscriptions)} contracts...")

        for contract in self.active_subscriptions:
            try:
                callback = self.subscription_callbacks.get(contract)
                if callback:
                    await callback(contract)
                    logger.info(f"Re-subscribed to {contract.symbol}")
                else:
                    logger.warning(
                        f"No callback for {contract.symbol}, cannot re-subscribe"
                    )
            except Exception as e:
                logger.error(f"Failed to re-subscribe to {contract.symbol}: {e}")

        self.state = ConnectionState.SUBSCRIBED

    def update_tick_time(self):
        """Update last tick received time (for health monitoring)"""
        self.last_tick_time = datetime.now()

    def get_health_status(self) -> dict:
        """
        Get connection health status.

        Returns:
            Dictionary with health metrics
        """
        now = datetime.now()

        health = {
            'connected': self.is_connected(),
            'state': self.state.name,
            'connection_attempts': self.connection_attempts,
            'reconnection_count': self.reconnection_count,
            'active_subscriptions': len(self.active_subscriptions),
        }

        if self.last_connection_time:
            uptime = (now - self.last_connection_time).total_seconds()
            health['uptime_seconds'] = uptime

        if self.last_tick_time:
            tick_age = (now - self.last_tick_time).total_seconds()
            health['last_tick_age_seconds'] = tick_age
            health['data_stale'] = tick_age > 30  # No tick in 30 seconds

        if self.last_error_time:
            error_age = (now - self.last_error_time).total_seconds()
            health['last_error_age_seconds'] = error_age

        return health


async def ib_request_with_retry(
    func: Callable,
    *args,
    max_retries: int = 5,
    base_delay: float = 2.0,
    **kwargs
) -> Any:
    """
    Execute an IB request with exponential backoff retry logic.

    Args:
        func: Async function to call
        *args: Positional arguments for func
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff
        **kwargs: Keyword arguments for func

    Returns:
        Result from func

    Raises:
        Exception if all retries fail
    """
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            error_str = str(e)

            # Extract error code if present
            error_code = None
            if 'error' in error_str.lower():
                try:
                    # Try to extract error code (format: "Error 162: ...")
                    parts = error_str.split()
                    for i, part in enumerate(parts):
                        if part.lower() == 'error' and i + 1 < len(parts):
                            error_code = int(parts[i + 1].rstrip(':'))
                            break
                except:
                    pass

            # Check if error is fatal
            if error_code in FATAL_ERRORS:
                logger.error(f"Fatal error {error_code}, not retrying: {e}")
                raise

            # Check if error is retryable
            if error_code and error_code not in RETRYABLE_ERRORS:
                if attempt == max_retries:
                    raise

            # Last attempt - raise exception
            if attempt == max_retries:
                logger.error(f"Request failed after {max_retries} retries: {e}")
                raise

            # Calculate delay with exponential backoff and jitter
            delay = min(base_delay * (2 ** attempt), 60.0)
            jitter = delay * (0.75 + random.random() * 0.5)

            # Special handling for pacing violations
            if error_code == 162 or 'pacing' in error_str.lower():
                delay = max(jitter, 15.0)  # Minimum 15 second wait
                logger.warning(f"Pacing violation, waiting {delay:.1f}s before retry")
            else:
                logger.warning(
                    f"Request failed (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {jitter:.1f}s: {e}"
                )

            await asyncio.sleep(delay)


if __name__ == '__main__':
    # Example usage
    async def test():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        manager = IBConnectionManager(host='127.0.0.1', port=4002, client_id=1)

        # Connect
        success = await manager.connect()

        if success:
            print("Connection successful!")
            print(f"Health status: {manager.get_health_status()}")

            # Keep alive for a bit
            await asyncio.sleep(5)

            # Disconnect
            manager.disconnect()
        else:
            print("Connection failed!")

    asyncio.run(test())
