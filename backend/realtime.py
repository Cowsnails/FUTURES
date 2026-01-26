"""
Real-Time Data Streaming

Provides real-time candlestick updates using two approaches:
1. Tick-by-tick streaming (true real-time, 50-300ms latency)
2. KeepUpToDate streaming (simpler, 5-second updates)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Dict, Any
from collections import deque
from ib_insync import IB, Contract, Ticker, TickByTickAllLast
import pytz

logger = logging.getLogger(__name__)


class LiveCandlestickBuilder:
    """
    Builds live candlesticks from tick-by-tick data.

    This is the TRUE real-time approach with 50-300ms latency.
    Use this when sub-second updates are critical.
    """

    def __init__(
        self,
        ib: IB,
        contract: Contract,
        bar_size_minutes: int = 1,
        on_bar_callback: Optional[Callable] = None
    ):
        """
        Initialize live candlestick builder.

        Args:
            ib: Connected IB instance
            contract: Futures contract to stream
            bar_size_minutes: Candlestick duration in minutes (default: 1)
            on_bar_callback: Async callback(bar_data: dict, is_new_bar: bool)
        """
        self.ib = ib
        self.contract = contract
        self.bar_size = timedelta(minutes=bar_size_minutes)
        self.on_bar = on_bar_callback
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        self.ticker: Optional[Ticker] = None
        self.current_bar: Optional[Dict[str, Any]] = None
        self.current_bar_start_time: Optional[datetime] = None

        self.stats = {
            'ticks_processed': 0,
            'bars_completed': 0,
            'bars_updated': 0,
        }

    async def start(self):
        """Start streaming tick-by-tick data"""
        logger.info(f"[{self.contract.symbol}] Starting tick-by-tick stream...")

        try:
            # Store event loop reference for scheduling async callbacks
            self.loop = asyncio.get_event_loop()
            logger.debug(f"[{self.contract.symbol}] Event loop stored: {self.loop}")

            # Check contract is qualified
            if not self.contract.conId or self.contract.conId == 0:
                raise Exception(
                    f"Contract not qualified! conId={self.contract.conId}, "
                    f"localSymbol={self.contract.localSymbol}. "
                    f"Call qualifyContractsAsync() first."
                )

            # Request tick-by-tick data
            # 'AllLast' captures all trade types (more comprehensive than 'Last')
            logger.info(
                f"[{self.contract.symbol}] Requesting tick-by-tick data from IB Gateway... "
                f"(conId: {self.contract.conId}, localSymbol: {self.contract.localSymbol}, "
                f"exchange: {self.contract.exchange})"
            )

            self.ticker = self.ib.reqTickByTickData(
                self.contract,
                'AllLast',
                numberOfTicks=0,  # 0 = continuous stream
                ignoreSize=False
            )

            logger.info(
                f"[{self.contract.symbol}] reqTickByTickData() returned ticker object: {self.ticker}"
            )

            if not self.ticker:
                raise Exception("reqTickByTickData returned None - subscription failed")

            # Check if ticker has tickByTicks attribute
            logger.info(
                f"[{self.contract.symbol}] Ticker attributes: {dir(self.ticker)}"
            )
            logger.info(
                f"[{self.contract.symbol}] Initial tickByTicks: {self.ticker.tickByTicks}"
            )

            # Subscribe to tick updates
            logger.info(f"[{self.contract.symbol}] Subscribing to ticker.updateEvent...")
            self.ticker.updateEvent += self._on_tick

            # CRITICAL: Give ib_insync a moment to process the subscription
            await asyncio.sleep(0.01)  # Reduced from 100ms to 10ms for lower latency

            # Check if updateEvent has subscribers
            logger.info(
                f"[{self.contract.symbol}] updateEvent subscribers: "
                f"{len(self.ticker.updateEvent) if hasattr(self.ticker.updateEvent, '__len__') else 'unknown'}"
            )

            logger.info(
                f"✓ [{self.contract.symbol}] Tick-by-tick stream started successfully! "
                f"(expected latency: 50-300ms) - waiting for ticks..."
            )

        except Exception as e:
            logger.error(f"❌ [{self.contract.symbol}] Failed to start tick-by-tick stream: {e}", exc_info=True)
            raise

    def stop(self):
        """Stop streaming"""
        if self.ticker:
            logger.info(f"Stopping tick-by-tick stream for {self.contract.symbol}")
            self.ib.cancelTickByTickData(self.contract, 'AllLast')
            self.ticker = None

    def _on_tick(self, ticker: Ticker):
        """
        Handle incoming tick data.

        Called on every trade (multiple times per second during active trading).
        """
        try:
            # Log first tick received to confirm stream is working
            if self.stats['ticks_processed'] == 0:
                logger.info(f"[{self.contract.symbol}] ✓ First tick received! Stream is live.")

            # Process new ticks
            if not ticker.tickByTicks or len(ticker.tickByTicks) == 0:
                logger.debug(f"[{self.contract.symbol}] No ticks in update")
                return

            for tick in ticker.tickByTicks:
                if not isinstance(tick, TickByTickAllLast):
                    continue

                self._process_tick(tick.price, tick.size, tick.time)
        except Exception as e:
            logger.error(f"[{self.contract.symbol}] Error in _on_tick callback: {e}", exc_info=True)

    def _process_tick(self, price: float, size: int, tick_time: datetime):
        """Process a single tick and update current bar"""
        self.stats['ticks_processed'] += 1

        # Log tick processing stats every 100 ticks
        if self.stats['ticks_processed'] % 100 == 0:
            logger.debug(
                f"[{self.contract.symbol}] Processed {self.stats['ticks_processed']} ticks, "
                f"{self.stats['bars_completed']} bars completed"
            )

        # Determine which bar this tick belongs to
        bar_start = self._get_bar_start_time(tick_time)

        # Determine if this is a new bar
        is_new_bar = False

        # Check if we need to finalize previous bar and start new one
        if self.current_bar_start_time and bar_start > self.current_bar_start_time:
            # Finalize current bar
            self._finalize_bar(is_new_bar=True)

            # Start new bar
            self._start_new_bar(bar_start, price, size)
            is_new_bar = True

        elif self.current_bar is None:
            # Initialize first bar
            self._start_new_bar(bar_start, price, size)
            is_new_bar = True

        else:
            # Update current bar
            self._update_bar(price, size)
            is_new_bar = False

        # Send update to callback (schedule async callback on event loop)
        if self.on_bar and self.current_bar and self.loop:
            bar_data = self.current_bar.copy()

            # Schedule the async callback on the event loop
            # This avoids "event loop already running" errors on Windows
            asyncio.run_coroutine_threadsafe(
                self.on_bar(bar_data, is_new_bar),
                self.loop
            )

            self.stats['bars_updated'] += 1

    def _get_bar_start_time(self, tick_time: datetime) -> datetime:
        """
        Get the start time of the bar this tick belongs to.

        IB's tick times are CORRECT - use them!
        """
        # Use IB's tick time - it's correct!
        # Round down to nearest bar_size interval
        bar_size_seconds = int(self.bar_size.total_seconds())
        timestamp = int(tick_time.timestamp())
        bar_timestamp = (timestamp // bar_size_seconds) * bar_size_seconds

        # Return timezone-aware datetime in UTC
        return datetime.fromtimestamp(bar_timestamp, tz=pytz.UTC)

    def _start_new_bar(self, bar_start: datetime, price: float, size: int):
        """Initialize a new candlestick bar"""
        self.current_bar_start_time = bar_start

        # Convert to Eastern time for display
        eastern = pytz.timezone('US/Eastern')
        bar_eastern = bar_start.astimezone(eastern)

        # Create "display timestamp" - treat Eastern time as if it were UTC
        display_time = datetime(
            bar_eastern.year,
            bar_eastern.month,
            bar_eastern.day,
            bar_eastern.hour,
            bar_eastern.minute,
            bar_eastern.second,
            tzinfo=pytz.UTC
        )

        self.current_bar = {
            'time': int(display_time.timestamp()),
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': size
        }

        logger.info(
            f"[{self.contract.symbol}] New bar started at {bar_eastern.strftime('%H:%M:%S %Z')} "
            f"| Open: {price:.2f}"
        )

    def _update_bar(self, price: float, size: int):
        """Update the current bar with new tick data"""
        if not self.current_bar:
            return

        self.current_bar['high'] = max(self.current_bar['high'], price)
        self.current_bar['low'] = min(self.current_bar['low'], price)
        self.current_bar['close'] = price
        self.current_bar['volume'] += size

    def _finalize_bar(self, is_new_bar: bool):
        """Finalize the current bar"""
        if self.current_bar:
            self.stats['bars_completed'] += 1
            bar_time = datetime.fromtimestamp(self.current_bar['time'])
            logger.info(
                f"[{self.contract.symbol}] Bar completed at {bar_time.strftime('%H:%M:%S')} "
                f"| O: {self.current_bar['open']:.2f} "
                f"H: {self.current_bar['high']:.2f} "
                f"L: {self.current_bar['low']:.2f} "
                f"C: {self.current_bar['close']:.2f} "
                f"V: {self.current_bar['volume']}"
            )

    def get_statistics(self) -> dict:
        """Get streaming statistics"""
        return self.stats.copy()


class KeepUpToDateStreamer:
    """
    Real-time streaming using keepUpToDate=True.

    This is the SIMPLER approach with ~5-second update intervals.
    Sufficient for most trading use cases.
    """

    def __init__(
        self,
        ib: IB,
        contract: Contract,
        bar_size: str = '1 min',
        initial_duration: str = '1 D',
        on_bar_callback: Optional[Callable] = None
    ):
        """
        Initialize keepUpToDate streamer.

        Args:
            ib: Connected IB instance
            contract: Futures contract
            bar_size: Bar size (e.g., '1 min', '5 mins')
            initial_duration: Initial historical load (e.g., '1 D')
            on_bar_callback: Callback(bar_data: dict, is_new_bar: bool)
        """
        self.ib = ib
        self.contract = contract
        self.bar_size = bar_size
        self.initial_duration = initial_duration
        self.on_bar = on_bar_callback

        self.bars = None
        self.stats = {
            'updates_received': 0,
            'bars_completed': 0,
        }

    async def start(self):
        """Start streaming with keepUpToDate"""
        logger.info(
            f"Starting keepUpToDate stream for {self.contract.symbol} "
            f"({self.bar_size} bars, ~5s update interval)"
        )

        try:
            # Request historical data with keepUpToDate=True
            # Must use async method for proper event loop integration
            self.bars = await self.ib.reqHistoricalDataAsync(
                self.contract,
                endDateTime='',
                durationStr=self.initial_duration,
                barSizeSetting=self.bar_size,
                whatToShow='TRADES',
                useRTH=False,  # Include extended hours for futures
                keepUpToDate=True  # This enables real-time updates
            )

            # Subscribe to bar updates
            self.bars.updateEvent += self._on_update

            logger.info(
                f"KeepUpToDate stream started for {self.contract.symbol} "
                f"(loaded {len(self.bars)} historical bars)"
            )

            return self.bars

        except Exception as e:
            logger.error(f"Failed to start keepUpToDate stream: {e}")
            raise

    def stop(self):
        """Stop streaming"""
        if self.bars:
            logger.info(f"Stopping keepUpToDate stream for {self.contract.symbol}")
            self.ib.cancelHistoricalData(self.bars)
            self.bars = None

    def _on_update(self, bars, hasNewBar: bool):
        """
        Handle bar updates.

        Called approximately every 5 seconds with updated bar data.

        Args:
            bars: BarDataList with all bars
            hasNewBar: True if a new bar was added, False if last bar was updated
        """
        self.stats['updates_received'] += 1

        if hasNewBar:
            self.stats['bars_completed'] += 1

        # Get the most recent bar
        current_bar = bars[-1]

        # Convert to standard format
        bar_data = self._convert_bar(current_bar)

        # Send to callback
        if self.on_bar:
            self.on_bar(bar_data, is_new_bar=hasNewBar)

    def _convert_bar(self, bar) -> Dict[str, Any]:
        """
        Convert IB bar to standard format with Eastern time display.

        CRITICAL: Convert to Eastern time for chart display.
        """
        eastern = pytz.timezone('US/Eastern')

        # Parse IB's timestamp
        if isinstance(bar.date, datetime):
            bar_time = bar.date
        else:
            # String format: 'YYYYMMDD  HH:MM:SS'
            naive_dt = datetime.strptime(bar.date, '%Y%m%d  %H:%M:%S')
            bar_time = pytz.UTC.localize(naive_dt)

        # Convert to Eastern time
        bar_eastern = bar_time.astimezone(eastern)

        # Create "display timestamp" - treat Eastern time as if it were UTC
        display_time = datetime(
            bar_eastern.year,
            bar_eastern.month,
            bar_eastern.day,
            bar_eastern.hour,
            bar_eastern.minute,
            bar_eastern.second,
            tzinfo=pytz.UTC
        )
        timestamp = int(display_time.timestamp())

        return {
            'time': timestamp,
            'open': float(bar.open),
            'high': float(bar.high),
            'low': float(bar.low),
            'close': float(bar.close),
            'volume': int(bar.volume)
        }

    def get_statistics(self) -> dict:
        """Get streaming statistics"""
        return self.stats.copy()


class RealtimeManager:
    """
    Manages real-time streaming for multiple contracts.

    Provides a unified interface for both streaming approaches.
    """

    def __init__(
        self,
        ib: IB,
        use_tick_by_tick: bool = False,
        bar_size_minutes: int = 1
    ):
        """
        Initialize realtime manager.

        Args:
            ib: Connected IB instance
            use_tick_by_tick: True for tick-by-tick, False for keepUpToDate
            bar_size_minutes: Candlestick duration in minutes
        """
        self.ib = ib
        self.use_tick_by_tick = use_tick_by_tick
        self.bar_size_minutes = bar_size_minutes

        self.streamers: Dict[str, Any] = {}

        logger.info(
            f"RealtimeManager initialized "
            f"(mode: {'tick-by-tick' if use_tick_by_tick else 'keepUpToDate'})"
        )

        if use_tick_by_tick:
            logger.warning(
                "Tick-by-tick mode has a limit of 3 simultaneous subscriptions. "
                "Each ticker uses one subscription. Current tickers: MNQ, MES, MGC (3/3 used)"
            )

    async def start_stream(
        self,
        contract: Contract,
        on_bar_callback: Callable
    ) -> bool:
        """
        Start streaming for a contract.

        Args:
            contract: Futures contract
            on_bar_callback: Callback function

        Returns:
            True if started successfully
        """
        symbol = contract.symbol

        if symbol in self.streamers:
            logger.info(
                f"Stream already active for {symbol} - reusing existing stream "
                f"(multiple WebSocket clients share the same tick stream)"
            )
            return True  # Stream exists, other clients will receive updates via broadcast

        try:
            logger.info(f"Creating new tick-by-tick stream for {symbol}...")
            if self.use_tick_by_tick:
                # Use tick-by-tick streaming
                streamer = LiveCandlestickBuilder(
                    self.ib,
                    contract,
                    bar_size_minutes=self.bar_size_minutes,
                    on_bar_callback=on_bar_callback
                )
                await streamer.start()

            else:
                # Use keepUpToDate streaming
                bar_size_str = f"{self.bar_size_minutes} min" if self.bar_size_minutes == 1 else f"{self.bar_size_minutes} mins"
                streamer = KeepUpToDateStreamer(
                    self.ib,
                    contract,
                    bar_size=bar_size_str,
                    on_bar_callback=on_bar_callback
                )
                await streamer.start()

            self.streamers[symbol] = streamer
            logger.info(f"Stream started for {symbol}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start stream for {symbol}: {e}", exc_info=True)
            return False

    def stop_stream(self, symbol: str):
        """Stop streaming for a contract"""
        if symbol in self.streamers:
            self.streamers[symbol].stop()
            del self.streamers[symbol]
            logger.info(f"Stream stopped for {symbol}")

    def stop_all_streams(self):
        """Stop all active streams"""
        symbols = list(self.streamers.keys())
        for symbol in symbols:
            self.stop_stream(symbol)

    def get_statistics(self, symbol: str) -> Optional[dict]:
        """Get statistics for a stream"""
        if symbol in self.streamers:
            return self.streamers[symbol].get_statistics()
        return None

    def get_all_statistics(self) -> Dict[str, dict]:
        """Get statistics for all streams"""
        return {
            symbol: streamer.get_statistics()
            for symbol, streamer in self.streamers.items()
        }


if __name__ == '__main__':
    # Example usage
    async def test():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        from .contracts import get_current_contract
        from .ib_service import IBConnectionManager

        # Connect
        manager = IBConnectionManager(host='127.0.0.1', port=4002)
        await manager.connect()

        if not manager.is_connected():
            print("Failed to connect")
            return

        # Get contract
        contract = get_current_contract('MNQ')
        await manager.ib.qualifyContractsAsync(contract)

        print(f"Starting real-time stream for {contract.symbol}...")

        # Callback for bar updates
        def on_bar_update(bar_data, is_new_bar):
            bar_type = "NEW BAR" if is_new_bar else "UPDATE"
            print(
                f"[{bar_type}] Time: {bar_data['time']}, "
                f"O: {bar_data['open']:.2f}, "
                f"H: {bar_data['high']:.2f}, "
                f"L: {bar_data['low']:.2f}, "
                f"C: {bar_data['close']:.2f}, "
                f"V: {bar_data['volume']}"
            )

        # Create realtime manager (use keepUpToDate for simplicity)
        rt_manager = RealtimeManager(manager.ib, use_tick_by_tick=False)

        # Start streaming
        await rt_manager.start_stream(contract, on_bar_update)

        # Run for 5 minutes
        print("Streaming for 5 minutes (Ctrl+C to stop)...")
        try:
            await asyncio.sleep(300)
        except KeyboardInterrupt:
            print("\nStopping...")

        # Stop and show statistics
        stats = rt_manager.get_statistics('MNQ')
        print(f"\nStatistics: {stats}")

        rt_manager.stop_all_streams()
        manager.disconnect()

    from ib_insync import util
    util.startLoop()
    asyncio.run(test())
