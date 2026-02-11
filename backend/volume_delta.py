"""
Volume Delta Calculation Engine

Processes classified trade ticks from TastyTrade's DXLink TimeAndSale stream
to calculate:
- Per-bar volume delta (buy volume - sell volume)
- Cumulative volume delta (CVD) across the session
- Price-level delta for footprint chart data
- Per-bar buy/sell volume breakdown

Integrates with the existing bar-building pipeline and WebSocket broadcast system.
"""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List, Tuple

import pytz

logger = logging.getLogger(__name__)

# CME session boundaries (Central Time -> Eastern Time for alignment)
# Globex session: Sunday 5:00 PM CT to Friday 4:00 PM CT
# Day session: 8:30 AM - 3:15 PM CT (9:30 AM - 4:15 PM ET)
CME_SESSION_RESET_HOUR_ET = 18  # 6:00 PM ET = 5:00 PM CT (new session start)
CME_RTH_OPEN_HOUR_ET = 9        # 9:30 AM ET
CME_RTH_OPEN_MINUTE = 30
CME_RTH_CLOSE_HOUR_ET = 16      # 4:00 PM ET


class VolumeDeltaEngine:
    """
    Core volume delta calculation engine.

    Maintains running state for:
    - Current bar delta (buy_vol - sell_vol)
    - Cumulative delta (CVD) across the session
    - Per-price-level delta (for footprint charts)
    - Historical bar deltas for indicator calculations

    Receives classified ticks from TastyTradeStreamer and produces
    delta data that integrates with the existing bar pipeline.
    """

    def __init__(self, bar_size_seconds: int = 60):
        """
        Args:
            bar_size_seconds: Bar duration in seconds (default: 60 for 1-min bars)
        """
        self.bar_size_seconds = bar_size_seconds

        # Per-symbol state
        self._state: Dict[str, SymbolDeltaState] = {}

        # Callbacks for delta updates
        self._delta_callbacks: Dict[str, List[Callable]] = {}

        # Historical delta bars: {symbol: [delta_bar_dict, ...]}
        self._delta_history: Dict[str, List[Dict[str, Any]]] = {}
        self._max_history_bars = 10000

    def get_or_create_state(self, symbol: str) -> 'SymbolDeltaState':
        """Get or create delta state for a symbol."""
        if symbol not in self._state:
            self._state[symbol] = SymbolDeltaState(
                symbol=symbol,
                bar_size_seconds=self.bar_size_seconds,
            )
        return self._state[symbol]

    def process_tick(self, tick_data: Dict[str, Any]):
        """
        Process a classified trade tick from the TastyTrade streamer.

        This is the main entry point called by TastyTradeStreamer callbacks.

        Args:
            tick_data: Dict with keys: type, symbol, price, size, side,
                      method, time, bid_price, ask_price, etc.
        """
        tick_type = tick_data.get('type')
        symbol = tick_data.get('symbol')
        if not symbol:
            return

        state = self.get_or_create_state(symbol)

        if tick_type == 'trade':
            delta_update = state.add_trade(
                price=tick_data['price'],
                size=tick_data['size'],
                side=tick_data['side'],
                timestamp_ms=tick_data.get('time', 0),
                is_spread_leg=tick_data.get('is_spread_leg', False),
            )

            if delta_update:
                # Dispatch delta update to callbacks
                self._dispatch_delta(symbol, delta_update)

                # If bar completed, archive it
                if delta_update.get('bar_completed'):
                    self._archive_bar(symbol, delta_update['completed_bar'])

        elif tick_type == 'correction':
            state.apply_correction(
                price=tick_data['price'],
                size=tick_data['size'],
            )

        elif tick_type == 'cancel':
            state.apply_cancellation(
                price=tick_data['price'],
                size=tick_data['size'],
            )

    def register_delta_callback(self, symbol: str, callback: Callable):
        """
        Register a callback for delta updates on a symbol.

        The callback receives a dict with delta state for the current bar:
        - bar_time: bar start timestamp
        - buy_volume: total buy volume in bar
        - sell_volume: total sell volume in bar
        - delta: buy_volume - sell_volume
        - cumulative_delta: session CVD
        - bar_completed: True if a bar just completed
        - price_levels: {price: {buy_vol, sell_vol, delta}} for footprint
        """
        if symbol not in self._delta_callbacks:
            self._delta_callbacks[symbol] = []
        self._delta_callbacks[symbol].append(callback)

    def _dispatch_delta(self, symbol: str, delta_update: Dict[str, Any]):
        """Dispatch delta update to registered callbacks."""
        callbacks = self._delta_callbacks.get(symbol, [])
        for cb in callbacks:
            try:
                result = cb(delta_update)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"Error in delta callback for {symbol}: {e}")

    def _archive_bar(self, symbol: str, bar_data: Dict[str, Any]):
        """Archive a completed delta bar to history."""
        if symbol not in self._delta_history:
            self._delta_history[symbol] = []

        self._delta_history[symbol].append(bar_data)

        # Trim to max history
        if len(self._delta_history[symbol]) > self._max_history_bars:
            self._delta_history[symbol] = self._delta_history[symbol][-self._max_history_bars:]

    def get_delta_history(
        self,
        symbol: str,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """Get historical delta bars for a symbol."""
        history = self._delta_history.get(symbol, [])
        return history[-limit:] if len(history) > limit else list(history)

    def get_current_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current delta state snapshot for a symbol."""
        state = self._state.get(symbol)
        if not state:
            return None
        return state.to_dict()

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get delta state for all tracked symbols."""
        return {
            symbol: state.to_dict()
            for symbol, state in self._state.items()
        }

    def reset_session(self, symbol: str):
        """Reset CVD and session state (call at session boundary)."""
        state = self._state.get(symbol)
        if state:
            state.reset_session()
            logger.info(f"Volume delta session reset for {symbol}")

    def reset_all_sessions(self):
        """Reset all symbols' session state."""
        for symbol in self._state:
            self.reset_session(symbol)


class SymbolDeltaState:
    """
    Maintains volume delta state for a single symbol.

    Tracks current bar, cumulative session delta, and per-price-level
    volume breakdown.
    """

    def __init__(self, symbol: str, bar_size_seconds: int = 60):
        self.symbol = symbol
        self.bar_size_seconds = bar_size_seconds

        # Current bar state
        self.current_bar_time: int = 0  # Unix timestamp of bar start
        self.buy_volume: int = 0
        self.sell_volume: int = 0
        self.trade_count: int = 0

        # Per-price-level tracking for footprint
        # {price: {'buy': vol, 'sell': vol}}
        self.price_levels: Dict[float, Dict[str, int]] = defaultdict(
            lambda: {'buy': 0, 'sell': 0}
        )

        # Session cumulative delta
        self.cumulative_delta: int = 0
        self.session_buy_volume: int = 0
        self.session_sell_volume: int = 0
        self.session_trade_count: int = 0

        # High/low watermarks for CVD
        self.cvd_high: int = 0
        self.cvd_low: int = 0

        # Last known price for mid-bar snapshots
        self.last_price: float = 0
        self.last_trade_time: int = 0

    def add_trade(
        self,
        price: float,
        size: int,
        side: str,
        timestamp_ms: int = 0,
        is_spread_leg: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Add a classified trade to the delta state.

        Returns a delta update dict, or None if the trade was filtered.
        """
        # Determine bar start time
        if timestamp_ms > 0:
            # Convert millisecond timestamp to bar-aligned timestamp
            timestamp_s = timestamp_ms // 1000 if timestamp_ms > 1e12 else timestamp_ms
            bar_time = (timestamp_s // self.bar_size_seconds) * self.bar_size_seconds
        else:
            bar_time = (int(time.time()) // self.bar_size_seconds) * self.bar_size_seconds

        completed_bar = None
        bar_completed = False

        # Check if we need to roll to a new bar
        if self.current_bar_time > 0 and bar_time > self.current_bar_time:
            # Finalize the current bar
            completed_bar = self._finalize_bar()
            bar_completed = True
            # Start new bar
            self._start_new_bar(bar_time)
        elif self.current_bar_time == 0:
            # First trade
            self._start_new_bar(bar_time)

        # Accumulate volume
        if side == 'buy':
            self.buy_volume += size
            self.session_buy_volume += size
            self.cumulative_delta += size
        else:  # 'sell'
            self.sell_volume += size
            self.session_sell_volume += size
            self.cumulative_delta -= size

        self.trade_count += 1
        self.session_trade_count += 1
        self.last_price = price
        self.last_trade_time = timestamp_ms

        # Track per-price-level
        level = self.price_levels[price]
        if side == 'buy':
            level['buy'] += size
        else:
            level['sell'] += size

        # Update CVD watermarks
        if self.cumulative_delta > self.cvd_high:
            self.cvd_high = self.cumulative_delta
        if self.cumulative_delta < self.cvd_low:
            self.cvd_low = self.cumulative_delta

        # Build update
        delta = self.buy_volume - self.sell_volume
        update = {
            'symbol': self.symbol,
            'bar_time': self.current_bar_time,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'delta': delta,
            'cumulative_delta': self.cumulative_delta,
            'trade_count': self.trade_count,
            'last_price': price,
            'last_side': side,
            'bar_completed': bar_completed,
            'is_spread_leg': is_spread_leg,
        }

        if completed_bar:
            update['completed_bar'] = completed_bar

        return update

    def _start_new_bar(self, bar_time: int):
        """Initialize state for a new bar."""
        self.current_bar_time = bar_time
        self.buy_volume = 0
        self.sell_volume = 0
        self.trade_count = 0
        self.price_levels = defaultdict(lambda: {'buy': 0, 'sell': 0})

    def _finalize_bar(self) -> Dict[str, Any]:
        """Finalize the current bar and return its data."""
        delta = self.buy_volume - self.sell_volume

        # Build footprint data (top N price levels by total volume)
        footprint = {}
        for price, vols in sorted(self.price_levels.items()):
            total = vols['buy'] + vols['sell']
            if total > 0:
                footprint[price] = {
                    'buy': vols['buy'],
                    'sell': vols['sell'],
                    'delta': vols['buy'] - vols['sell'],
                    'total': total,
                }

        return {
            'time': self.current_bar_time,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'delta': delta,
            'cumulative_delta': self.cumulative_delta,
            'trade_count': self.trade_count,
            'footprint': footprint,
        }

    def apply_correction(self, price: float, size: int):
        """Apply a trade correction (adjusts delta by removing the original)."""
        # Corrections reduce volume; exact side unknown from correction alone
        # In practice, you'd need to track the original trade
        logger.debug(f"[{self.symbol}] Correction: price={price} size={size}")

    def apply_cancellation(self, price: float, size: int):
        """Apply a trade cancellation."""
        logger.debug(f"[{self.symbol}] Cancellation: price={price} size={size}")

    def reset_session(self):
        """Reset cumulative delta and session stats (at session boundary)."""
        self.cumulative_delta = 0
        self.session_buy_volume = 0
        self.session_sell_volume = 0
        self.session_trade_count = 0
        self.cvd_high = 0
        self.cvd_low = 0
        logger.info(f"[{self.symbol}] Session delta reset")

    def to_dict(self) -> Dict[str, Any]:
        """Snapshot current state as a dictionary."""
        delta = self.buy_volume - self.sell_volume
        return {
            'symbol': self.symbol,
            'bar_time': self.current_bar_time,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'delta': delta,
            'cumulative_delta': self.cumulative_delta,
            'cvd_high': self.cvd_high,
            'cvd_low': self.cvd_low,
            'session_buy_volume': self.session_buy_volume,
            'session_sell_volume': self.session_sell_volume,
            'session_trade_count': self.session_trade_count,
            'trade_count': self.trade_count,
            'last_price': self.last_price,
            'last_trade_time': self.last_trade_time,
        }


class SessionResetScheduler:
    """
    Schedules CVD resets at CME session boundaries.

    CME Globex sessions start at 5:00 PM CT (6:00 PM ET) Sunday-Thursday.
    This scheduler watches the clock and triggers session resets automatically.
    """

    def __init__(self, delta_engine: VolumeDeltaEngine):
        self.engine = delta_engine
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the session reset scheduler."""
        self._running = True
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info("Session reset scheduler started (resets at 6:00 PM ET)")

    async def _schedule_loop(self):
        """Background loop that checks for session boundaries."""
        eastern = pytz.timezone('US/Eastern')
        last_reset_date = None

        while self._running:
            try:
                now = datetime.now(eastern)

                # Check if it's past 6:00 PM ET and we haven't reset today
                if (now.hour >= CME_SESSION_RESET_HOUR_ET
                        and now.date() != last_reset_date
                        and now.weekday() < 5):  # Mon-Fri only
                    logger.info("CME session boundary reached - resetting CVD")
                    self.engine.reset_all_sessions()
                    last_reset_date = now.date()

                # Check every 30 seconds
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session reset scheduler error: {e}")
                await asyncio.sleep(60)

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
