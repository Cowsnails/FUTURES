"""
TastyTrade Live Bar Builder

Builds OHLCV candlestick bars from TastyTrade's DXLink TimeAndSale events.
Replaces IB's LiveCandlestickBuilder with the same output format, plus
volume delta fields as a bonus.

Output format matches exactly what the rest of the system expects:
    bar_data = {
        'time': int,       # Unix timestamp (Eastern display time)
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': int,
        # Bonus fields from TastyTrade order flow:
        'buy_volume': int,
        'sell_volume': int,
        'delta': int,      # buy_volume - sell_volume
    }
"""

import asyncio
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List

import pytz

logger = logging.getLogger(__name__)

EASTERN = pytz.timezone('US/Eastern')


class TastyTradeBarBuilder:
    """
    Builds live OHLCV bars from TastyTrade TimeAndSale tick data.

    Drop-in replacement for LiveCandlestickBuilder. Registers as a tick
    callback on TastyTradeStreamer and produces the same bar_data dicts
    that the existing WebSocket broadcast, pattern matching, setup detectors,
    and stats tracker all consume.

    Additionally tracks buy/sell volume per bar for delta analysis.
    """

    def __init__(
        self,
        symbol: str,
        bar_size_minutes: int = 1,
        on_bar_callback: Optional[Callable] = None,
    ):
        """
        Args:
            symbol: Internal symbol (e.g., 'MNQ')
            bar_size_minutes: Bar duration in minutes (default: 1)
            on_bar_callback: async callback(bar_data: dict, is_new_bar: bool)
        """
        self.symbol = symbol
        self.bar_size_seconds = bar_size_minutes * 60
        self.on_bar = on_bar_callback

        # Current bar state
        self.current_bar: Optional[Dict[str, Any]] = None
        self.current_bar_start: int = 0  # UTC-aligned bar start (seconds)

        # Per-bar delta tracking
        self.buy_volume: int = 0
        self.sell_volume: int = 0

        # Statistics
        self.stats = {
            'ticks_processed': 0,
            'bars_completed': 0,
            'bars_updated': 0,
        }

    def on_tick(self, tick_data: Dict[str, Any]):
        """
        Process a classified tick from TastyTradeStreamer.

        This is registered as a tick callback on the streamer.
        Builds OHLCV bars and calls the on_bar callback.
        """
        if tick_data.get('type') != 'trade':
            return

        price = tick_data.get('price', 0)
        size = tick_data.get('size', 0)
        side = tick_data.get('side', 'buy')

        if price <= 0 or size <= 0:
            return

        self.stats['ticks_processed'] += 1

        # Determine bar start time from tick timestamp
        tick_ms = tick_data.get('time', 0)
        if tick_ms > 1e12:
            tick_s = tick_ms // 1000
        elif tick_ms > 0:
            tick_s = tick_ms
        else:
            tick_s = int(time.time())

        bar_start = (tick_s // self.bar_size_seconds) * self.bar_size_seconds

        completed_bar = None
        is_new_bar = False

        # Check for bar rollover
        if self.current_bar_start > 0 and bar_start > self.current_bar_start:
            # Finalize the completed bar
            completed_bar = self._finalize_current_bar()
            # Start new bar
            self._start_new_bar(bar_start, price, size, side)
            is_new_bar = True
        elif self.current_bar is None:
            # First tick
            self._start_new_bar(bar_start, price, size, side)
            is_new_bar = True
        else:
            # Update current bar
            self._update_bar(price, size, side)

        # Dispatch completed bar
        if completed_bar and self.on_bar:
            self.stats['bars_completed'] += 1
            self._dispatch(completed_bar, is_new_bar=True)

        # Dispatch current bar update
        if self.current_bar and self.on_bar:
            self.stats['bars_updated'] += 1
            self._dispatch(self.current_bar.copy(), is_new_bar=is_new_bar)

    def _start_new_bar(self, bar_start: int, price: float, size: int, side: str):
        """Start a new bar."""
        self.current_bar_start = bar_start
        display_time = self._to_eastern_display_timestamp(bar_start)

        self.buy_volume = size if side == 'buy' else 0
        self.sell_volume = size if side == 'sell' else 0

        self.current_bar = {
            'time': display_time,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': size,
            'buy_volume': self.buy_volume,
            'sell_volume': self.sell_volume,
            'delta': self.buy_volume - self.sell_volume,
        }

    def _update_bar(self, price: float, size: int, side: str):
        """Update the current bar with a new tick."""
        if not self.current_bar:
            return

        self.current_bar['high'] = max(self.current_bar['high'], price)
        self.current_bar['low'] = min(self.current_bar['low'], price)
        self.current_bar['close'] = price
        self.current_bar['volume'] += size

        if side == 'buy':
            self.buy_volume += size
        else:
            self.sell_volume += size

        self.current_bar['buy_volume'] = self.buy_volume
        self.current_bar['sell_volume'] = self.sell_volume
        self.current_bar['delta'] = self.buy_volume - self.sell_volume

    def _finalize_current_bar(self) -> Optional[Dict[str, Any]]:
        """Finalize and return the current bar."""
        if not self.current_bar:
            return None

        bar = self.current_bar.copy()

        logger.debug(
            f"[{self.symbol}] Bar completed: "
            f"O={bar['open']:.2f} H={bar['high']:.2f} "
            f"L={bar['low']:.2f} C={bar['close']:.2f} "
            f"V={bar['volume']} D={bar['delta']}"
        )

        return bar

    def _dispatch(self, bar_data: Dict[str, Any], is_new_bar: bool):
        """Dispatch bar data to callback."""
        try:
            result = self.on_bar(bar_data, is_new_bar)
            if asyncio.iscoroutine(result):
                asyncio.create_task(result)
        except Exception as e:
            logger.error(f"[{self.symbol}] Bar callback error: {e}")

    @staticmethod
    def _to_eastern_display_timestamp(utc_seconds: int) -> int:
        """
        Convert UTC timestamp to Eastern "display timestamp".

        Matches the IB system's convention: treat Eastern time as if it were UTC,
        so the chart shows 16:59 instead of 21:59 for a 4:59 PM Eastern bar.
        """
        utc_dt = datetime.fromtimestamp(utc_seconds, tz=pytz.UTC)
        eastern_dt = utc_dt.astimezone(EASTERN)

        # Create display timestamp: Eastern time "pretending" to be UTC
        display_dt = datetime(
            eastern_dt.year, eastern_dt.month, eastern_dt.day,
            eastern_dt.hour, eastern_dt.minute, eastern_dt.second,
            tzinfo=pytz.UTC
        )
        return int(display_dt.timestamp())

    def get_statistics(self) -> Dict[str, Any]:
        """Get bar builder statistics."""
        return dict(self.stats)


class TastyTradeBarManager:
    """
    Manages TastyTrade bar builders for multiple symbols.

    Analogous to RealtimeManager but backed by TastyTrade streaming.
    Provides the same interface so app.py can use either IB or TastyTrade.
    """

    def __init__(self, streamer, bar_size_minutes: int = 1):
        """
        Args:
            streamer: TastyTradeStreamer instance
            bar_size_minutes: Bar size in minutes
        """
        self.streamer = streamer
        self.bar_size_minutes = bar_size_minutes
        self.builders: Dict[str, TastyTradeBarBuilder] = {}

    def start_stream(self, symbol: str, on_bar_callback: Callable) -> bool:
        """
        Start building bars for a symbol.

        Args:
            symbol: Internal symbol (e.g., 'MNQ')
            on_bar_callback: async callback(bar_data, is_new_bar)

        Returns:
            True if started
        """
        if symbol in self.builders:
            logger.info(f"Bar builder already active for {symbol}")
            return True

        builder = TastyTradeBarBuilder(
            symbol=symbol,
            bar_size_minutes=self.bar_size_minutes,
            on_bar_callback=on_bar_callback,
        )

        # Register the builder as a tick callback on the streamer
        self.streamer.register_tick_callback(symbol, builder.on_tick)
        self.builders[symbol] = builder

        logger.info(f"TastyTrade bar builder started for {symbol}")
        return True

    def stop_stream(self, symbol: str):
        """Stop building bars for a symbol."""
        if symbol in self.builders:
            self.streamer.unregister_tick_callbacks(symbol)
            del self.builders[symbol]
            logger.info(f"Bar builder stopped for {symbol}")

    def stop_all_streams(self):
        """Stop all bar builders."""
        for symbol in list(self.builders.keys()):
            self.stop_stream(symbol)

    def get_statistics(self, symbol: str) -> Optional[Dict]:
        if symbol in self.builders:
            return self.builders[symbol].get_statistics()
        return None

    def get_all_statistics(self) -> Dict[str, Dict]:
        return {s: b.get_statistics() for s, b in self.builders.items()}
