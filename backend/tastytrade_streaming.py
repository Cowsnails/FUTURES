"""
TastyTrade DXLink Streaming Service

Connects to dxFeed via TastyTrade's DXLink WebSocket and streams
TimeAndSale events for tick-by-tick futures trade data with exchange-native
aggressor side flags from CME Globex.

This module feeds the volume_delta engine with classified trade ticks.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any, List, Set
from decimal import Decimal

logger = logging.getLogger(__name__)


class TastyTradeStreamer:
    """
    Manages DXLink WebSocket streaming for TimeAndSale events.

    Uses the tastytrade SDK's DXLinkStreamer for async event delivery.
    Subscribes to TimeAndSale (lossless Stream delivery) plus optional
    Quote and Trade events for supplemental data.

    Features:
    - Lossless TimeAndSale delivery (no conflation)
    - Aggressor side classification (CME MDP 3.0 tag 5797)
    - Automatic token refresh via TastyTradeService
    - Per-symbol callback routing
    - Connection health monitoring
    """

    def __init__(self, tastytrade_service):
        """
        Args:
            tastytrade_service: Authenticated TastyTradeService instance
        """
        self.tt_service = tastytrade_service
        self.streamer = None  # DXLinkStreamer context manager
        self._stream_task: Optional[asyncio.Task] = None
        self._running = False

        # Per-symbol callbacks: {'MNQ': [callback1, ...], ...}
        self._tick_callbacks: Dict[str, List[Callable]] = {}
        # Symbols currently subscribed to
        self._subscribed_symbols: Set[str] = set()

        # Statistics
        self.stats = {
            'ticks_received': 0,
            'ticks_classified_exchange': 0,   # aggressorSide == BUY/SELL
            'ticks_classified_quote': 0,      # quote rule fallback
            'ticks_classified_tick': 0,       # tick rule fallback
            'ticks_undefined': 0,             # could not classify
            'corrections': 0,
            'cancellations': 0,
            'spread_legs_filtered': 0,
            'errors': 0,
            'stream_started_at': None,
        }

        # Track previous trade price per symbol for tick rule fallback
        self._prev_prices: Dict[str, float] = {}
        self._prev_sides: Dict[str, str] = {}

    async def start(self, symbols: List[str]) -> bool:
        """
        Start streaming TimeAndSale events for the given symbols.

        Args:
            symbols: Internal symbols (e.g., ['MNQ', 'MES', 'MGC'])

        Returns:
            True if streaming started successfully
        """
        if not self.tt_service.is_authenticated():
            logger.error("Cannot start streaming - TastyTrade not authenticated")
            return False

        # Resolve streamer symbols
        streamer_symbols = await self.tt_service.resolve_streamer_symbols(symbols)
        if not streamer_symbols:
            logger.error("No streamer symbols resolved - cannot start streaming")
            return False

        self._running = True
        self.stats['stream_started_at'] = datetime.now().isoformat()

        # Launch the streaming loop as a background task
        self._stream_task = asyncio.create_task(
            self._streaming_loop(symbols, streamer_symbols)
        )

        logger.info(
            f"TastyTrade streaming started for: "
            f"{', '.join(f'{s}->{streamer_symbols[s]}' for s in symbols if s in streamer_symbols)}"
        )
        return True

    async def _streaming_loop(
        self,
        symbols: List[str],
        streamer_symbols: Dict[str, str]
    ):
        """
        Main streaming loop - connects to DXLink and processes TimeAndSale events.
        """
        try:
            from tastytrade import DXLinkStreamer
            from tastytrade.dxfeed import TimeAndSale, Quote, Trade

            # Build reverse map: streamer_symbol -> internal_symbol
            reverse_map = {v: k for k, v in streamer_symbols.items()}
            dx_symbols = list(streamer_symbols.values())

            await self.tt_service.refresh_session()

            async with DXLinkStreamer(self.tt_service.session) as streamer:
                self.streamer = streamer
                self.tt_service.state = (
                    self.tt_service.__class__.__mro__[0]  # Get enum from service
                    if hasattr(self.tt_service, 'state') else None
                )
                # Update state to STREAMING
                from .tastytrade_service import TastyTradeConnectionState
                self.tt_service.state = TastyTradeConnectionState.STREAMING

                # Subscribe to TimeAndSale (lossless, Stream delivery)
                await streamer.subscribe(TimeAndSale, dx_symbols)
                logger.info(f"Subscribed to TimeAndSale: {dx_symbols}")

                # Also subscribe to Quote for supplemental BBO data
                await streamer.subscribe(Quote, dx_symbols)
                logger.info(f"Subscribed to Quote: {dx_symbols}")

                self._subscribed_symbols = set(symbols)

                # Process events
                async for tns in streamer.listen(TimeAndSale):
                    if not self._running:
                        break

                    try:
                        self._process_time_and_sale(tns, reverse_map)
                    except Exception as e:
                        self.stats['errors'] += 1
                        if self.stats['errors'] <= 10:
                            logger.error(f"Error processing TimeAndSale: {e}")

        except ImportError:
            logger.error(
                "tastytrade package not installed. "
                "Install with: pip install tastytrade"
            )
        except Exception as e:
            logger.error(f"TastyTrade streaming error: {e}", exc_info=True)
        finally:
            self.streamer = None
            self._running = False
            from .tastytrade_service import TastyTradeConnectionState
            if self.tt_service.is_authenticated():
                self.tt_service.state = TastyTradeConnectionState.AUTHENTICATED
            logger.info("TastyTrade streaming loop ended")

    def _process_time_and_sale(self, tns, reverse_map: Dict[str, str]):
        """
        Process a single TimeAndSale event.

        Classifies the aggressor side using a three-tier approach:
        1. Exchange flag (CME MDP 3.0 tag 5797) - gold standard
        2. Quote rule (embedded bid/ask at time of trade)
        3. Tick rule (price vs previous price) - least accurate
        """
        event_symbol = tns.eventSymbol
        internal_symbol = reverse_map.get(event_symbol)
        if not internal_symbol:
            return

        self.stats['ticks_received'] += 1

        # Filter: skip corrections and cancellations (track for adjustments)
        event_type = getattr(tns, 'type', 'NEW')
        if event_type == 'CORRECTION':
            self.stats['corrections'] += 1
            self._dispatch_tick(internal_symbol, {
                'type': 'correction',
                'event_symbol': event_symbol,
                'price': float(tns.price) if tns.price else 0,
                'size': int(tns.size) if tns.size else 0,
                'time': getattr(tns, 'time', 0),
            })
            return
        elif event_type == 'CANCEL':
            self.stats['cancellations'] += 1
            self._dispatch_tick(internal_symbol, {
                'type': 'cancel',
                'event_symbol': event_symbol,
                'price': float(tns.price) if tns.price else 0,
                'size': int(tns.size) if tns.size else 0,
                'time': getattr(tns, 'time', 0),
            })
            return

        # Filter: skip invalid ticks
        if hasattr(tns, 'validTick') and not tns.validTick:
            return

        # Filter: optionally skip spread legs
        is_spread_leg = getattr(tns, 'spreadLeg', False)
        if is_spread_leg:
            self.stats['spread_legs_filtered'] += 1
            # Still process but flag it - let the delta engine decide
            pass

        price = float(tns.price) if tns.price else 0
        size = int(tns.size) if tns.size else 0
        if price <= 0 or size <= 0:
            return

        # Three-tier aggressor classification
        side, method = self._classify_aggressor(tns, internal_symbol)

        # Track classification stats
        if method == 'exchange':
            self.stats['ticks_classified_exchange'] += 1
        elif method == 'quote':
            self.stats['ticks_classified_quote'] += 1
        elif method == 'tick':
            self.stats['ticks_classified_tick'] += 1
        else:
            self.stats['ticks_undefined'] += 1

        # Update previous price/side for tick rule
        self._prev_prices[internal_symbol] = price
        self._prev_sides[internal_symbol] = side

        # Build tick data for dispatch
        tick_data = {
            'type': 'trade',
            'symbol': internal_symbol,
            'event_symbol': event_symbol,
            'price': price,
            'size': size,
            'side': side,              # 'buy' or 'sell'
            'method': method,          # 'exchange', 'quote', 'tick', 'unknown'
            'time': getattr(tns, 'time', 0),
            'time_nano': getattr(tns, 'timeNanoPart', 0),
            'sequence': getattr(tns, 'sequence', 0),
            'bid_price': float(tns.bidPrice) if getattr(tns, 'bidPrice', None) else 0,
            'ask_price': float(tns.askPrice) if getattr(tns, 'askPrice', None) else 0,
            'aggressor_side': getattr(tns, 'aggressorSide', 'UNDEFINED'),
            'exchange_code': getattr(tns, 'exchangeCode', ''),
            'is_spread_leg': is_spread_leg,
            'is_eth': getattr(tns, 'extendedTradingHours', False),
        }

        # Dispatch to registered callbacks
        self._dispatch_tick(internal_symbol, tick_data)

    def _classify_aggressor(self, tns, symbol: str) -> tuple:
        """
        Three-tier aggressor classification for CME futures.

        Returns:
            (side, method) where side is 'buy'/'sell' and method is
            'exchange'/'quote'/'tick'/'unknown'
        """
        # Tier 1: Direct exchange flag (CME MDP 3.0 tag 5797)
        aggressor = getattr(tns, 'aggressorSide', 'UNDEFINED')
        if aggressor == 'BUY':
            return ('buy', 'exchange')
        elif aggressor == 'SELL':
            return ('sell', 'exchange')

        # Tier 2: Quote rule using embedded NBBO
        bid = getattr(tns, 'bidPrice', None)
        ask = getattr(tns, 'askPrice', None)
        price = float(tns.price) if tns.price else 0

        if bid and ask:
            bid_f = float(bid)
            ask_f = float(ask)
            if bid_f > 0 and ask_f > 0 and price > 0:
                if price >= ask_f:
                    return ('buy', 'quote')
                elif price <= bid_f:
                    return ('sell', 'quote')
                else:
                    mid = (bid_f + ask_f) / 2
                    return (('buy' if price >= mid else 'sell'), 'quote')

        # Tier 3: Tick rule (least accurate)
        prev_price = self._prev_prices.get(symbol)
        if prev_price is not None and price > 0:
            if price > prev_price:
                return ('buy', 'tick')
            elif price < prev_price:
                return ('sell', 'tick')

        # Fall through: use previous side or default
        prev_side = self._prev_sides.get(symbol, 'buy')
        return (prev_side, 'unknown')

    def _dispatch_tick(self, symbol: str, tick_data: Dict[str, Any]):
        """Dispatch tick data to all registered callbacks for a symbol."""
        callbacks = self._tick_callbacks.get(symbol, [])
        for cb in callbacks:
            try:
                result = cb(tick_data)
                # Support both sync and async callbacks
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"Error in tick callback for {symbol}: {e}")

    def register_tick_callback(self, symbol: str, callback: Callable):
        """
        Register a callback for tick data on a symbol.

        The callback receives a dict with:
        - type: 'trade', 'correction', or 'cancel'
        - symbol: internal symbol
        - price: trade price
        - size: trade size (contracts)
        - side: 'buy' or 'sell'
        - method: classification method used
        - time: millisecond timestamp
        - bid_price/ask_price: NBBO at time of trade
        - aggressor_side: raw exchange flag
        """
        if symbol not in self._tick_callbacks:
            self._tick_callbacks[symbol] = []
        self._tick_callbacks[symbol].append(callback)
        logger.info(f"Registered tick callback for {symbol}")

    def unregister_tick_callbacks(self, symbol: str):
        """Remove all tick callbacks for a symbol."""
        self._tick_callbacks.pop(symbol, None)

    async def stop(self):
        """Stop streaming and clean up."""
        self._running = False

        if self._stream_task and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        self._subscribed_symbols.clear()
        self._tick_callbacks.clear()
        self._prev_prices.clear()
        self._prev_sides.clear()
        logger.info("TastyTrade streamer stopped")

    def is_streaming(self) -> bool:
        """Check if actively streaming."""
        return self._running and self._stream_task is not None

    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        stats = dict(self.stats)
        stats['streaming'] = self.is_streaming()
        stats['subscribed_symbols'] = list(self._subscribed_symbols)
        stats['callback_count'] = sum(len(v) for v in self._tick_callbacks.values())

        # Classification breakdown
        total = stats['ticks_received']
        if total > 0:
            stats['exchange_pct'] = round(
                stats['ticks_classified_exchange'] / total * 100, 1
            )
            stats['quote_pct'] = round(
                stats['ticks_classified_quote'] / total * 100, 1
            )
            stats['tick_pct'] = round(
                stats['ticks_classified_tick'] / total * 100, 1
            )

        return stats
