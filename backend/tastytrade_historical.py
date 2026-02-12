"""
TastyTrade Historical Data Fetcher

Fetches historical OHLCV candle data via DXLink Candle events, replacing
IB's HistoricalDataFetcher. Produces the exact same DataFrame format
(time, open, high, low, close, volume) and uses the same cache system.

Bonus: Candle events include bidVolume/askVolume for pre-aggregated delta.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import pandas as pd
import pytz

from .cache import DataCache, optimize_dataframe

logger = logging.getLogger(__name__)

EASTERN = pytz.timezone('US/Eastern')

# Map timeframe names to DXLink candle period suffixes
TIMEFRAME_TO_CANDLE_PERIOD = {
    '1min': '1m',
    '5min': '5m',
    '15min': '15m',
    '30min': '30m',
    '1H': '1h',
    '2H': '2h',
    '4H': '4h',
    '1D': '1d',
}


class TastyTradeHistoricalFetcher:
    """
    Fetches historical candle data from TastyTrade's DXLink Candle events.

    Features:
    - Fetches historical OHLCV bars via subscribe_candle with fromTime
    - Includes bidVolume/askVolume (buy/sell split) per candle
    - Same cache format as HistoricalDataFetcher for seamless switching
    - Incremental updates (only fetches gap)
    - Aggregates to all timeframes
    """

    def __init__(
        self,
        tastytrade_service,
        cache: Optional[DataCache] = None,
    ):
        """
        Args:
            tastytrade_service: Authenticated TastyTradeService instance
            cache: DataCache instance (creates new if None)
        """
        self.tt_service = tastytrade_service
        self.cache = cache or DataCache()

        self.stats = {
            'total_bars_fetched': 0,
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    async def fetch_recent(
        self,
        symbol: str,
        duration: str = '60 D',
        cache_all_timeframes: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch recent historical data with incremental updates.

        Drop-in replacement for HistoricalDataFetcher.fetch_recent().
        Uses TastyTrade Candle events instead of IB historical data requests.

        Args:
            symbol: Internal symbol (e.g., 'MNQ')
            duration: Duration string (e.g., '60 D', '1 Y')
            cache_all_timeframes: Aggregate and cache all timeframes

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
            Plus bonus columns: buy_volume, sell_volume
        """
        try:
            # Check for existing cache
            existing_data = self.cache.load(symbol, bar_size='1min', max_age_hours=None)

            if existing_data is not None and len(existing_data) > 0:
                # Calculate gap
                last_timestamp = existing_data['time'].max()
                last_datetime = datetime.fromtimestamp(last_timestamp, tz=pytz.UTC)
                now = datetime.now(pytz.UTC)
                gap = now - last_datetime

                if gap.total_seconds() < 120:
                    # Cache is very fresh
                    self.stats['cache_hits'] += 1
                    logger.info(
                        f"[{symbol}] Cache is fresh "
                        f"({gap.total_seconds():.0f}s old) - skipping fetch"
                    )
                    return existing_data

                # Fetch only the gap
                gap_days = max(2, gap.days + 2)
                from_time = last_datetime - timedelta(days=1)  # Overlap for safety

                logger.info(
                    f"[{symbol}] Incremental update: cache has {len(existing_data)} bars, "
                    f"gap is {gap.days}d - fetching from {from_time.strftime('%Y-%m-%d')}"
                )

                new_data = await self._fetch_candles(symbol, from_time=from_time)

                if new_data is not None and len(new_data) > 0:
                    combined = pd.concat([existing_data, new_data], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['time'], keep='last')
                    combined = combined.sort_values('time').reset_index(drop=True)

                    new_bars = len(combined) - len(existing_data)
                    logger.info(
                        f"[{symbol}] Merged: {len(existing_data)} existing + "
                        f"{len(new_data)} new = {len(combined)} total "
                        f"({new_bars} new bars)"
                    )

                    self.cache.save(symbol, combined, bar_size='1min')
                    if cache_all_timeframes:
                        self._cache_all_timeframes(symbol, combined)

                    return combined
                else:
                    logger.warning(f"[{symbol}] No new data - using existing cache")
                    return existing_data
            else:
                # No cache - fetch full duration
                self.stats['cache_misses'] += 1
                duration_days = self._parse_duration_days(duration)
                from_time = datetime.now(pytz.UTC) - timedelta(days=duration_days)

                logger.info(
                    f"[{symbol}] No cache - fetching {duration_days} days from TastyTrade"
                )

                data = await self._fetch_candles(symbol, from_time=from_time)

                if data is not None and len(data) > 0:
                    self.cache.save(symbol, data, bar_size='1min')
                    logger.info(f"[{symbol}] Cached {len(data)} bars")

                    if cache_all_timeframes:
                        self._cache_all_timeframes(symbol, data)

                return data

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            if existing_data is not None:
                return existing_data
            return None

    async def _fetch_candles(
        self,
        symbol: str,
        from_time: datetime,
        period: str = '1m',
    ) -> Optional[pd.DataFrame]:
        """
        Fetch candle data from TastyTrade DXLink.

        Uses subscribe_candle with fromTime to get historical bars
        that transition to live updates.

        Args:
            symbol: Internal symbol (e.g., 'MNQ')
            from_time: Start time for candle data
            period: Candle period (e.g., '1m', '5m', '1h')

        Returns:
            DataFrame with OHLCV + delta columns
        """
        if not self.tt_service.is_authenticated():
            logger.error("TastyTrade not authenticated")
            return None

        await self.tt_service.refresh_session()

        # Resolve streamer symbol
        streamer_sym = self.tt_service.get_streamer_symbol(symbol)
        if not streamer_sym:
            resolved = await self.tt_service.resolve_streamer_symbols([symbol])
            streamer_sym = resolved.get(symbol)
            if not streamer_sym:
                logger.error(f"Could not resolve streamer symbol for {symbol}")
                return None

        # Build candle symbol: e.g., /MNQH26:XCME{=1m}
        candle_symbol = f"{streamer_sym}{{={period}}}"

        try:
            from tastytrade import DXLinkStreamer
            from tastytrade.dxfeed import Candle

            self.stats['total_requests'] += 1

            logger.info(
                f"[{symbol}] Fetching candles: {candle_symbol} "
                f"from {from_time.strftime('%Y-%m-%d %H:%M')}"
            )

            candles = []
            timeout_seconds = 30  # Wait for candle delivery
            batch_complete = asyncio.Event()

            async with DXLinkStreamer(self.tt_service.session) as streamer:
                await streamer.subscribe_candle(
                    [candle_symbol],
                    from_time=from_time,
                )

                # Collect candles with timeout
                # DXLink sends historical candles first, then transitions to live
                last_candle_time = asyncio.get_event_loop().time()

                while True:
                    try:
                        candle = await asyncio.wait_for(
                            streamer.get_event(Candle),
                            timeout=5.0
                        )

                        candles.append(candle)
                        last_candle_time = asyncio.get_event_loop().time()

                        # Log progress periodically
                        if len(candles) % 1000 == 0:
                            logger.info(
                                f"[{symbol}] Received {len(candles)} candles..."
                            )

                    except asyncio.TimeoutError:
                        # No candle for 5 seconds - check if we're done
                        elapsed = asyncio.get_event_loop().time() - last_candle_time
                        if elapsed > 5.0 and len(candles) > 0:
                            # No new candles for 5s = batch complete
                            break
                        elif len(candles) == 0 and elapsed > timeout_seconds:
                            logger.warning(
                                f"[{symbol}] Timeout waiting for candles"
                            )
                            break

            if not candles:
                logger.warning(f"[{symbol}] No candles received")
                return None

            # Convert to DataFrame
            df = self._candles_to_dataframe(candles)

            self.stats['total_bars_fetched'] += len(df)

            logger.info(
                f"[{symbol}] Fetched {len(df)} candles from TastyTrade "
                f"({df['time'].min()} to {df['time'].max()})"
            )

            return df

        except ImportError:
            logger.error("tastytrade package not installed")
            return None
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}", exc_info=True)
            return None

    def _candles_to_dataframe(self, candles) -> pd.DataFrame:
        """
        Convert DXLink Candle events to DataFrame.

        Matches the exact format that the rest of the system expects.
        """
        data = []

        for candle in candles:
            # Get candle timestamp
            candle_time = getattr(candle, 'time', 0)
            if candle_time <= 0:
                continue

            # Convert to seconds if in milliseconds
            if candle_time > 1e12:
                candle_time_s = candle_time // 1000
            else:
                candle_time_s = candle_time

            # Convert to Eastern display timestamp (matches IB convention)
            display_time = self._to_eastern_display_timestamp(candle_time_s)

            open_price = float(getattr(candle, 'open', 0) or 0)
            high_price = float(getattr(candle, 'high', 0) or 0)
            low_price = float(getattr(candle, 'low', 0) or 0)
            close_price = float(getattr(candle, 'close', 0) or 0)
            volume = int(getattr(candle, 'volume', 0) or 0)

            # Skip invalid candles
            if open_price <= 0 or high_price <= 0 or volume <= 0:
                continue

            bar = {
                'time': display_time,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
            }

            # Bonus: bidVolume/askVolume for pre-aggregated delta
            bid_vol = getattr(candle, 'bidVolume', None)
            ask_vol = getattr(candle, 'askVolume', None)
            if bid_vol is not None and ask_vol is not None:
                bar['buy_volume'] = int(ask_vol or 0)
                bar['sell_volume'] = int(bid_vol or 0)

            data.append(bar)

        df = pd.DataFrame(data)

        if df.empty:
            return df

        # Validate and clean
        df = self._validate_and_clean(df)

        # Optimize memory
        df = optimize_dataframe(df)

        return df

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data (same logic as HistoricalDataFetcher)."""
        if df.empty:
            return df

        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['time'], keep='first')
        if len(df) < original_len:
            logger.warning(f"Removed {original_len - len(df)} duplicate bars")

        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)

        # Validate OHLC relationships
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['open'] > df['high']) | (df['open'] < df['low']) |
            (df['close'] > df['high']) | (df['close'] < df['low'])
        )

        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning(f"Removed {invalid_count} bars with invalid OHLC")
            df = df[~invalid_mask].reset_index(drop=True)

        return df

    @staticmethod
    def _to_eastern_display_timestamp(utc_seconds: int) -> int:
        """
        Convert UTC timestamp to Eastern display timestamp.

        Matches the IB system's convention exactly.
        """
        utc_dt = datetime.fromtimestamp(utc_seconds, tz=pytz.UTC)
        eastern_dt = utc_dt.astimezone(EASTERN)

        display_dt = datetime(
            eastern_dt.year, eastern_dt.month, eastern_dt.day,
            eastern_dt.hour, eastern_dt.minute, eastern_dt.second,
            tzinfo=pytz.UTC
        )
        return int(display_dt.timestamp())

    def _cache_all_timeframes(self, symbol: str, df: pd.DataFrame):
        """Aggregate and cache all timeframes from 1-minute data."""
        for tf in ['5min', '15min', '30min', '1H', '2H', '4H']:
            try:
                aggregated = self._aggregate_bars(df, tf)
                self.cache.save(symbol, aggregated, bar_size=tf)
                logger.debug(f"[{symbol}] Cached {len(aggregated)} {tf} bars")
            except Exception as e:
                logger.error(f"[{symbol}] Error aggregating {tf}: {e}")

    @staticmethod
    def _aggregate_bars(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to higher timeframe.

        Same logic as HistoricalDataFetcher.aggregate_bars().
        """
        if df.empty:
            return df

        timeframe_map = {
            '5min': 5, '15min': 15, '30min': 30,
            '1H': 60, '2H': 120, '4H': 240,
        }

        minutes = timeframe_map.get(target_timeframe)
        if not minutes:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")

        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['time'], unit='s')
        df_copy.set_index('timestamp', inplace=True)

        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }

        # Include delta columns if available
        if 'buy_volume' in df_copy.columns:
            agg_dict['buy_volume'] = 'sum'
        if 'sell_volume' in df_copy.columns:
            agg_dict['sell_volume'] = 'sum'

        aggregated = df_copy.resample(
            f'{minutes}min', label='left', closed='left'
        ).agg(agg_dict).dropna()

        aggregated['time'] = aggregated.index.astype(int) // 10**9
        aggregated.reset_index(drop=True, inplace=True)

        cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        if 'buy_volume' in aggregated.columns:
            cols.extend(['buy_volume', 'sell_volume'])
        result = aggregated[cols].copy()

        return result

    @staticmethod
    def _parse_duration_days(duration: str) -> int:
        """Parse duration string to days."""
        duration = duration.strip().upper()
        if 'D' in duration:
            return int(duration.split('D')[0].strip().split()[-1])
        elif 'W' in duration:
            return int(duration.split('W')[0].strip().split()[-1]) * 7
        elif 'M' in duration:
            return int(duration.split('M')[0].strip().split()[-1]) * 30
        elif 'Y' in duration:
            return int(duration.split('Y')[0].strip().split()[-1]) * 365
        return 60  # Default

    def get_statistics(self) -> Dict[str, Any]:
        """Get fetching statistics."""
        return dict(self.stats)
