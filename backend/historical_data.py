"""
Historical Data Fetching System

Downloads historical futures data from IB Gateway with proper rate limiting and caching.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd
from ib_insync import IB, Contract, BarDataList, util
import pytz

from .pacing import PacingManager, HistoricalRequest
from .cache import DataCache, optimize_dataframe
from .ib_service import ib_request_with_retry

logger = logging.getLogger(__name__)


class HistoricalDataFetcher:
    """
    Fetches historical data with rate limiting and caching.

    Features:
    - Chunks large requests into daily segments
    - Automatic pacing to avoid violations
    - Caching for fast reloads
    - Incremental updates
    - Data validation
    """

    def __init__(
        self,
        ib: IB,
        cache: Optional[DataCache] = None,
        pacing_manager: Optional[PacingManager] = None
    ):
        """
        Initialize historical data fetcher.

        Args:
            ib: Connected IB instance
            cache: Data cache instance (creates new if None)
            pacing_manager: Pacing manager instance (creates new if None)
        """
        self.ib = ib
        self.cache = cache or DataCache()
        self.pacing_manager = pacing_manager or PacingManager()

        self.stats = {
            'total_bars_fetched': 0,
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }

    async def fetch_year(
        self,
        contract: Contract,
        end_date: Optional[datetime] = None,
        cache_all_timeframes: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch 1 year of 1-minute historical data with INCREMENTAL updates.

        Smart caching behavior:
        1. If cache exists, only fetches the gap between last cached bar and now
        2. If no cache, fetches full year in daily chunks
        3. Merges new data with existing cache
        4. Aggregates and caches all timeframes

        Args:
            contract: Futures contract
            end_date: End date for data (default: now)
            cache_all_timeframes: If True, aggregate and cache all timeframes

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        symbol = contract.symbol

        # Check for existing cached data (no age limit - we'll do incremental update)
        existing_data = self.cache.load(symbol, bar_size='1min', max_age_hours=None)

        if existing_data is not None and len(existing_data) > 0:
            # We have cached data - calculate the gap
            last_timestamp = existing_data['time'].max()
            last_datetime = datetime.fromtimestamp(last_timestamp, tz=pytz.UTC)
            now = datetime.now(pytz.UTC)

            gap = now - last_datetime
            gap_seconds = gap.total_seconds()

            if gap_seconds < 120:
                # Cache is very fresh (less than 2 minutes), no need to fetch
                self.stats['cache_hits'] += 1
                logger.info(f"[{symbol}] Cache is fresh ({gap_seconds:.0f}s old) - skipping fetch")
                return existing_data

            # Calculate gap in days with margin for session boundaries
            # Always fetch at least 2 days to handle RTH/overnight transitions
            gap_days = max(2, gap.days + 2)

            if gap_days <= 5:
                # Small gap - use single request via fetch_recent logic
                logger.info(
                    f"[{symbol}] Incremental update: cache has {len(existing_data)} bars, "
                    f"gap is {gap.days}d - fetching {gap_days} days"
                )
                return await self.fetch_recent(
                    contract,
                    duration=f'{gap_days} D',
                    cache_all_timeframes=cache_all_timeframes
                )
            else:
                # Larger gap - use chunked fetch but only for the gap period
                logger.info(
                    f"[{symbol}] Incremental update: cache has {len(existing_data)} bars, "
                    f"gap is {gap_days} days - using chunked fetch"
                )
                self.stats['cache_misses'] += 1

                # Fetch only the gap
                new_data = await self._fetch_year_chunked(
                    contract,
                    end_date=end_date,
                    start_date=last_datetime  # Only fetch from last cached date
                )

                if new_data is not None and len(new_data) > 0:
                    # Merge with existing data
                    combined = pd.concat([existing_data, new_data], ignore_index=True)
                    combined = combined.drop_duplicates(subset=['time'], keep='last')
                    combined = combined.sort_values('time').reset_index(drop=True)

                    new_bars = len(combined) - len(existing_data)
                    logger.info(
                        f"[{symbol}] Merged: {len(existing_data)} existing + {len(new_data)} new "
                        f"= {len(combined)} total ({new_bars} new bars)"
                    )

                    # Save combined data
                    self.cache.save(symbol, combined, bar_size='1min')

                    # Aggregate timeframes
                    if cache_all_timeframes:
                        self._cache_all_timeframes(symbol, combined)

                    return combined
                else:
                    logger.warning(f"[{symbol}] No new data fetched - using existing cache")
                    return existing_data
        else:
            # No cache - fetch full duration (chunked)
            self.stats['cache_misses'] += 1
            logger.info(f"[{symbol}] No cache found - fetching historical data (chunked)...")

            data = await self._fetch_year_chunked(contract, end_date)

            if data is not None and len(data) > 0:
                # Save to cache
                self.cache.save(symbol, data, bar_size='1min')
                logger.info(f"[{symbol}] Cached {len(data)} bars")

                # Aggregate timeframes
                if cache_all_timeframes:
                    self._cache_all_timeframes(symbol, data)

            return data

    def _cache_all_timeframes(self, symbol: str, df: pd.DataFrame):
        """Aggregate and cache all timeframes from 1-minute data."""
        for tf in ['5min', '15min', '30min', '1H', '2H', '4H']:
            try:
                aggregated = self.aggregate_bars(df, tf)
                self.cache.save(symbol, aggregated, bar_size=tf)
                logger.debug(f"[{symbol}] Cached {len(aggregated)} {tf} bars")
            except Exception as e:
                logger.error(f"[{symbol}] Error aggregating {tf} bars: {e}")

    async def _fetch_year_chunked(
        self,
        contract: Contract,
        end_date: Optional[datetime] = None,
        start_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data in daily chunks with rate limiting.

        Args:
            contract: Futures contract
            end_date: End date for data (default: now)
            start_date: Start date for data (default: 1 year before end_date)

        Returns:
            Combined DataFrame
        """
        end = end_date or datetime.now(pytz.UTC)
        if start_date:
            start = start_date
        else:
            # Default to 60 days, not 365
            start = end - timedelta(days=60)

        all_bars = []
        current_end = end
        request_count = 0
        errors_count = 0
        max_consecutive_errors = 3

        days_to_fetch = (end - start).days
        logger.info(
            f"[{contract.symbol}] Fetching {days_to_fetch} days of data "
            f"({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})"
        )

        while current_end > start and errors_count < max_consecutive_errors:
            try:
                # Create pacing request
                pacing_request = HistoricalRequest(
                    contract_id=contract.conId or 0,
                    end_datetime=current_end.strftime('%Y%m%d-%H:%M:%S'),
                    duration='1 D',
                    bar_size='1 min',
                    what_to_show='TRADES',
                    timestamp=datetime.now().timestamp()
                )

                # Wait for pacing if needed
                wait_time = await self.pacing_manager.wait_if_needed(pacing_request)

                if wait_time > 0:
                    logger.debug(f"Pacing delay: {wait_time:.1f}s")

                # Request historical data with retry
                bars = await ib_request_with_retry(
                    self.ib.reqHistoricalDataAsync,
                    contract,
                    endDateTime=current_end.strftime('%Y%m%d-%H:%M:%S'),
                    durationStr='1 D',
                    barSizeSetting='1 min',
                    whatToShow='TRADES',
                    useRTH=False,  # Essential for futures - include extended hours
                    formatDate=1,
                    keepUpToDate=False,
                    timeout=120,
                    max_retries=3
                )

                request_count += 1
                self.stats['total_requests'] += 1

                if bars and len(bars) > 0:
                    # Prepend older data
                    all_bars = list(bars) + all_bars
                    self.stats['total_bars_fetched'] += len(bars)

                    # Move to day before first bar received
                    first_bar_time = self._parse_bar_date(bars[0].date)
                    current_end = first_bar_time - timedelta(minutes=1)

                    logger.info(
                        f"Fetched {len(bars)} bars "
                        f"(total: {len(all_bars)}, "
                        f"requests: {request_count})"
                    )

                    # Reset error count on success
                    errors_count = 0

                else:
                    # No data for this period, move back a day
                    current_end -= timedelta(days=1)
                    logger.debug(f"No data, moving to {current_end}")

                # Long pause every 50 requests to avoid pacing violations
                if request_count % 50 == 0:
                    logger.info(f"Fetched {len(all_bars)} bars, pausing for 60s...")
                    await asyncio.sleep(60)

            except Exception as e:
                errors_count += 1
                logger.error(
                    f"Error fetching historical data (error {errors_count}/"
                    f"{max_consecutive_errors}): {e}"
                )

                # If we have some data, continue from error point
                if all_bars:
                    current_end -= timedelta(days=1)
                    await asyncio.sleep(10)  # Wait before retry
                else:
                    # No data yet, this is a critical error
                    if errors_count >= max_consecutive_errors:
                        logger.error("Too many consecutive errors, aborting")
                        return None

        if not all_bars:
            logger.error("No historical data fetched")
            return None

        # Convert to DataFrame
        df = self._bars_to_dataframe(all_bars)

        logger.info(
            f"Completed: fetched {len(df)} total bars in {request_count} requests"
        )

        # Log pacing statistics
        pacing_stats = self.pacing_manager.get_statistics()
        logger.info(
            f"Pacing stats: {pacing_stats['total_delays']} delays, "
            f"avg {pacing_stats['average_delay_seconds']:.1f}s"
        )

        return df

    def _parse_bar_date(self, date) -> datetime:
        """
        Parse bar date (handles both datetime and string formats).

        IB Gateway returns timezone-aware datetime objects in UTC.
        We just need to ensure they're timezone-aware before converting to timestamps.
        """
        if isinstance(date, datetime):
            # IB returns timezone-aware datetimes in UTC - use as-is
            # Don't localize if already timezone-aware
            return date
        elif isinstance(date, str):
            # Format: 'YYYYMMDD  HH:MM:SS'
            # String format is typically in UTC
            naive_dt = datetime.strptime(date, '%Y%m%d  %H:%M:%S')
            # Assume UTC for string dates from IB
            return pytz.UTC.localize(naive_dt)
        else:
            raise ValueError(f"Unknown date format: {type(date)}")

    def _bars_to_dataframe(self, bars: List) -> pd.DataFrame:
        """
        Convert IB bars to DataFrame with Eastern time display.

        CRITICAL: LightweightCharts displays timestamps in the browser's local timezone.
        Since the user may be in UTC, we need to convert to Eastern time for display.

        Args:
            bars: List of IB Bar objects

        Returns:
            DataFrame with standardized format
        """
        data = []
        eastern = pytz.timezone('US/Eastern')

        for i, bar in enumerate(bars):
            # Parse IB's timestamp
            bar_time = self._parse_bar_date(bar.date)

            # Convert to Eastern time
            bar_eastern = bar_time.astimezone(eastern)

            # Create "display timestamp" - treat Eastern time as if it were UTC
            # This makes the chart show 16:59 instead of 21:59 for a 4:59 PM Eastern bar
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

            # Debug logging for first and last bars
            if i == 0 or i == len(bars) - 1:
                logger.info(
                    f"Bar {i}: IB={bar.date}, Eastern={bar_eastern.strftime('%Y-%m-%d %H:%M %Z')}, "
                    f"display={display_time.strftime('%Y-%m-%d %H:%M')}, timestamp={timestamp}"
                )

            data.append({
                'time': timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            })

        df = pd.DataFrame(data)

        # Validate data
        df = self._validate_and_clean(df)

        # Optimize memory usage
        df = optimize_dataframe(df)

        return df

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.

        Args:
            df: DataFrame to validate

        Returns:
            Cleaned DataFrame
        """
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
            logger.warning(f"Found {invalid_count} bars with invalid OHLC relationships")

            # Log a few examples
            invalid_bars = df[invalid_mask].head()
            for idx, row in invalid_bars.iterrows():
                logger.debug(
                    f"Invalid bar at {row['time']}: "
                    f"O={row['open']:.2f} H={row['high']:.2f} "
                    f"L={row['low']:.2f} C={row['close']:.2f}"
                )

            # Remove invalid bars
            df = df[~invalid_mask].reset_index(drop=True)

        # Check for gaps
        df['time_diff'] = df['time'].diff()
        expected_diff = 60  # 1 minute in seconds
        gaps = df[df['time_diff'] > expected_diff * 2]  # Allow some tolerance

        if len(gaps) > 0:
            logger.info(f"Found {len(gaps)} gaps in data (normal for non-trading hours)")

        df = df.drop(columns=['time_diff'])

        return df

    def _parse_duration_days(self, duration: str) -> int:
        """Parse duration string to number of days."""
        duration = duration.strip().upper()
        if duration.endswith('D'):
            return int(duration[:-1].strip())
        elif duration.endswith('W'):
            return int(duration[:-1].strip()) * 7
        elif duration.endswith('M'):
            return int(duration[:-1].strip()) * 30
        elif duration.endswith('Y'):
            return int(duration[:-1].strip()) * 365
        else:
            # Assume days if no suffix
            return int(duration.split()[0])

    async def fetch_recent(
        self,
        contract: Contract,
        duration: str = '1 D',
        bar_size: str = '1 min',
        cache_all_timeframes: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Fetch recent data with INCREMENTAL updates - only fetches missing data.

        Smart behavior:
        1. If cache exists: only fetches gap between last bar and now
        2. If no cache and duration > 5 days: uses chunked fetching
        3. If no cache and duration <= 5 days: single request

        Args:
            contract: Futures contract
            duration: Duration string (e.g., '1 D', '60 D') - used if NO cache exists
            bar_size: Bar size (e.g., '1 min', '5 mins')
            cache_all_timeframes: If True, aggregate and cache all timeframes

        Returns:
            DataFrame with all data (existing + new)
        """
        symbol = contract.symbol

        # Parse duration upfront for logging
        duration_days = self._parse_duration_days(duration)
        logger.info(f"[{symbol}] fetch_recent called: duration='{duration}' ({duration_days} days)")

        try:
            # Check for existing cached data
            existing_data = self.cache.load(symbol, bar_size='1min', max_age_hours=None)
            metadata = self.cache.get_metadata(symbol, bar_size='1min')

            if existing_data is not None and len(existing_data) > 0 and metadata:
                # We have existing data - do incremental update
                last_timestamp = existing_data['time'].max()
                last_datetime = datetime.fromtimestamp(last_timestamp, tz=pytz.UTC)
                now = datetime.now(pytz.UTC)

                # Calculate gap in days
                gap = now - last_datetime
                gap_seconds = gap.total_seconds()

                if gap_seconds < 120:
                    # Data is very fresh (less than 2 minutes old), no need to fetch
                    logger.info(f"[{symbol}] Cache is fresh (last update: {gap_seconds:.0f}s ago) - skipping fetch")
                    return existing_data

                # Calculate gap in days with extra margin
                # Always fetch at least 2 days to handle session boundaries properly
                # (IB session data can have gaps at RTH/overnight transitions)
                gap_days = max(2, gap.days + 2)  # Add 2 days margin for safety

                # Determine fetch approach based on gap
                if gap_days <= 5:
                    fetch_duration = f'{gap_days} D'
                    use_chunked = False
                else:
                    # Large gap - use chunked fetching
                    use_chunked = True
                    start_date = last_datetime

                logger.info(
                    f"[{symbol}] Incremental update: cache has {len(existing_data)} bars "
                    f"(last: {last_datetime.strftime('%Y-%m-%d %H:%M')}), "
                    f"gap is {gap.days}d {gap_seconds % 86400 / 3600:.1f}h, fetching {gap_days} days"
                )

                if use_chunked:
                    # Use chunked fetch for large gaps
                    new_data = await self._fetch_year_chunked(contract, start_date=start_date)
                    if new_data is not None and len(new_data) > 0:
                        combined = pd.concat([existing_data, new_data], ignore_index=True)
                        combined = combined.drop_duplicates(subset=['time'], keep='last')
                        combined = combined.sort_values('time').reset_index(drop=True)

                        # Save and aggregate
                        self.cache.save(symbol, combined, bar_size='1min')
                        if cache_all_timeframes:
                            self._cache_all_timeframes(symbol, combined)
                        return combined
                    return existing_data
            else:
                # No existing data - determine if we need chunked fetch
                existing_data = None
                duration_days = self._parse_duration_days(duration)

                if duration_days > 5:
                    # Large duration - use chunked fetching
                    start_date = datetime.now(pytz.UTC) - timedelta(days=duration_days)
                    logger.info(f"[{symbol}] No cache - using CHUNKED fetch for {duration_days} days (start: {start_date.strftime('%Y-%m-%d')})")
                    data = await self._fetch_year_chunked(contract, start_date=start_date)

                    if data is not None and len(data) > 0:
                        self.cache.save(symbol, data, bar_size='1min')
                        logger.info(f"[{symbol}] Saved {len(data)} bars to cache")
                        if cache_all_timeframes:
                            self._cache_all_timeframes(symbol, data)
                    return data
                else:
                    fetch_duration = duration
                    logger.info(f"[{symbol}] No cache found - fetching {duration}")

            # Create pacing request
            pacing_request = HistoricalRequest(
                contract_id=contract.conId or 0,
                end_datetime='',
                duration=fetch_duration,
                bar_size=bar_size,
                what_to_show='TRADES',
                timestamp=datetime.now().timestamp()
            )

            # Wait for pacing
            await self.pacing_manager.wait_if_needed(pacing_request)

            # Request data using async method
            bars = await ib_request_with_retry(
                self.ib.reqHistoricalDataAsync,
                contract,
                endDateTime='',
                durationStr=fetch_duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1,
                timeout=60
            )

            if bars:
                new_df = self._bars_to_dataframe(bars)
                logger.info(f"[{symbol}] Fetched {len(new_df)} new bars from IB")

                # Merge with existing data if we have it
                if existing_data is not None:
                    # Combine old and new data
                    combined = pd.concat([existing_data, new_df], ignore_index=True)

                    # Remove duplicates (keep newer data if timestamps overlap)
                    combined = combined.drop_duplicates(subset=['time'], keep='last')

                    # Sort by time
                    combined = combined.sort_values('time').reset_index(drop=True)

                    new_bars_added = len(combined) - len(existing_data)
                    logger.info(
                        f"[{symbol}] Merged: {len(existing_data)} existing + {len(new_df)} fetched "
                        f"= {len(combined)} total ({new_bars_added} new bars added)"
                    )

                    final_df = combined
                else:
                    final_df = new_df

                # Save combined data to cache
                self.cache.save(symbol, final_df, bar_size='1min')
                logger.info(f"[{symbol}] Saved {len(final_df)} bars to cache")

                # Aggregate and cache all timeframes
                if cache_all_timeframes and bar_size == '1 min':
                    for tf in ['5min', '15min', '30min', '1H', '2H', '4H']:
                        try:
                            aggregated = self.aggregate_bars(final_df, tf)
                            self.cache.save(symbol, aggregated, bar_size=tf)
                            logger.debug(f"[{symbol}] Cached {len(aggregated)} {tf} bars")
                        except Exception as e:
                            logger.error(f"Error aggregating {tf} bars: {e}")

                return final_df
            else:
                # No new data from IB, return existing if available
                if existing_data is not None:
                    logger.warning(f"[{symbol}] No new data from IB - using existing cache")
                    return existing_data
                logger.warning(f"[{symbol}] No data available")
                return None

        except Exception as e:
            logger.error(f"Error fetching recent data for {symbol}: {e}")
            # Return existing cache on error
            if existing_data is not None:
                logger.info(f"[{symbol}] Returning existing cache due to fetch error")
                return existing_data
            return None

    def aggregate_bars(
        self,
        df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Aggregate 1-minute bars to higher timeframe.

        Args:
            df: DataFrame with 1-minute OHLCV data
            target_timeframe: Target timeframe ('5min', '15min', '30min', '1H', '2H', '4H')

        Returns:
            Aggregated DataFrame with same structure
        """
        if df.empty:
            return df

        # Map timeframe to minutes
        timeframe_map = {
            '5min': 5,
            '15min': 15,
            '30min': 30,
            '1H': 60,
            '2H': 120,
            '4H': 240
        }

        minutes = timeframe_map.get(target_timeframe)
        if not minutes:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")

        # Convert timestamp to datetime for resampling
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['time'], unit='s')
        df_copy.set_index('timestamp', inplace=True)

        # Resample to target timeframe
        # label='left' means bar is labeled with start time
        # closed='left' means interval is closed on left side [09:30, 09:35)
        aggregated = df_copy.resample(f'{minutes}min', label='left', closed='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Convert back to unix timestamp
        aggregated['time'] = aggregated.index.astype(int) // 10**9
        aggregated.reset_index(drop=True, inplace=True)

        result = aggregated[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

        logger.info(
            f"Aggregated {len(df)} 1-min bars to {len(result)} {target_timeframe} bars"
        )

        return result

    def get_statistics(self) -> dict:
        """Get fetching statistics"""
        pacing_stats = self.pacing_manager.get_statistics()

        return {
            **self.stats,
            'pacing': pacing_stats
        }


if __name__ == '__main__':
    # Example usage (requires IB Gateway connection)
    async def test():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        from .contracts import get_current_contract
        from .ib_service import IBConnectionManager

        # Connect to IB Gateway
        manager = IBConnectionManager(host='127.0.0.1', port=4002)
        await manager.connect()

        if not manager.is_connected():
            print("Failed to connect to IB Gateway")
            return

        # Get contract
        contract = get_current_contract('MNQ')
        await manager.ib.qualifyContractsAsync(contract)

        print(f"Fetching historical data for {contract.symbol}...")

        # Create fetcher
        fetcher = HistoricalDataFetcher(manager.ib)

        # Fetch 1 year of data
        data = await fetcher.fetch_year(contract)

        if data is not None:
            print(f"\nSuccess! Fetched {len(data)} bars")
            print(f"Date range: {data['time'].min()} to {data['time'].max()}")
            print(f"\nFirst few bars:")
            print(data.head())
            print(f"\nStatistics:")
            print(fetcher.get_statistics())
        else:
            print("Failed to fetch data")

        # Disconnect
        manager.disconnect()

    # Run test
    util.startLoop()
    asyncio.run(test())
