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
        use_cache: bool = True,
        cache_max_age_hours: int = 24
    ) -> Optional[pd.DataFrame]:
        """
        Fetch 1 year of 1-minute historical data.

        Args:
            contract: Futures contract
            end_date: End date for data (default: now)
            use_cache: Whether to use cached data if available
            cache_max_age_hours: Maximum cache age in hours

        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        symbol = contract.symbol

        # Try cache first
        if use_cache:
            cached_data = self.cache.load(
                symbol,
                bar_size='1min',
                max_age_hours=cache_max_age_hours
            )

            if cached_data is not None:
                self.stats['cache_hits'] += 1
                logger.info(f"Using cached data for {symbol} ({len(cached_data)} bars)")
                return cached_data

        self.stats['cache_misses'] += 1

        # Fetch from IB Gateway
        logger.info(f"Fetching 1 year of data for {symbol} from IB Gateway...")

        data = await self._fetch_year_chunked(contract, end_date)

        if data is not None and len(data) > 0:
            # Save to cache
            self.cache.save(symbol, data, bar_size='1min')
            logger.info(f"Successfully fetched and cached {len(data)} bars for {symbol}")

        return data

    async def _fetch_year_chunked(
        self,
        contract: Contract,
        end_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch 1 year of data in daily chunks with rate limiting.

        Args:
            contract: Futures contract
            end_date: End date for data

        Returns:
            Combined DataFrame
        """
        end = end_date or datetime.now()
        start = end - timedelta(days=365)

        all_bars = []
        current_end = end
        request_count = 0
        errors_count = 0
        max_consecutive_errors = 3

        logger.info(
            f"Fetching data from {start.strftime('%Y-%m-%d')} "
            f"to {end.strftime('%Y-%m-%d')}"
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
        """Parse bar date (handles both datetime and string formats)"""
        if isinstance(date, datetime):
            return date
        elif isinstance(date, str):
            # Format: 'YYYYMMDD  HH:MM:SS'
            return datetime.strptime(date, '%Y%m%d  %H:%M:%S')
        else:
            raise ValueError(f"Unknown date format: {type(date)}")

    def _bars_to_dataframe(self, bars: List) -> pd.DataFrame:
        """
        Convert IB bars to DataFrame.

        Args:
            bars: List of IB Bar objects

        Returns:
            DataFrame with standardized format
        """
        data = []

        for bar in bars:
            # Parse date to Unix timestamp
            bar_time = self._parse_bar_date(bar.date)
            timestamp = int(bar_time.timestamp())

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

    async def fetch_recent(
        self,
        contract: Contract,
        duration: str = '1 D',
        bar_size: str = '1 min'
    ) -> Optional[pd.DataFrame]:
        """
        Fetch recent data (useful for incremental cache updates).

        Args:
            contract: Futures contract
            duration: Duration string (e.g., '1 D', '1 W')
            bar_size: Bar size (e.g., '1 min', '5 mins')

        Returns:
            DataFrame with recent data
        """
        try:
            # Create pacing request
            pacing_request = HistoricalRequest(
                contract_id=contract.conId or 0,
                end_datetime='',
                duration=duration,
                bar_size=bar_size,
                what_to_show='TRADES',
                timestamp=datetime.now().timestamp()
            )

            # Wait for pacing
            await self.pacing_manager.wait_if_needed(pacing_request)

            # Request data using async method for proper event loop integration
            bars = await ib_request_with_retry(
                self.ib.reqHistoricalDataAsync,
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1,
                timeout=60
            )

            if bars:
                df = self._bars_to_dataframe(bars)
                logger.info(f"Fetched {len(df)} recent bars for {contract.symbol}")

                # Save to cache
                self.cache.save(contract.symbol, df, bar_size='1min')
                logger.info(f"Saved {len(df)} bars to cache for {contract.symbol}")

                return df
            else:
                logger.warning(f"No recent data for {contract.symbol}")
                return None

        except Exception as e:
            logger.error(f"Error fetching recent data: {e}")
            return None

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
