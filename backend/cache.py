"""
Data caching system using Parquet format

Efficiently stores and loads historical market data with metadata tracking.
"""

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json

logger = logging.getLogger(__name__)


class DataCache:
    """
    Cache manager for historical market data.

    Features:
    - Stores data in Parquet format (fast, compressed)
    - Tracks metadata (fetch time, bar count, date range)
    - Validates cache freshness
    - Supports incremental updates
    """

    def __init__(self, cache_dir: str = './data/cache'):
        """
        Initialize data cache.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Cache initialized at {self.cache_dir}")

    def _get_cache_path(self, symbol: str, bar_size: str = '1min') -> Path:
        """Get cache file path for symbol"""
        filename = f"{symbol}_{bar_size}.parquet"
        return self.cache_dir / filename

    def _get_metadata_path(self, symbol: str, bar_size: str = '1min') -> Path:
        """Get metadata file path for symbol"""
        filename = f"{symbol}_{bar_size}_metadata.json"
        return self.cache_dir / filename

    def save(
        self,
        symbol: str,
        data: pd.DataFrame,
        bar_size: str = '1min',
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Save data to cache.

        Args:
            symbol: Contract symbol (e.g., 'MNQ')
            data: DataFrame with columns: time, open, high, low, close, volume
            bar_size: Bar size (e.g., '1min', '5min', '1H')
            metadata: Optional additional metadata

        Returns:
            True if saved successfully
        """
        try:
            cache_path = self._get_cache_path(symbol, bar_size)
            metadata_path = self._get_metadata_path(symbol, bar_size)

            # Validate data
            required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                missing = [c for c in required_columns if c not in data.columns]
                raise ValueError(f"Missing required columns: {missing}")

            # Save data
            data.to_parquet(cache_path, compression='snappy', index=False)

            # Create metadata (convert numpy types to native Python types)
            meta = {
                'symbol': symbol,
                'bar_size': bar_size,
                'fetch_time': datetime.now().isoformat(),
                'bar_count': int(len(data)),
                'first_date': str(data['time'].min()) if len(data) > 0 else None,
                'last_date': str(data['time'].max()) if len(data) > 0 else None,
                'file_size_bytes': int(cache_path.stat().st_size),
            }

            # Add custom metadata
            if metadata:
                meta.update(metadata)

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(meta, f, indent=2)

            logger.info(
                f"Cached {len(data)} bars for {symbol} ({bar_size}) "
                f"- {cache_path.stat().st_size / 1024 / 1024:.2f} MB"
            )

            return True

        except Exception as e:
            logger.error(f"Error saving cache for {symbol}: {e}")
            return False

    def load(
        self,
        symbol: str,
        bar_size: str = '1min',
        max_age_hours: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load data from cache.

        Args:
            symbol: Contract symbol
            bar_size: Bar size
            max_age_hours: Maximum cache age in hours (None = no limit)

        Returns:
            DataFrame if cache exists and is valid, None otherwise
        """
        try:
            cache_path = self._get_cache_path(symbol, bar_size)
            metadata_path = self._get_metadata_path(symbol, bar_size)

            if not cache_path.exists():
                logger.debug(f"No cache found for {symbol} ({bar_size})")
                return None

            # Load and validate metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Check cache age if max_age specified
                if max_age_hours is not None:
                    fetch_time = datetime.fromisoformat(metadata['fetch_time'])
                    age = datetime.now() - fetch_time

                    if age > timedelta(hours=max_age_hours):
                        logger.info(
                            f"Cache for {symbol} ({bar_size}) is stale "
                            f"({age.total_seconds() / 3600:.1f}h old)"
                        )
                        return None

                logger.info(
                    f"Loading cache for {symbol} ({bar_size}): "
                    f"{metadata['bar_count']} bars"
                )
            else:
                logger.warning(f"Cache metadata missing for {symbol} ({bar_size})")

            # Load data
            data = pd.read_parquet(cache_path)

            logger.debug(f"Loaded {len(data)} bars from cache")

            return data

        except Exception as e:
            logger.error(f"Error loading cache for {symbol}: {e}")
            return None

    def get_metadata(self, symbol: str, bar_size: str = '1min') -> Optional[Dict]:
        """Get cache metadata"""
        try:
            metadata_path = self._get_metadata_path(symbol, bar_size)

            if not metadata_path.exists():
                return None

            with open(metadata_path, 'r') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error loading metadata for {symbol}: {e}")
            return None

    def is_fresh(
        self,
        symbol: str,
        bar_size: str = '1min',
        max_age_hours: int = 24
    ) -> bool:
        """
        Check if cache is fresh.

        Args:
            symbol: Contract symbol
            bar_size: Bar size
            max_age_hours: Maximum acceptable age in hours

        Returns:
            True if cache exists and is fresh
        """
        metadata = self.get_metadata(symbol, bar_size)

        if not metadata:
            return False

        fetch_time = datetime.fromisoformat(metadata['fetch_time'])
        age = datetime.now() - fetch_time

        return age <= timedelta(hours=max_age_hours)

    def delete(self, symbol: str, bar_size: str = '1min') -> bool:
        """Delete cache for symbol"""
        try:
            cache_path = self._get_cache_path(symbol, bar_size)
            metadata_path = self._get_metadata_path(symbol, bar_size)

            deleted = False

            if cache_path.exists():
                cache_path.unlink()
                deleted = True

            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True

            if deleted:
                logger.info(f"Deleted cache for {symbol} ({bar_size})")

            return deleted

        except Exception as e:
            logger.error(f"Error deleting cache for {symbol}: {e}")
            return False

    def list_cached_symbols(self) -> List[Dict]:
        """List all cached symbols with metadata"""
        cached = []

        for parquet_file in self.cache_dir.glob('*.parquet'):
            # Parse filename: symbol_barsize.parquet
            parts = parquet_file.stem.split('_')
            if len(parts) < 2:
                continue

            symbol = parts[0]
            bar_size = '_'.join(parts[1:])

            metadata = self.get_metadata(symbol, bar_size)

            if metadata:
                cached.append(metadata)

        return cached

    def get_cache_size(self) -> Dict:
        """Get total cache size information"""
        total_size = 0
        file_count = 0

        for file in self.cache_dir.glob('*.parquet'):
            total_size += file.stat().st_size
            file_count += 1

        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / 1024 / 1024,
            'file_count': file_count
        }

    def clear_all(self) -> int:
        """Clear all cache files"""
        count = 0

        for file in self.cache_dir.glob('*'):
            try:
                file.unlink()
                count += 1
            except Exception as e:
                logger.error(f"Error deleting {file}: {e}")

        logger.info(f"Cleared {count} cache files")
        return count


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage.

    Converts float64 to float32 where appropriate,
    and ensures efficient data types.

    Args:
        df: DataFrame to optimize

    Returns:
        Optimized DataFrame
    """
    df = df.copy()

    # Convert price columns to float32 (sufficient precision for futures)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].astype('float32')

    # Ensure volume is int32
    if 'volume' in df.columns:
        df['volume'] = df['volume'].astype('int32')

    # Ensure time is int64 (Unix timestamp)
    if 'time' in df.columns:
        df['time'] = df['time'].astype('int64')

    return df


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    cache = DataCache()

    # Create sample data
    sample_data = pd.DataFrame({
        'time': [1640000000 + i * 60 for i in range(1000)],
        'open': [100.0 + i * 0.1 for i in range(1000)],
        'high': [100.5 + i * 0.1 for i in range(1000)],
        'low': [99.5 + i * 0.1 for i in range(1000)],
        'close': [100.2 + i * 0.1 for i in range(1000)],
        'volume': [1000 + i * 10 for i in range(1000)],
    })

    # Save to cache
    print("\nSaving to cache...")
    cache.save('MNQ', sample_data, '1min')

    # Load from cache
    print("\nLoading from cache...")
    loaded_data = cache.load('MNQ', '1min')
    print(f"Loaded {len(loaded_data)} bars")

    # Check freshness
    print(f"\nIs fresh? {cache.is_fresh('MNQ', '1min', max_age_hours=24)}")

    # Get metadata
    print("\nMetadata:")
    metadata = cache.get_metadata('MNQ', '1min')
    print(json.dumps(metadata, indent=2))

    # List all cached
    print("\nAll cached symbols:")
    for item in cache.list_cached_symbols():
        print(f"  {item['symbol']} ({item['bar_size']}): {item['bar_count']} bars")

    # Get cache size
    size_info = cache.get_cache_size()
    print(f"\nTotal cache size: {size_info['total_size_mb']:.2f} MB")
