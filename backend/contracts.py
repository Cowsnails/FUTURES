"""
Futures contract definitions for MNQ, MES, and MGC

This module provides contract specifications and helper functions for
Interactive Brokers futures contracts.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from ib_insync import Future, ContFuture, IB
import logging

logger = logging.getLogger(__name__)


# Contract Specifications
CONTRACT_SPECS = {
    'MNQ': {
        'name': 'Micro E-mini Nasdaq-100',
        'exchange': 'CME',
        'currency': 'USD',
        'multiplier': 2,
        'tick_size': 0.25,
        'tick_value': 0.50,
        'trading_hours': 'Nearly 24/5',
    },
    'MES': {
        'name': 'Micro E-mini S&P 500',
        'exchange': 'CME',
        'currency': 'USD',
        'multiplier': 5,
        'tick_size': 0.25,
        'tick_value': 1.25,
        'trading_hours': 'Nearly 24/5',
    },
    'MGC': {
        'name': 'Micro Gold',
        'exchange': 'COMEX',
        'currency': 'USD',
        'multiplier': 10,
        'tick_size': 0.10,
        'tick_value': 1.00,
        'trading_hours': 'Nearly 24/5',
    }
}


def create_contract(symbol: str, last_trade_date: str, include_expired: bool = False) -> Future:
    """
    Create a futures contract for the specified symbol and expiration.

    Args:
        symbol: Contract symbol (MNQ, MES, or MGC)
        last_trade_date: Contract month in YYYYMM format (e.g., '202503')
        include_expired: Whether to include expired contracts (for historical data)

    Returns:
        Future contract object

    Example:
        >>> contract = create_contract('MNQ', '202503')
        >>> contract = create_contract('MNQ', '202409', include_expired=True)
    """
    if symbol not in CONTRACT_SPECS:
        raise ValueError(f"Unknown symbol: {symbol}. Must be one of {list(CONTRACT_SPECS.keys())}")

    spec = CONTRACT_SPECS[symbol]

    return Future(
        symbol=symbol,
        exchange=spec['exchange'],
        currency=spec['currency'],
        lastTradeDateOrContractMonth=last_trade_date,
        includeExpired=include_expired
    )


def create_continuous_contract(symbol: str) -> ContFuture:
    """
    Create a continuous futures contract for historical data spanning multiple expirations.

    Note: ContFuture can only be used for historical data, NOT for real-time streaming or orders.

    Args:
        symbol: Contract symbol (MNQ, MES, or MGC)

    Returns:
        Continuous futures contract

    Example:
        >>> contract = create_continuous_contract('MNQ')
    """
    if symbol not in CONTRACT_SPECS:
        raise ValueError(f"Unknown symbol: {symbol}. Must be one of {list(CONTRACT_SPECS.keys())}")

    spec = CONTRACT_SPECS[symbol]

    return ContFuture(
        symbol=symbol,
        exchange=spec['exchange'],
        currency=spec['currency']
    )


def get_current_contract(symbol: str) -> Future:
    """
    Get the current front month contract for the specified symbol.

    This uses hard-coded dates that should be updated quarterly.
    For production, use get_contracts_for_rolling() with IB connection.

    Args:
        symbol: Contract symbol (MNQ, MES, or MGC)

    Returns:
        Front month futures contract
    """
    # These dates should be updated quarterly
    # Current front months as of February 2026:
    CURRENT_CONTRACTS = {
        'MNQ': '202603',  # March 2026 (H26)
        'MES': '202603',  # March 2026 (H26)
        'MGC': '202604',  # April 2026 (J26 - rolls monthly)
    }

    if symbol not in CURRENT_CONTRACTS:
        raise ValueError(f"Unknown symbol: {symbol}")

    return create_contract(symbol, CURRENT_CONTRACTS[symbol])


async def get_contracts_for_rolling(ib: IB, symbol: str) -> Tuple[Optional[Future], Optional[Future]]:
    """
    Get front month and next contract for roll management.

    This queries IB Gateway to get all active contracts and returns
    the two nearest to expiration.

    Args:
        ib: Connected IB instance
        symbol: Contract symbol (MNQ, MES, or MGC)

    Returns:
        Tuple of (front_month_contract, back_month_contract)

    Example:
        >>> front, back = await get_contracts_for_rolling(ib, 'MNQ')
        >>> print(f"Front: {front.lastTradeDateOrContractMonth}")
        >>> print(f"Back: {back.lastTradeDateOrContractMonth}")
    """
    if symbol not in CONTRACT_SPECS:
        raise ValueError(f"Unknown symbol: {symbol}")

    spec = CONTRACT_SPECS[symbol]

    # Create a generic contract to query available contracts
    generic = Future(symbol=symbol, exchange=spec['exchange'], currency=spec['currency'])

    try:
        details = await ib.reqContractDetailsAsync(generic)

        # Filter for active contracts (not yet expired)
        today = datetime.now().strftime('%Y%m%d')
        active = [
            d for d in details
            if d.contract.lastTradeDateOrContractMonth >= today
        ]

        # Sort by expiration date
        active.sort(key=lambda x: x.contract.lastTradeDateOrContractMonth)

        front_month = active[0].contract if active else None
        back_month = active[1].contract if len(active) > 1 else None

        if front_month:
            logger.info(f"{symbol} front month: {front_month.lastTradeDateOrContractMonth}")
        if back_month:
            logger.info(f"{symbol} back month: {back_month.lastTradeDateOrContractMonth}")

        return front_month, back_month

    except Exception as e:
        logger.error(f"Error getting contracts for {symbol}: {e}")
        return None, None


def should_roll(contract: Future, days_before: int = 8) -> bool:
    """
    Check if contract should be rolled based on calendar.

    Industry standard: Roll 8-10 days before expiration.

    Args:
        contract: Futures contract to check
        days_before: Days before expiration to trigger roll (default: 8)

    Returns:
        True if roll should occur, False otherwise

    Example:
        >>> contract = create_contract('MNQ', '202503')
        >>> if should_roll(contract):
        ...     print("Time to roll!")
    """
    try:
        # Parse expiration date (YYYYMM format)
        expiry_str = contract.lastTradeDateOrContractMonth

        if len(expiry_str) == 6:
            # YYYYMM format - find third Friday of month
            year = int(expiry_str[:4])
            month = int(expiry_str[4:6])

            # Find third Friday
            first_day = datetime(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(days=14)

            expiry_date = third_friday
        else:
            # YYYYMMDD format
            expiry_date = datetime.strptime(expiry_str, '%Y%m%d')

        # Calculate roll date
        roll_date = expiry_date - timedelta(days=days_before)

        # Check if we're past the roll date
        result = datetime.now() >= roll_date

        if result:
            logger.warning(
                f"Contract {contract.symbol} {contract.lastTradeDateOrContractMonth} "
                f"should be rolled (expiry: {expiry_date.strftime('%Y-%m-%d')}, "
                f"roll date: {roll_date.strftime('%Y-%m-%d')})"
            )

        return result

    except Exception as e:
        logger.error(f"Error checking roll date for {contract.symbol}: {e}")
        return False


def get_contract_info(symbol: str) -> Dict:
    """
    Get detailed contract specifications.

    Args:
        symbol: Contract symbol (MNQ, MES, or MGC)

    Returns:
        Dictionary with contract specifications
    """
    if symbol not in CONTRACT_SPECS:
        raise ValueError(f"Unknown symbol: {symbol}. Must be one of {list(CONTRACT_SPECS.keys())}")

    return CONTRACT_SPECS[symbol].copy()


# Roll schedule for 2025-2026 (quarterly contracts: H=Mar, M=Jun, U=Sep, Z=Dec)
ROLL_SCHEDULE = {
    'MNQ': [
        {'contract': 'H25', 'expiry': '2025-03-21', 'roll_start': '2025-03-11'},
        {'contract': 'M25', 'expiry': '2025-06-20', 'roll_start': '2025-06-10'},
        {'contract': 'U25', 'expiry': '2025-09-19', 'roll_start': '2025-09-09'},
        {'contract': 'Z25', 'expiry': '2025-12-19', 'roll_start': '2025-12-09'},
        {'contract': 'H26', 'expiry': '2026-03-20', 'roll_start': '2026-03-10'},
    ],
    'MES': [
        {'contract': 'H25', 'expiry': '2025-03-21', 'roll_start': '2025-03-11'},
        {'contract': 'M25', 'expiry': '2025-06-20', 'roll_start': '2025-06-10'},
        {'contract': 'U25', 'expiry': '2025-09-19', 'roll_start': '2025-09-09'},
        {'contract': 'Z25', 'expiry': '2025-12-19', 'roll_start': '2025-12-09'},
        {'contract': 'H26', 'expiry': '2026-03-20', 'roll_start': '2026-03-10'},
    ],
    'MGC': [
        # Gold rolls monthly
        {'contract': 'G25', 'expiry': '2025-02-26', 'roll_start': '2025-02-16'},
        {'contract': 'J25', 'expiry': '2025-04-28', 'roll_start': '2025-04-18'},
        {'contract': 'M25', 'expiry': '2025-06-26', 'roll_start': '2025-06-16'},
        {'contract': 'Q25', 'expiry': '2025-08-27', 'roll_start': '2025-08-17'},
    ]
}


if __name__ == '__main__':
    # Example usage
    print("Contract Specifications:")
    print("-" * 60)

    for symbol in ['MNQ', 'MES', 'MGC']:
        info = get_contract_info(symbol)
        print(f"\n{symbol} - {info['name']}")
        print(f"  Exchange: {info['exchange']}")
        print(f"  Multiplier: ${info['multiplier']}/point")
        print(f"  Tick Size: {info['tick_size']}")
        print(f"  Tick Value: ${info['tick_value']}")

        # Create current contract
        contract = get_current_contract(symbol)
        print(f"  Current Contract: {contract.lastTradeDateOrContractMonth}")

        # Check if should roll
        if should_roll(contract):
            print(f"  ⚠️  Time to roll to next contract!")
