#!/usr/bin/env python
"""
IB Gateway Connection Test Script

This script tests the connection to IB Gateway and validates contract access.
Run this after setting up IB Gateway to verify everything is working.

Usage:
    python test_connection.py

Requirements:
    - IB Gateway must be running
    - API must be enabled in Gateway settings
    - Market data subscriptions must be active
    - API Acknowledgement form must be completed
"""

import asyncio
from ib_insync import IB, util
from backend.contracts import create_contract, get_current_contract, get_contract_info
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connection settings
HOST = os.getenv('IB_HOST', '127.0.0.1')
PORT = int(os.getenv('IB_PORT', '4002'))
CLIENT_ID = int(os.getenv('IB_CLIENT_ID', '1'))


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_success(text):
    """Print success message"""
    print(f"✓ {text}")


def print_error(text):
    """Print error message"""
    print(f"✗ {text}")


def print_info(text):
    """Print info message"""
    print(f"  {text}")


async def test_connection():
    """Test basic IB Gateway connection"""
    print_header("Testing IB Gateway Connection")

    ib = IB()

    try:
        print_info(f"Connecting to IB Gateway at {HOST}:{PORT} with client ID {CLIENT_ID}...")

        await ib.connectAsync(host=HOST, port=PORT, clientId=CLIENT_ID, timeout=10)

        if ib.isConnected():
            print_success(f"Connected successfully!")
            print_info(f"Server version: {ib.serverVersion()}")
            print_info(f"Connection time: {ib.timezoneTWS()}")
            return ib
        else:
            print_error("Connection failed")
            return None

    except Exception as e:
        print_error(f"Connection error: {e}")
        print_info("\nTroubleshooting tips:")
        print_info("1. Make sure IB Gateway is running")
        print_info("2. Check that API is enabled: Configure → Settings → API → Settings")
        print_info("3. Verify port number (4002 for paper, 4001 for live)")
        print_info("4. Ensure 'Enable ActiveX and Socket Clients' is checked")
        return None


async def test_contract_qualification(ib):
    """Test contract qualification"""
    print_header("Testing Contract Qualification")

    symbols = ['MNQ', 'MES', 'MGC']
    qualified_contracts = []

    for symbol in symbols:
        try:
            print_info(f"\nQualifying {symbol}...")

            # Get current contract
            contract = get_current_contract(symbol)
            info = get_contract_info(symbol)

            print_info(f"  {info['name']}")
            print_info(f"  Exchange: {info['exchange']}")
            print_info(f"  Contract: {contract.lastTradeDateOrContractMonth}")

            # Qualify with IB
            qualified = await ib.qualifyContractsAsync(contract)

            if qualified:
                qualified_contract = qualified[0]
                print_success(f"Qualified successfully!")
                print_info(f"  Contract ID: {qualified_contract.conId}")
                print_info(f"  Local Symbol: {qualified_contract.localSymbol}")
                qualified_contracts.append(qualified_contract)
            else:
                print_error(f"Failed to qualify {symbol}")

        except Exception as e:
            print_error(f"Error qualifying {symbol}: {e}")

    return qualified_contracts


async def test_market_data_subscription(ib, contracts):
    """Test market data subscriptions"""
    print_header("Testing Market Data Subscriptions")

    if not contracts:
        print_error("No qualified contracts to test")
        return

    # Test with first contract (MNQ)
    contract = contracts[0]

    try:
        print_info(f"\nRequesting market data for {contract.symbol}...")

        # Request market data (this will fail if not subscribed)
        ticker = ib.reqMktData(contract, '', False, False)

        # Wait a moment for data
        await asyncio.sleep(2)

        if ticker.last and ticker.last > 0:
            print_success("Market data received!")
            print_info(f"  Last price: {ticker.last}")
            print_info(f"  Bid: {ticker.bid}")
            print_info(f"  Ask: {ticker.ask}")
            print_info(f"  Volume: {ticker.volume}")

            # Cancel market data
            ib.cancelMktData(contract)
            return True

        elif ticker.last == -1 or not ticker.last:
            print_error("No market data received")
            print_info("\nPossible issues:")
            print_info("1. Market data subscription not active")
            print_info("2. API Acknowledgement form not completed")
            print_info("3. Paper account data sharing not enabled (requires 24h wait)")
            print_info("4. Markets are closed (try delayed data)")
            return False

    except Exception as e:
        print_error(f"Market data error: {e}")

        if "354" in str(e):
            print_info("\nError 354: Not subscribed to market data")
            print_info("Action required:")
            print_info("1. Subscribe to CME Real-Time data in Client Portal")
            print_info("2. Complete API Acknowledgement form")
            print_info("3. Enable paper account data sharing (if using paper)")
        return False


async def test_historical_data(ib, contracts):
    """Test historical data request"""
    print_header("Testing Historical Data Request")

    if not contracts:
        print_error("No qualified contracts to test")
        return

    contract = contracts[0]

    try:
        print_info(f"\nRequesting 1 day of historical data for {contract.symbol}...")

        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr='1 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=False,
            formatDate=1
        )

        if bars and len(bars) > 0:
            print_success(f"Historical data received!")
            print_info(f"  Number of bars: {len(bars)}")
            print_info(f"  First bar: {bars[0].date}")
            print_info(f"  Last bar: {bars[-1].date}")
            print_info(f"  Latest close: {bars[-1].close}")
            return True
        else:
            print_error("No historical data received")
            return False

    except Exception as e:
        print_error(f"Historical data error: {e}")

        if "162" in str(e):
            print_info("\nError 162: Pacing violation")
            print_info("Wait 15 seconds before making another request")
        elif "354" in str(e):
            print_info("\nError 354: Not subscribed")
            print_info("Historical data requires active market data subscription")

        return False


async def main():
    """Run all connection tests"""
    print("\n" + "█" * 70)
    print("  IB Gateway Connection Test Suite")
    print("█" * 70)

    # Test 1: Connection
    ib = await test_connection()
    if not ib:
        print("\n" + "=" * 70)
        print("❌ CONNECTION TEST FAILED")
        print("=" * 70)
        sys.exit(1)

    # Test 2: Contract Qualification
    contracts = await test_contract_qualification(ib)
    if not contracts:
        print("\n" + "=" * 70)
        print("⚠️  CONTRACT QUALIFICATION FAILED")
        print("=" * 70)
        ib.disconnect()
        sys.exit(1)

    # Test 3: Market Data
    market_data_ok = await test_market_data_subscription(ib, contracts)

    # Test 4: Historical Data
    historical_ok = await test_historical_data(ib, contracts)

    # Disconnect
    ib.disconnect()

    # Summary
    print_header("Test Summary")
    print_success("Connection: PASSED")
    print_success("Contract Qualification: PASSED")

    if market_data_ok:
        print_success("Market Data: PASSED")
    else:
        print_error("Market Data: FAILED (see above for details)")

    if historical_ok:
        print_success("Historical Data: PASSED")
    else:
        print_error("Historical Data: FAILED (see above for details)")

    print("\n" + "=" * 70)

    if market_data_ok and historical_ok:
        print("✅ ALL TESTS PASSED")
        print("=" * 70)
        print("\nYou're ready to start the application!")
        sys.exit(0)
    else:
        print("⚠️  SOME TESTS FAILED")
        print("=" * 70)
        print("\nReview the errors above and fix issues before proceeding.")
        sys.exit(1)


if __name__ == '__main__':
    # Use ib_insync event loop
    util.startLoop()
    asyncio.run(main())
