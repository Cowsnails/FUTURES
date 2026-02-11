"""
TastyTrade API Connection Service

Handles OAuth2 authentication, session management, and token refresh
for TastyTrade's DXLink streaming API.

Analogous to ib_service.py but for TastyTrade's REST + WebSocket APIs.
"""

import asyncio
import logging
import time
from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class TastyTradeConnectionState(Enum):
    """Connection state machine states for TastyTrade"""
    DISCONNECTED = 0
    AUTHENTICATING = 1
    AUTHENTICATED = 2      # OAuth2 session active
    STREAMING = 3           # DXLink WebSocket connected
    ERROR = 4               # Auth or connection error


# Maps TastyTrade symbols to DXLink streamer symbols
# Month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun,
#              N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
MONTH_CODES = {
    1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
    7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'
}

# Map our internal symbols to TastyTrade product codes
SYMBOL_TO_PRODUCT_CODE = {
    'MNQ': 'MNQ',
    'MES': 'MES',
    'MGC': 'MGC',
    'ES': 'ES',
    'NQ': 'NQ',
    'GC': 'GC',
}


class TastyTradeService:
    """
    Manages TastyTrade API connection with OAuth2 authentication.

    Features:
    - OAuth2 authentication with refresh tokens (never expire)
    - Automatic access token refresh (~15 min expiry)
    - Streamer symbol resolution via REST API
    - Connection health monitoring
    - DXLink WebSocket token management
    """

    def __init__(
        self,
        client_secret: str = '',
        refresh_token: str = '',
        is_sandbox: bool = False,
        enabled: bool = False,
    ):
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self.is_sandbox = is_sandbox
        self.enabled = enabled

        self.state = TastyTradeConnectionState.DISCONNECTED
        self.session = None  # tastytrade.Session object

        # Cached streamer symbols: {'MNQ': '/MNQH26:XCME', ...}
        self._streamer_symbols: Dict[str, str] = {}
        # Cached contract metadata
        self._contract_info: Dict[str, Dict[str, Any]] = {}

        # Connection metrics
        self.last_auth_time: Optional[datetime] = None
        self.auth_attempts = 0
        self.last_error: Optional[str] = None

    async def connect(self) -> bool:
        """
        Authenticate to TastyTrade via OAuth2.

        Returns True if authentication succeeded.
        """
        if not self.enabled:
            logger.info("TastyTrade integration disabled (no credentials configured)")
            return False

        if not self.client_secret or not self.refresh_token:
            logger.warning(
                "TastyTrade credentials not configured. "
                "Set tastytrade.client_secret and tastytrade.refresh_token in config.yaml"
            )
            self.state = TastyTradeConnectionState.ERROR
            return False

        self.state = TastyTradeConnectionState.AUTHENTICATING
        self.auth_attempts += 1

        try:
            from tastytrade import Session

            logger.info(
                f"Authenticating to TastyTrade "
                f"({'sandbox' if self.is_sandbox else 'production'})..."
            )

            # OAuth2 authentication with refresh token
            # Session constructor handles the token exchange internally
            self.session = Session(
                self.client_secret,
                self.refresh_token,
                is_test=self.is_sandbox
            )

            self.state = TastyTradeConnectionState.AUTHENTICATED
            self.last_auth_time = datetime.now()
            self.last_error = None

            logger.info("TastyTrade OAuth2 authentication successful")
            return True

        except ImportError:
            self.last_error = "tastytrade package not installed (pip install tastytrade)"
            logger.error(self.last_error)
            self.state = TastyTradeConnectionState.ERROR
            return False

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"TastyTrade authentication failed: {e}")
            self.state = TastyTradeConnectionState.ERROR
            return False

    async def refresh_session(self) -> bool:
        """
        Refresh the access token if expired.

        Access tokens last ~15 minutes. The refresh token never expires.
        """
        if not self.session:
            return await self.connect()

        try:
            from tastytrade.utils import now_in_new_york
            if now_in_new_york() > self.session.session_expiration:
                logger.info("TastyTrade access token expired, refreshing...")
                self.session.refresh()
                self.last_auth_time = datetime.now()
                logger.info("TastyTrade access token refreshed")
            return True
        except Exception as e:
            logger.error(f"TastyTrade token refresh failed: {e}")
            # Try full re-authentication
            return await self.connect()

    async def resolve_streamer_symbols(
        self,
        symbols: List[str]
    ) -> Dict[str, str]:
        """
        Resolve internal symbols to DXLink streamer symbols via the REST API.

        Never hardcodes streamer symbols - always queries the API for the
        correct format (e.g., /MNQH26:XCME).

        Args:
            symbols: List of internal symbols (e.g., ['MNQ', 'MES', 'MGC'])

        Returns:
            Dict mapping internal symbol to DXLink streamer symbol
        """
        if not self.session:
            logger.error("Cannot resolve symbols - not authenticated")
            return {}

        await self.refresh_session()

        try:
            from tastytrade.instruments import Future

            resolved = {}

            for symbol in symbols:
                product_code = SYMBOL_TO_PRODUCT_CODE.get(symbol, symbol)

                try:
                    futures = Future.get_futures(
                        self.session,
                        product_codes=[product_code]
                    )

                    if not futures:
                        logger.warning(f"No futures found for product code: {product_code}")
                        continue

                    # Find the active front-month contract
                    active = [f for f in futures if f.active_month]
                    if active:
                        contract = active[0]
                    else:
                        # Fall back to first available
                        contract = futures[0]

                    streamer_sym = contract.streamer_symbol
                    resolved[symbol] = streamer_sym

                    # Cache contract info
                    self._contract_info[symbol] = {
                        'tt_symbol': contract.symbol,
                        'streamer_symbol': streamer_sym,
                        'description': getattr(contract, 'description', ''),
                        'exchange': getattr(contract, 'exchange', 'CME'),
                        'active_month': getattr(contract, 'active_month', False),
                    }

                    logger.info(
                        f"Resolved {symbol} -> {streamer_sym} "
                        f"(TT: {contract.symbol})"
                    )

                except Exception as e:
                    logger.error(f"Failed to resolve symbol {symbol}: {e}")

            self._streamer_symbols = resolved
            return resolved

        except ImportError:
            logger.error("tastytrade package not installed")
            return {}

    def get_streamer_symbol(self, symbol: str) -> Optional[str]:
        """Get cached DXLink streamer symbol for an internal symbol."""
        return self._streamer_symbols.get(symbol)

    def is_authenticated(self) -> bool:
        """Check if TastyTrade session is active."""
        return (
            self.session is not None
            and self.state in (
                TastyTradeConnectionState.AUTHENTICATED,
                TastyTradeConnectionState.STREAMING,
            )
        )

    def disconnect(self):
        """Clean up TastyTrade session."""
        self.session = None
        self.state = TastyTradeConnectionState.DISCONNECTED
        self._streamer_symbols.clear()
        self._contract_info.clear()
        logger.info("TastyTrade session closed")

    def get_health_status(self) -> Dict[str, Any]:
        """Get connection health status."""
        health = {
            'enabled': self.enabled,
            'state': self.state.name,
            'authenticated': self.is_authenticated(),
            'auth_attempts': self.auth_attempts,
            'streamer_symbols': dict(self._streamer_symbols),
        }

        if self.last_auth_time:
            uptime = (datetime.now() - self.last_auth_time).total_seconds()
            health['auth_uptime_seconds'] = uptime

        if self.last_error:
            health['last_error'] = self.last_error

        return health
