"""
Security middleware and utilities for the application

Provides:
- Rate limiting
- Security headers
- Input validation
- IP filtering
"""

import time
import hashlib
from collections import defaultdict
from typing import Dict, Tuple, Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests: Dict[str, list] = defaultdict(list)
        self.hour_requests: Dict[str, list] = defaultdict(list)

    def _clean_old_requests(self, requests: list, max_age: float):
        """Remove requests older than max_age seconds"""
        current_time = time.time()
        return [req_time for req_time in requests if current_time - req_time < max_age]

    def check_rate_limit(self, client_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check if client has exceeded rate limits.

        Returns:
            (allowed, error_message) tuple
        """
        current_time = time.time()

        # Clean old requests
        self.minute_requests[client_id] = self._clean_old_requests(
            self.minute_requests[client_id], 60
        )
        self.hour_requests[client_id] = self._clean_old_requests(
            self.hour_requests[client_id], 3600
        )

        # Check limits
        minute_count = len(self.minute_requests[client_id])
        hour_count = len(self.hour_requests[client_id])

        if minute_count >= self.requests_per_minute:
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"

        if hour_count >= self.requests_per_hour:
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"

        # Record this request
        self.minute_requests[client_id].append(current_time)
        self.hour_requests[client_id].append(current_time)

        return True, None

    def get_stats(self, client_id: str) -> Dict:
        """Get rate limit statistics for a client"""
        self.minute_requests[client_id] = self._clean_old_requests(
            self.minute_requests[client_id], 60
        )
        self.hour_requests[client_id] = self._clean_old_requests(
            self.hour_requests[client_id], 3600
        )

        return {
            "requests_this_minute": len(self.minute_requests[client_id]),
            "requests_this_hour": len(self.hour_requests[client_id]),
            "limit_per_minute": self.requests_per_minute,
            "limit_per_hour": self.requests_per_hour,
            "remaining_minute": max(0, self.requests_per_minute - len(self.minute_requests[client_id])),
            "remaining_hour": max(0, self.requests_per_hour - len(self.hour_requests[client_id])),
        }


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add HSTS header (only if using HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://unpkg.com; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' ws: wss:; "
            "img-src 'self' data:; "
        )

        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""

    def __init__(self, app, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter

    def _get_client_id(self, request: Request) -> str:
        """Get unique client identifier"""
        # Try to get real IP from headers (if behind proxy)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()

        if not client_ip:
            client_ip = request.headers.get("X-Real-IP", "")

        if not client_ip and request.client:
            client_ip = request.client.host

        return client_ip or "unknown"

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)

        # Get client identifier
        client_id = self._get_client_id(request)

        # Check rate limit
        allowed, error_msg = self.rate_limiter.check_rate_limit(client_id)

        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_id}: {error_msg}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=error_msg
            )

        # Add rate limit info to response headers
        response = await call_next(request)

        stats = self.rate_limiter.get_stats(client_id)
        response.headers["X-RateLimit-Limit-Minute"] = str(stats["limit_per_minute"])
        response.headers["X-RateLimit-Remaining-Minute"] = str(stats["remaining_minute"])
        response.headers["X-RateLimit-Limit-Hour"] = str(stats["limit_per_hour"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(stats["remaining_hour"])

        return response


def validate_symbol(symbol: str) -> bool:
    """
    Validate symbol input.

    Returns:
        True if valid, raises HTTPException otherwise
    """
    valid_symbols = ["MNQ", "MES", "MGC"]

    if symbol not in valid_symbols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid symbol: {symbol}. Must be one of {valid_symbols}"
        )

    return True


def validate_bar_size(bar_size: str) -> bool:
    """
    Validate bar size input.

    Returns:
        True if valid, raises HTTPException otherwise
    """
    valid_bar_sizes = ["1 min", "5 mins", "15 mins", "30 mins", "1 hour", "1 day"]

    if bar_size not in valid_bar_sizes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid bar size: {bar_size}. Must be one of {valid_bar_sizes}"
        )

    return True


def validate_indicator_params(indicator_type: str, params: dict) -> bool:
    """
    Validate indicator parameters.

    Returns:
        True if valid, raises HTTPException otherwise
    """
    valid_indicators = ["sma", "ema", "rsi", "macd", "bb", "stochastic", "cci", "roc", "wma", "vwap"]

    if indicator_type not in valid_indicators:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid indicator type: {indicator_type}"
        )

    # Validate period parameter if present
    if "period" in params:
        period = params["period"]
        if not isinstance(period, int) or period < 1 or period > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid period: {period}. Must be between 1 and 500"
            )

    # Validate num_std parameter for Bollinger Bands
    if indicator_type == "bb" and "num_std" in params:
        num_std = params["num_std"]
        if not isinstance(num_std, (int, float)) or num_std < 0.1 or num_std > 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid num_std: {num_std}. Must be between 0.1 and 10"
            )

    return True


def generate_api_key_hash(api_key: str) -> str:
    """Generate hash of API key for storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def verify_api_key_hash(api_key: str, stored_hash: str) -> bool:
    """Verify API key against stored hash"""
    return generate_api_key_hash(api_key) == stored_hash


class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """IP whitelisting middleware (optional, for high-security deployments)"""

    def __init__(self, app, allowed_ips: list = None):
        super().__init__(app)
        self.allowed_ips = allowed_ips or []

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Try to get real IP from headers (if behind proxy)
        client_ip = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()

        if not client_ip:
            client_ip = request.headers.get("X-Real-IP", "")

        if not client_ip and request.client:
            client_ip = request.client.host

        return client_ip or "unknown"

    async def dispatch(self, request: Request, call_next):
        # Skip if no whitelist configured
        if not self.allowed_ips:
            return await call_next(request)

        # Skip for health check
        if request.url.path == "/health":
            return await call_next(request)

        client_ip = self._get_client_ip(request)

        # Check if IP is whitelisted
        if client_ip not in self.allowed_ips:
            logger.warning(f"Blocked request from non-whitelisted IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )

        return await call_next(request)
