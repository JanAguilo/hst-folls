"""
Polymarket Binary Options Greeks Calculator

This module provides tools for calculating option Greeks on Polymarket prediction markets.
Binary/digital options have specific Greek calculations that differ from vanilla options,
and near-expiry normalization is critical to prevent blow-up.

Two modeling approaches are provided:
1. Black-Scholes Digital Option Model: Standard approach for digital options
2. Brownian Bridge Model: Better suited for prediction markets that must resolve to 0 or 1

Key Insight: The Brownian Bridge model naturally handles the "pinning" behavior at expiry,
where volatility must go to zero as the market resolves to a binary outcome.

Author: Pol (Polymarket Hackathon 2025)
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timezone
import json


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PolymarketMarket:
    """
    Represents a Polymarket market with fields from the Gamma API.
    
    API Response fields used:
    - id: Market ID
    - question: Market question text  
    - conditionId: Condition ID for the market
    - slug: URL slug
    - endDate: Resolution date (ISO format)
    - outcomePrices: JSON string like '["0.65", "0.35"]' for [Yes, No]
    - clobTokenIds: JSON string with token IDs for trading
    - outcomes: JSON string like '["Yes", "No"]'
    """
    id: str
    question: str
    condition_id: str
    slug: str
    end_date: datetime
    outcome_prices: List[float]  # [yes_price, no_price]
    clob_token_ids: List[str]
    outcomes: List[str]
    volume: float = 0.0
    liquidity: float = 0.0
    
    @classmethod
    def from_api_response(cls, data: dict) -> 'PolymarketMarket':
        """Parse a market from Polymarket Gamma API response."""
        outcome_prices = json.loads(data.get('outcomePrices', '["0.5", "0.5"]'))
        outcome_prices = [float(p) for p in outcome_prices]
        
        clob_token_ids = json.loads(data.get('clobTokenIds', '[]'))
        if isinstance(clob_token_ids, str):
            clob_token_ids = [clob_token_ids]
            
        outcomes = json.loads(data.get('outcomes', '["Yes", "No"]'))
        
        end_date_str = data.get('endDate') or data.get('endDateIso')
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        else:
            end_date = datetime.now(timezone.utc)
        
        return cls(
            id=data.get('id', ''),
            question=data.get('question', ''),
            condition_id=data.get('conditionId', ''),
            slug=data.get('slug', ''),
            end_date=end_date,
            outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids,
            outcomes=outcomes,
            volume=float(data.get('volumeNum', 0) or 0),
            liquidity=float(data.get('liquidityNum', 0) or 0),
        )
    
    @property
    def yes_price(self) -> float:
        return self.outcome_prices[0] if self.outcome_prices else 0.5
    
    @property
    def no_price(self) -> float:
        return self.outcome_prices[1] if len(self.outcome_prices) > 1 else 1 - self.yes_price


@dataclass 
class Greeks:
    """
    Option Greeks for a binary option position.
    
    Interpretations for prediction markets:
    - delta: $ P&L per 1 percentage point move in probability
    - gamma: Change in delta per 1 pp move (risk curvature)
    - vega: $ P&L per 1% increase in volatility
    - theta: Daily time decay in $
    - rho: Interest rate sensitivity (usually ~0 for prediction markets)
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float = 0.0
    
    def __add__(self, other: 'Greeks') -> 'Greeks':
        return Greeks(
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
        )
    
    def __mul__(self, scalar: float) -> 'Greeks':
        return Greeks(
            delta=self.delta * scalar,
            gamma=self.gamma * scalar,
            vega=self.vega * scalar,
            theta=self.theta * scalar,
            rho=self.rho * scalar,
        )
    
    def __rmul__(self, scalar: float) -> 'Greeks':
        return self.__mul__(scalar)
    
    def to_dict(self) -> dict:
        return {
            'delta': round(self.delta, 6),
            'gamma': round(self.gamma, 6),
            'vega': round(self.vega, 6),
            'theta': round(self.theta, 8),
            'rho': round(self.rho, 6),
        }


# =============================================================================
# MODEL 1: BLACK-SCHOLES DIGITAL OPTION
# =============================================================================

class BlackScholesDigital:
    """
    Black-Scholes model for cash-or-nothing digital (binary) options.
    
    A digital call pays $1 if the event occurs, $0 otherwise.
    The market price p represents the risk-neutral probability of payout.
    
    We model probability as following a bounded diffusion:
        dp = σ * p * (1-p) * dW
    
    This is the "Jacobi process" which ensures p ∈ (0,1).
    """
    
    def __init__(self, risk_free_rate: float = 0.0):
        self.r = risk_free_rate
    
    def price(self, p: float, sigma: float, T: float, is_yes: bool = True) -> float:
        """Fair value is simply the probability for a digital option."""
        return p if is_yes else (1 - p)
    
    def greeks(self, p: float, sigma: float, T: float, is_yes: bool = True) -> Greeks:
        """
        Calculate Greeks for a digital option on a prediction market.
        
        Args:
            p: Current YES probability (0 < p < 1)
            sigma: Volatility of the probability process (annualized)
            T: Time to expiry in years
            is_yes: True for YES position, False for NO
        
        Returns:
            Greeks object
        """
        if T <= 1e-10:
            return Greeks(delta=0, gamma=0, vega=0, theta=0, rho=0)
        
        p = np.clip(p, 0.01, 0.99)
        sqrt_T = np.sqrt(T)
        
        # Probability volatility (Jacobi process)
        p_vol = sigma * p * (1 - p)
        
        # ===== DELTA =====
        # For YES position worth $1 at payout: Value = p, so dV/dp = 1
        # Per $1 notional, a 1pp move in probability = $0.01 change
        delta = 1.0
        
        # ===== GAMMA =====
        # How does the "riskiness" of the position change with p?
        # d(p_vol)/dp = σ(1 - 2p)
        # Positive when p < 0.5 (underdog), negative when p > 0.5 (favorite)
        gamma = sigma * (1 - 2*p)
        
        # ===== VEGA =====
        # dV/dσ: How does value change with volatility?
        # For p > 0.5: higher vol = more chance outcome flips (bad for YES holder)
        # For p < 0.5: higher vol = more chance outcome flips (good for YES holder)
        vega = -(p - 0.5) * sqrt_T * p * (1 - p)
        
        # ===== THETA =====
        # Time decay: as T→0, uncertainty resolves
        # If you're winning (p > 0.5), time passing is good
        # If you're losing (p < 0.5), time passing is bad
        theta_direction = p - 0.5
        theta_magnitude = p_vol**2 / (2 * T) if T > 0 else 0
        theta = theta_direction * theta_magnitude / 365  # Daily
        
        sign = 1 if is_yes else -1
        
        return Greeks(
            delta=sign * delta,
            gamma=sign * gamma,
            vega=sign * vega,
            theta=sign * theta,
            rho=0,
        )


# =============================================================================
# MODEL 2: BROWNIAN BRIDGE
# =============================================================================

class BrownianBridge:
    """
    Brownian Bridge model for prediction market probabilities.
    
    Key Insight: A prediction market probability must resolve to exactly 0 or 1.
    This is fundamentally different from equity prices.
    
    The Brownian Bridge captures this "pinning" behavior:
    - Volatility is highest mid-life
    - Volatility goes to zero as we approach expiry
    - σ_eff(τ) = σ * √(τ/T) where τ is time remaining, T is total duration
    
    This naturally regularizes Greeks near expiry without ad-hoc capping.
    """
    
    def __init__(self):
        pass
    
    def effective_volatility(self, tau: float, T: float, sigma: float) -> float:
        """
        Calculate the effective volatility under bridge dynamics.
        
        σ_eff = σ * √(τ/T)
        
        This goes to 0 as τ→0, reflecting certainty at expiry.
        """
        if T <= 0 or tau <= 0:
            return 0.0
        return sigma * np.sqrt(tau / T)
    
    def greeks(self, p: float, sigma: float, tau: float, T: float, is_yes: bool = True) -> Greeks:
        """
        Calculate Greeks using Brownian Bridge dynamics.
        
        Args:
            p: Current YES probability (0 < p < 1)
            sigma: Base volatility parameter
            tau: Time to expiry (remaining time)
            T: Total market duration (for bridge scaling)
            is_yes: True for YES position, False for NO
        
        Returns:
            Greeks with natural near-expiry regularization
        """
        if tau <= 1e-10:
            return Greeks(delta=0, gamma=0, vega=0, theta=0, rho=0)
        
        if T <= 0:
            T = tau
        
        p = np.clip(p, 0.01, 0.99)
        
        # Effective volatility decays as we approach expiry
        sigma_eff = self.effective_volatility(tau, T, sigma)
        
        # Instantaneous probability volatility
        p_vol = sigma_eff * p * (1 - p)
        
        # ===== DELTA =====
        delta = 1.0
        
        # ===== GAMMA =====  
        # Scaled by effective volatility
        gamma = sigma_eff * (1 - 2*p)
        
        # ===== VEGA =====
        # Sensitivity to BASE volatility (chain rule through sigma_eff)
        bridge_factor = np.sqrt(tau / T)
        vega = -(p - 0.5) * bridge_factor * p * (1 - p)
        
        # ===== THETA =====
        # Time decay modified by bridge: σ_eff decreases with τ
        # d(σ_eff)/dτ = σ / (2√(τT)) > 0, so vol drops as time passes
        theta_direction = p - 0.5
        theta_magnitude = p_vol**2 / (2 * tau) if tau > 0 else 0
        # Additional decay from effective vol dropping
        theta_vol_decay = 0.5 * sigma * p * (1-p) / np.sqrt(tau * T) if tau > 0 and T > 0 else 0
        theta = (theta_direction * theta_magnitude - abs(theta_direction) * theta_vol_decay) / 365
        
        sign = 1 if is_yes else -1
        
        return Greeks(
            delta=sign * delta,
            gamma=sign * gamma,
            vega=sign * vega,
            theta=sign * theta,
            rho=0,
        )


# =============================================================================
# UNDERLYING ASSET DATA (e.g., BTC, ETH prices)
# =============================================================================

def fetch_yahoo_prices(ticker: str, period: str = "1mo") -> Optional[List[float]]:
    """
    Fetch historical prices from Yahoo Finance.
    
    Args:
        ticker: Yahoo Finance ticker (e.g., 'BTC-USD', 'ETH-USD')
        period: Time period ('1d', '5d', '1mo', '3mo', '1y')
    
    Returns:
        List of closing prices, or None if fetch fails
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if len(hist) > 0:
            return hist['Close'].tolist()
        return None
    except:
        return None


def calculate_realized_vol(prices: List[float], annualize: bool = True) -> float:
    """Calculate realized volatility from price series."""
    if len(prices) < 2:
        return 0.5
    
    prices = np.array(prices)
    returns = np.diff(np.log(prices))  # Log returns
    
    vol = np.std(returns)
    if annualize:
        vol *= np.sqrt(252)  # Annualize assuming daily data
    
    return vol


def infer_ticker_from_question(question: str) -> Optional[str]:
    """
    Attempt to infer the underlying ticker from the market question.
    
    Returns Yahoo Finance ticker or None.
    """
    question_lower = question.lower()
    
    # Crypto
    if 'bitcoin' in question_lower or 'btc' in question_lower:
        return 'BTC-USD'
    if 'ethereum' in question_lower or 'eth' in question_lower:
        return 'ETH-USD'
    if 'solana' in question_lower or 'sol' in question_lower:
        return 'SOL-USD'
    
    # Commodities
    if 'gold' in question_lower:
        return 'GC=F'
    if 'silver' in question_lower:
        return 'SI=F'
    if 'oil' in question_lower or 'crude' in question_lower:
        return 'CL=F'
    
    # Indices
    if 's&p 500' in question_lower or 'sp500' in question_lower or 's&p' in question_lower:
        return '^GSPC'
    if 'nasdaq' in question_lower:
        return '^IXIC'
    if 'dow' in question_lower:
        return '^DJI'
    
    # Individual stocks (common ones)
    if 'tesla' in question_lower or 'tsla' in question_lower:
        return 'TSLA'
    if 'apple' in question_lower or 'aapl' in question_lower:
        return 'AAPL'
    if 'nvidia' in question_lower or 'nvda' in question_lower:
        return 'NVDA'
    
    return None


# =============================================================================
# VOLATILITY ESTIMATION
# =============================================================================

class VolatilityEstimator:
    """Estimate implied volatility from Polymarket data."""
    
    @staticmethod
    def from_price_history(history: List[dict], annualize: bool = True) -> float:
        """
        Estimate volatility from historical prices.
        
        Args:
            history: List of {"t": timestamp, "p": price} from CLOB API
            annualize: If True, annualize the volatility
        
        Returns:
            Estimated volatility
        """
        if len(history) < 2:
            return 0.50  # Default
        
        history = sorted(history, key=lambda x: x['t'])
        prices = np.array([h['p'] for h in history])
        times = np.array([h['t'] for h in history])
        
        prices = np.clip(prices, 0.01, 0.99)
        
        # Calculate returns in probability space
        returns = np.diff(prices)
        dt = np.diff(times) / (365.25 * 24 * 3600)  # Years
        
        # Realized variance
        if len(returns) > 0 and np.mean(dt) > 0:
            var_per_year = np.var(returns) / np.mean(dt)
            return np.sqrt(var_per_year) if annualize else np.std(returns)
        
        return 0.50
    
    @staticmethod
    def from_spread(bid: float, ask: float, tau: float) -> float:
        """
        Estimate implied volatility from bid-ask spread.
        
        Wider spreads often indicate higher uncertainty.
        """
        spread = ask - bid
        mid = np.clip((bid + ask) / 2, 0.05, 0.95)
        
        # Heuristic: spread ≈ σ * p * (1-p) * √τ for market makers
        denominator = mid * (1 - mid) * np.sqrt(tau) if tau > 0 else 1
        if denominator > 0.01:
            return np.clip(spread / denominator, 0.1, 2.0)
        return 0.50
    
    @staticmethod
    def from_underlying_asset(question: str, period: str = "1mo") -> Optional[float]:
        """
        Estimate volatility from the underlying asset's price history.
        
        Args:
            question: Market question to parse for ticker
            period: Historical period to analyze
        
        Returns:
            Realized volatility of underlying, or None if not found
        """
        ticker = infer_ticker_from_question(question)
        if not ticker:
            return None
        
        prices = fetch_yahoo_prices(ticker, period)
        if not prices:
            return None
        
        return calculate_realized_vol(prices, annualize=True)


# =============================================================================
# PORTFOLIO MANAGEMENT
# =============================================================================

@dataclass
class Position:
    """A position in a single market."""
    market: PolymarketMarket
    quantity: float  # Positive = long, negative = short
    is_yes: bool
    entry_price: float = 0.0
    
    @property
    def notional(self) -> float:
        """Current notional value."""
        price = self.market.yes_price if self.is_yes else self.market.no_price
        return abs(self.quantity) * price
    
    @property
    def pnl(self) -> float:
        """Unrealized P&L."""
        current = self.market.yes_price if self.is_yes else self.market.no_price
        return self.quantity * (current - self.entry_price)


class Portfolio:
    """Portfolio of Polymarket positions with aggregate Greeks."""
    
    # Threshold: use Bridge for markets > 7 days out, BS for near-term
    BRIDGE_THRESHOLD_DAYS = 7
    
    def __init__(self, model: str = 'auto'):
        """
        Args:
            model: 'auto' (recommended), 'bridge', or 'bs'
                   'auto' uses Bridge for longer-dated, BS for near-expiry
        """
        self.positions: List[Position] = []
        self.model = model
        self._bs = BlackScholesDigital()
        self._bridge = BrownianBridge()
    
    def add_position(self, position: Position):
        self.positions.append(position)
    
    def remove_position(self, market_id: str):
        self.positions = [p for p in self.positions if p.market.id != market_id]
    
    def _time_to_expiry(self, market: PolymarketMarket) -> float:
        """Time to expiry in years."""
        now = datetime.now(timezone.utc)
        delta = market.end_date - now
        return max(delta.total_seconds() / (365.25 * 24 * 3600), 1/365)  # Min 1 day
    
    def _time_to_expiry_days(self, market: PolymarketMarket) -> float:
        """Time to expiry in days."""
        return self._time_to_expiry(market) * 365.25
    
    def _market_duration(self, market: PolymarketMarket) -> float:
        """Estimate total market duration (default 30 days)."""
        return 30 / 365.25
    
    def _select_model(self, market: PolymarketMarket) -> str:
        """Auto-select model based on time to expiry."""
        if self.model != 'auto':
            return self.model
        
        days = self._time_to_expiry_days(market)
        
        # Use Bridge for longer-dated (better theoretical foundation)
        # Use BS for near-expiry (simpler, gamma behavior more predictable)
        if days > self.BRIDGE_THRESHOLD_DAYS:
            return 'bridge'
        else:
            return 'bs'
    
    def calculate_greeks(
        self, 
        position: Position, 
        sigma: float = 0.5
    ) -> Greeks:
        """Calculate Greeks for a single position."""
        market = position.market
        tau = self._time_to_expiry(market)
        p = market.yes_price if position.is_yes else market.no_price
        
        # Ensure valid probability
        p = np.clip(p, 0.01, 0.99)
        
        selected_model = self._select_model(market)
        
        if selected_model == 'bridge':
            T = max(self._market_duration(market), tau)
            greeks = self._bridge.greeks(p, sigma, tau, T, position.is_yes)
        else:
            greeks = self._bs.greeks(p, sigma, tau, position.is_yes)
        
        # Scale by position quantity
        greeks = greeks * position.quantity
        
        return greeks
    
    def aggregate_greeks(
        self, 
        volatilities: Optional[Dict[str, float]] = None
    ) -> Greeks:
        """Calculate aggregate portfolio Greeks."""
        volatilities = volatilities or {}
        total = Greeks(delta=0, gamma=0, vega=0, theta=0, rho=0)
        
        for position in self.positions:
            sigma = volatilities.get(position.market.id, 0.5)
            greeks = self.calculate_greeks(position, sigma)
            total = total + greeks
        
        return total
    
    def greeks_breakdown(
        self, 
        volatilities: Optional[Dict[str, float]] = None
    ) -> List[Tuple[Position, Greeks]]:
        """Get Greeks for each position."""
        volatilities = volatilities or {}
        return [
            (pos, self.calculate_greeks(pos, volatilities.get(pos.market.id, 0.5)))
            for pos in self.positions
        ]


# =============================================================================
# TRADE ANALYSIS
# =============================================================================

def analyze_trade_impact(
    portfolio: Portfolio,
    market: PolymarketMarket,
    quantity: float,
    is_yes: bool,
    sigma: float = 0.5
) -> Dict:
    """
    Analyze how a proposed trade would impact portfolio Greeks.
    
    Returns:
        Dict with 'before', 'after', and 'change' Greeks
    """
    before = portfolio.aggregate_greeks()
    
    # Calculate impact of new position
    temp_position = Position(market=market, quantity=quantity, is_yes=is_yes)
    impact = portfolio.calculate_greeks(temp_position, sigma)
    
    after = before + impact
    change = Greeks(
        delta=after.delta - before.delta,
        gamma=after.gamma - before.gamma,
        vega=after.vega - before.vega,
        theta=after.theta - before.theta,
        rho=after.rho - before.rho,
    )
    
    return {
        'before': before.to_dict(),
        'after': after.to_dict(),
        'change': change.to_dict(),
    }


def suggest_hedge(
    portfolio: Portfolio,
    target_delta: float = 0.0,
    available_markets: List[PolymarketMarket] = None,
    sigma: float = 0.5
) -> List[Dict]:
    """
    Suggest positions to achieve target delta.
    
    Simple delta hedging - for more sophisticated optimization,
    extend to multi-Greek targeting with quadratic programming.
    """
    current_delta = portfolio.aggregate_greeks().delta
    delta_needed = target_delta - current_delta
    
    if abs(delta_needed) < 0.01:
        return []  # Already at target
    
    suggestions = []
    for market in (available_markets or []):
        # Each YES share has delta ≈ 1
        quantity_needed = delta_needed  # Approximate
        suggestions.append({
            'market_id': market.id,
            'market_question': market.question,
            'side': 'YES' if delta_needed > 0 else 'NO',
            'quantity': abs(quantity_needed),
            'rationale': f"Add {abs(quantity_needed):.1f} shares to move delta by {delta_needed:.2f}"
        })
    
    return suggestions


# =============================================================================
# LIVE DATA FETCHING
# =============================================================================

def fetch_live_markets(limit: int = 10, search: str = None) -> List[PolymarketMarket]:
    """
    Fetch live markets from Polymarket Gamma API.
    
    Args:
        limit: Number of markets to fetch
        search: Optional search term to filter markets
    
    Returns:
        List of PolymarketMarket objects
    """
    import requests
    
    # Headers to avoid Brotli encoding issues
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',  # Explicitly exclude 'br' (Brotli)
        'User-Agent': 'PolymarketGreeksCalculator/1.0',
    }
    
    # If searching, use the events endpoint
    if search:
        url = "https://gamma-api.polymarket.com/events"
        params = {
            "active": "true",
            "closed": "false",
            "limit": 100,  # Fetch more to filter
            "order": "volume24hr",
            "ascending": "false",
        }
    else:
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "order": "volume24hr",
            "ascending": "false",
        }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        markets = []
        
        # Handle events endpoint (contains nested markets)
        if search:
            for event in data:
                # Check if event title matches search
                title = event.get('title', '') or event.get('question', '') or ''
                description = event.get('description', '') or ''
                
                # Search in title and description
                search_lower = search.lower()
                if search_lower not in title.lower() and search_lower not in description.lower():
                    continue
                    
                # Get markets from event
                event_markets = event.get('markets', [])
                if isinstance(event_markets, list):
                    for m in event_markets:
                        try:
                            market = PolymarketMarket.from_api_response(m)
                            markets.append(market)
                        except:
                            continue
                            
                if len(markets) >= limit:
                    break
        else:
            # Direct markets endpoint
            for m in data:
                try:
                    market = PolymarketMarket.from_api_response(m)
                    markets.append(market)
                except:
                    continue
        
        return markets[:limit]
        
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []


def fetch_price_history(token_id: str, interval: str = "1w") -> List[dict]:
    """
    Fetch historical prices from Polymarket CLOB API.
    
    Args:
        token_id: The CLOB token ID
        interval: Time interval (1h, 6h, 1d, 1w, max)
    
    Returns:
        List of {"t": timestamp, "p": price} dicts
    """
    import requests
    
    # Clean the token ID - remove any quotes or brackets
    token_id = str(token_id).strip('[]"\'')
    
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'PolymarketGreeksCalculator/1.0',
    }
    
    url = "https://clob.polymarket.com/prices-history"
    params = {
        "market": token_id,
        "interval": interval,
        "fidelity": 60,  # 60-minute candles
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("history", [])
    except:
        # Silently fail - price history is optional
        return []


def run_live_demo():
    """Run the Greeks calculator with live Polymarket data."""
    import argparse
    
    # Get args from parent scope
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", "-l", action="store_true")
    parser.add_argument("-n", "--num-markets", type=int, default=10)
    parser.add_argument("-s", "--search", type=str, default=None)
    args, _ = parser.parse_known_args()
    
    print("=" * 70)
    print("POLYMARKET GREEKS CALCULATOR - LIVE DATA")
    print("=" * 70)
    
    search_term = args.search
    num_markets = args.num_markets
    
    if search_term:
        print(f"\nSearching for markets matching: '{search_term}'...")
    else:
        print("\nFetching top markets by volume...")
    
    markets = fetch_live_markets(limit=num_markets, search=search_term)
    
    if not markets:
        print("No markets found. Try a different search term or check your connection.")
        return
    
    print(f"\nFound {len(markets)} markets:\n")
    
    # Use auto model selection
    portfolio_calc = Portfolio(model='auto')
    
    # Try to get underlying volatility for better estimates
    print("Attempting to fetch underlying asset volatility from Yahoo Finance...")
    underlying_vols = {}
    seen_tickers = set()
    
    for market in markets:
        ticker = infer_ticker_from_question(market.question)
        if ticker and ticker not in seen_tickers:
            seen_tickers.add(ticker)
            vol = VolatilityEstimator.from_underlying_asset(market.question)
            if vol:
                print(f"  {ticker}: {vol*100:.1f}% annualized")
                # Map this vol to all markets with this ticker
                for m in markets:
                    if infer_ticker_from_question(m.question) == ticker:
                        underlying_vols[m.id] = vol
    
    # Default sigma
    default_sigma = 0.8
    
    print(f"\n{'#':<3} {'Market':<45} {'YES':>7} {'Days':>6} {'Model':>7} {'Vol%':>6} {'Gamma':>8}")
    print("-" * 90)
    
    valid_markets = []
    for i, market in enumerate(markets):
        # Calculate time to expiry
        now = datetime.now(timezone.utc)
        tau_seconds = (market.end_date - now).total_seconds()
        tau_days = tau_seconds / (24 * 3600)
        tau_years = tau_seconds / (365.25 * 24 * 3600)
        
        if tau_days <= 0:
            continue
        
        valid_markets.append(market)
        
        # Determine which model to use
        model_used = 'Bridge' if tau_days > Portfolio.BRIDGE_THRESHOLD_DAYS else 'BS'
        
        # Use underlying volatility if available, otherwise default
        sigma = underlying_vols.get(market.id, default_sigma)
        
        # Calculate Greeks with appropriate model
        if model_used == 'Bridge':
            T_total = 30 / 365
            greeks = BrownianBridge().greeks(market.yes_price, sigma, tau_years, T_total, is_yes=True)
        else:
            greeks = BlackScholesDigital().greeks(market.yes_price, sigma, tau_years, is_yes=True)
        
        # Truncate question for display
        q = market.question[:42] + "..." if len(market.question) > 45 else market.question
        
        print(f"{len(valid_markets):<3} {q:<45} {market.yes_price*100:>6.1f}% {tau_days:>5.0f}d {model_used:>7} {sigma*100:>5.0f}% {greeks.gamma:>+8.4f}")
    
    if not valid_markets:
        print("No valid markets with future expiry dates found.")
        return
    
    # Detailed view of ALL markets
    print("\n" + "=" * 70)
    print("DETAILED GREEKS FOR ALL MARKETS")
    print("=" * 70)
    
    bs = BlackScholesDigital()
    bridge = BrownianBridge()
    
    for market in valid_markets:
        now = datetime.now(timezone.utc)
        tau_years = (market.end_date - now).total_seconds() / (365.25 * 24 * 3600)
        tau_days = tau_years * 365
        T_total = 30 / 365
        
        if tau_years <= 0:
            continue
        
        # Use underlying vol if available
        sigma = underlying_vols.get(market.id, default_sigma)
        
        # Determine recommended model
        recommended = 'Bridge' if tau_days > Portfolio.BRIDGE_THRESHOLD_DAYS else 'BS'
        
        print(f"\n📊 {market.question}")
        print(f"   YES: {market.yes_price*100:.1f}% | NO: {market.no_price*100:.1f}% | Expires in {tau_days:.0f} days")
        print(f"   Volume: ${market.volume:,.0f} | Liquidity: ${market.liquidity:,.0f}")
        print(f"   Recommended model: {recommended} | Using volatility: {sigma*100:.1f}%")
        
        bs_g = bs.greeks(market.yes_price, sigma, tau_years, is_yes=True)
        br_g = bridge.greeks(market.yes_price, sigma, tau_years, T_total, is_yes=True)
        
        print(f"\n   {'Greek':<8} {'Black-Scholes':>14} {'Brownian Bridge':>16}")
        print(f"   {'-'*42}")
        print(f"   {'Delta':<8} {bs_g.delta:>14.4f} {br_g.delta:>16.4f}")
        print(f"   {'Gamma':<8} {bs_g.gamma:>14.4f} {br_g.gamma:>16.4f}")
        print(f"   {'Vega':<8} {bs_g.vega:>14.4f} {br_g.vega:>16.4f}")
        print(f"   {'Theta':<8} {bs_g.theta:>14.6f} {br_g.theta:>16.6f}")
        
        # Try to estimate volatility from price history
        if market.clob_token_ids:
            token_id = market.clob_token_ids[0] if market.clob_token_ids else None
            if token_id:
                history = fetch_price_history(token_id, "1w")
                if len(history) >= 2:
                    est_vol = VolatilityEstimator.from_price_history(history)
                    print(f"\n   📈 Estimated vol from price history: {est_vol*100:.1f}%")
    
    # Portfolio simulation
    print("\n" + "=" * 70)
    print("SAMPLE PORTFOLIO (simulated positions)")
    print("=" * 70)
    
    if len(valid_markets) >= 2:
        portfolio = Portfolio(model='auto')
        
        m1 = valid_markets[0]
        m2 = valid_markets[1]
        
        # Create realistic entry prices
        entry1 = max(0.01, m1.yes_price - 0.05)
        entry2 = max(0.01, m2.no_price - 0.03)
        
        pos1 = Position(
            market=m1,
            quantity=100,
            is_yes=True,
            entry_price=entry1
        )
        pos2 = Position(
            market=m2,
            quantity=50,
            is_yes=False,
            entry_price=entry2
        )
        
        portfolio.add_position(pos1)
        portfolio.add_position(pos2)
        
        print(f"\nPositions:")
        for pos in portfolio.positions:
            side = "YES" if pos.is_yes else "NO"
            price = pos.market.yes_price if pos.is_yes else pos.market.no_price
            model_used = 'Bridge' if portfolio._time_to_expiry_days(pos.market) > Portfolio.BRIDGE_THRESHOLD_DAYS else 'BS'
            print(f"  • {pos.quantity} {side} @ {pos.market.question[:40]}...")
            print(f"    Entry: ${pos.entry_price:.2f} → Current: ${price:.2f} | P&L: ${pos.pnl:+.2f} | Model: {model_used}")
        
        # Calculate aggregate Greeks (use underlying vols if available)
        vols = {m.id: underlying_vols.get(m.id, default_sigma) for m in valid_markets}
        agg = portfolio.aggregate_greeks(volatilities=vols)
        
        print(f"\nAggregate Portfolio Greeks:")
        for k, v in agg.to_dict().items():
            print(f"  {k.capitalize():>6}: {v:>12.4f}")
        
        # Show breakdown
        print(f"\nGreeks Breakdown by Position:")
        breakdown = portfolio.greeks_breakdown(volatilities=vols)
        for pos, greeks in breakdown:
            side = "YES" if pos.is_yes else "NO"
            print(f"  {pos.quantity} {side} {pos.market.question[:30]}... → Δ={greeks.delta:.2f}, Γ={greeks.gamma:.4f}")


def run_mock_demo():
    """Run the Greeks calculator with mock data."""
    print("=" * 70)
    print("POLYMARKET BINARY OPTIONS GREEKS CALCULATOR")
    print("=" * 70)
    
    # Create mock market
    mock_data = {
        "id": "12345",
        "question": "Will BTC be above $100k by Feb 1?",
        "conditionId": "0xabc123",
        "slug": "btc-above-100k-feb",
        "endDate": "2025-02-01T00:00:00Z",
        "outcomePrices": '["0.65", "0.35"]',
        "clobTokenIds": '["token_yes", "token_no"]',
        "outcomes": '["Yes", "No"]',
        "volumeNum": 1000000,
        "liquidityNum": 50000,
    }
    
    market = PolymarketMarket.from_api_response(mock_data)
    print(f"\nMarket: {market.question}")
    print(f"YES: {market.yes_price*100:.1f}%  |  NO: {market.no_price*100:.1f}%")
    
    # Initialize models
    bs = BlackScholesDigital()
    bridge = BrownianBridge()
    
    p = market.yes_price
    sigma = 1.0  # 100% annualized vol (typical for prediction markets)
    T_total = 30 / 365
    
    print("\n" + "=" * 70)
    print("GREEKS COMPARISON: BLACK-SCHOLES vs BROWNIAN BRIDGE")
    print("=" * 70)
    print(f"Parameters: p={p}, σ={sigma}, T_total={T_total*365:.0f} days")
    
    for days in [14, 7, 3, 1]:
        tau = days / 365
        
        bs_g = bs.greeks(p, sigma, tau, is_yes=True)
        br_g = bridge.greeks(p, sigma, tau, T_total, is_yes=True)
        
        print(f"\n--- {days} days to expiry ---")
        print(f"{'Greek':<8} {'Black-Scholes':>14} {'Brownian Bridge':>16}")
        print("-" * 42)
        print(f"{'Delta':<8} {bs_g.delta:>14.4f} {br_g.delta:>16.4f}")
        print(f"{'Gamma':<8} {bs_g.gamma:>14.4f} {br_g.gamma:>16.4f}")
        print(f"{'Vega':<8} {bs_g.vega:>14.4f} {br_g.vega:>16.4f}")
        print(f"{'Theta':<8} {bs_g.theta:>14.6f} {br_g.theta:>16.6f}")
    
    print("\n" + "=" * 70)
    print("GAMMA ACROSS PROBABILITIES (7 days to expiry)")
    print("=" * 70)
    print(f"\n{'Prob':>6} {'BS Gamma':>12} {'Bridge Gamma':>14} {'Interpretation':<30}")
    print("-" * 65)
    
    tau = 7 / 365
    for prob in [0.10, 0.25, 0.50, 0.75, 0.90]:
        bs_g = bs.greeks(prob, sigma, tau, is_yes=True)
        br_g = bridge.greeks(prob, sigma, tau, T_total, is_yes=True)
        
        if prob < 0.5:
            interp = "Underdog - vol helps"
        elif prob > 0.5:
            interp = "Favorite - vol hurts"
        else:
            interp = "Neutral"
        
        print(f"{prob:>6.0%} {bs_g.gamma:>12.4f} {br_g.gamma:>14.4f} {interp:<30}")
    
    print("\n" + "=" * 70)
    print("PORTFOLIO EXAMPLE")
    print("=" * 70)
    
    portfolio = Portfolio(model='auto')  # Auto-selects based on time to expiry
    
    position = Position(
        market=market,
        quantity=100,
        is_yes=True,
        entry_price=0.60,
    )
    portfolio.add_position(position)
    
    greeks = portfolio.aggregate_greeks(volatilities={market.id: sigma})
    
    print(f"\nPosition: 100 YES shares @ ${position.entry_price:.2f}")
    print(f"Current price: ${market.yes_price:.2f}")
    print(f"Unrealized P&L: ${position.pnl:.2f}")
    print(f"\nPortfolio Greeks:")
    for k, v in greeks.to_dict().items():
        print(f"  {k.capitalize():>6}: {v:>10.4f}")
    
    print("\n" + "=" * 70)
    print("TRADE IMPACT ANALYSIS")
    print("=" * 70)
    
    # Hypothetical second market
    mock_data2 = mock_data.copy()
    mock_data2['id'] = '67890'
    mock_data2['question'] = 'Will ETH be above $4k?'
    mock_data2['outcomePrices'] = '["0.40", "0.60"]'
    market2 = PolymarketMarket.from_api_response(mock_data2)
    
    impact = analyze_trade_impact(portfolio, market2, quantity=50, is_yes=True, sigma=sigma)
    
    print(f"\nProposed: Buy 50 YES shares of '{market2.question}'")
    print(f"\n{'Metric':<8} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 42)
    for metric in ['delta', 'gamma', 'vega', 'theta']:
        print(f"{metric.capitalize():<8} {impact['before'][metric]:>10.4f} {impact['after'][metric]:>10.4f} {impact['change'][metric]:>+10.4f}")
    
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print("""
1. DELTA = 1 for a $1 notional binary option (linear in probability)

2. GAMMA reflects how "risky" the position is:
   - Positive when p < 0.5 (underdog benefits from vol)
   - Negative when p > 0.5 (favorite hurt by vol)
   - Zero at p = 0.5

3. VEGA shows volatility sensitivity:
   - Favorites (p > 0.5) have negative vega
   - Underdogs (p < 0.5) have positive vega

4. THETA is the daily P&L from time passing:
   - Positive if you're winning (p > 0.5 for YES)
   - Negative if you're losing

5. BROWNIAN BRIDGE naturally handles near-expiry:
   - Effective vol σ_eff = σ * √(τ/T) → 0 as τ→0
   - No need for ad-hoc Greek capping
   - More theoretically sound for binary outcomes
""")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Polymarket Binary Options Greeks Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python polymarket_greeks.py              # Run with mock data
  python polymarket_greeks.py --live       # Fetch live Polymarket data
  python polymarket_greeks.py --live -n 20 # Fetch top 20 markets
  python polymarket_greeks.py --live -s "bitcoin" # Search for Bitcoin markets
        """
    )
    parser.add_argument(
        "--live", "-l",
        action="store_true",
        help="Fetch live data from Polymarket API"
    )
    parser.add_argument(
        "-n", "--num-markets",
        type=int,
        default=10,
        help="Number of markets to fetch (default: 10)"
    )
    parser.add_argument(
        "-s", "--search",
        type=str,
        default=None,
        help="Search term to filter markets"
    )
    
    args = parser.parse_args()
    
    if args.live:
        run_live_demo()
    else:
        run_mock_demo()