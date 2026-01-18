"""
Polymarket Binary Options Greeks Calculator - Live Data Version

Calculate option Greeks for live Polymarket prediction markets.
Supports both probability-based models and barrier option models for markets with strikes.

Greeks are NORMALIZED for practical use:
- Delta: $ change per 1% move in underlying (or per 1pp move in probability)
- Gamma: Delta change per 1% move
- Vega: $ change per 1% increase in volatility
- Theta: $ change per day

Usage:
    python polymarket_greeks.py -n 20                    # Fetch top 20 markets
    python polymarket_greeks.py -s "bitcoin"             # Search for Bitcoin markets
    python polymarket_greeks.py -s "silver" -n 15        # Top 15 Silver markets

Author: Pol (Polymarket Hackathon 2025)
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timezone
import json
import re


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PolymarketMarket:
    id: str
    question: str
    condition_id: str
    slug: str
    end_date: datetime
    outcome_prices: List[float]
    clob_token_ids: List[str]
    outcomes: List[str]
    volume: float = 0.0
    liquidity: float = 0.0
    related_commodity: Optional[str] = None  # NEW: from event level
    
    @classmethod
    def from_api_response(cls, data: dict, event_data: dict = None) -> 'PolymarketMarket':
        outcome_prices = [float(p) for p in json.loads(data.get('outcomePrices', '["0.5", "0.5"]'))]
        clob_token_ids = json.loads(data.get('clobTokenIds', '[]'))
        if isinstance(clob_token_ids, str):
            clob_token_ids = [clob_token_ids]
        outcomes = json.loads(data.get('outcomes', '["Yes", "No"]'))
        
        # FIXED: Better date parsing
        end_date_str = data.get('endDate') or data.get('endDateIso')
        if end_date_str:
            try:
                # Handle ISO format with Z
                if isinstance(end_date_str, str):
                    if end_date_str.endswith('Z'):
                        end_date_str = end_date_str[:-1] + '+00:00'
                    end_date = datetime.fromisoformat(end_date_str)
                else:
                    # Might be timestamp
                    end_date = datetime.fromtimestamp(float(end_date_str), tz=timezone.utc)
            except Exception as e:
                print(f"Warning: Could not parse date '{end_date_str}': {e}")
                end_date = datetime.now(timezone.utc)
        else:
            end_date = datetime.now(timezone.utc)
        
        # Extract relatedCommodity from event data if available
        related_commodity = None
        if event_data:
            related_commodity = event_data.get('relatedCommodity')
        
        return cls(
            id=data.get('id', ''), question=data.get('question', ''), condition_id=data.get('conditionId', ''),
            slug=data.get('slug', ''), end_date=end_date, outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids, outcomes=outcomes, volume=float(data.get('volumeNum', 0) or 0),
            liquidity=float(data.get('liquidityNum', 0) or 0), related_commodity=related_commodity
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
    Normalized Greeks for practical interpretation:
    - delta: $ change per 1% move in underlying (or 1pp probability)
    - gamma: Change in delta per 1% move
    - vega: $ change per 1% vol increase
    - theta: $ change per day
    """
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float = 0.0
    
    def __add__(self, other: 'Greeks') -> 'Greeks':
        return Greeks(self.delta + other.delta, self.gamma + other.gamma, self.vega + other.vega,
                     self.theta + other.theta, self.rho + other.rho)
    
    def __mul__(self, scalar: float) -> 'Greeks':
        return Greeks(self.delta * scalar, self.gamma * scalar, self.vega * scalar,
                     self.theta * scalar, self.rho * scalar)
    
    def __rmul__(self, scalar: float) -> 'Greeks':
        return self.__mul__(scalar)
    
    def to_dict(self) -> dict:
        return {'delta': round(self.delta, 4), 'gamma': round(self.gamma, 4), 
                'vega': round(self.vega, 4), 'theta': round(self.theta, 6), 'rho': round(self.rho, 4)}


# =============================================================================
# PROBABILITY-BASED MODELS (for markets without clear underlying)
# =============================================================================

class BlackScholesDigital:
    """Greeks for binary options in probability space."""
    
    def greeks(self, p: float, sigma: float, T: float, is_yes: bool = True) -> Greeks:
        """
        Normalized Greeks for a digital option on probability.
        
        Delta = $0.01 per 1 percentage point move in probability
        (e.g., if p moves from 65% to 66%, position changes by $0.01 per share)
        """
        if T <= 1e-10:
            return Greeks(0, 0, 0, 0, 0)
        
        p = np.clip(p, 0.01, 0.99)
        sqrt_T = np.sqrt(T)
        p_vol = sigma * p * (1 - p)
        
        # Delta: For a digital option, value = p, so dV/dp = 1
        # Normalized: $0.01 change per 1pp move (since shares are $1 max)
        delta = 0.01
        
        # Gamma: d(delta)/dp - how does sensitivity change?
        # In prob space: gamma = sigma * (1 - 2p) reflects vol changing with p
        gamma = sigma * (1 - 2*p) * 0.0001  # Normalized and scaled
        
        # Vega: dV/d(sigma) - sensitivity to vol
        vega = -(p - 0.5) * sqrt_T * p * (1 - p) * 0.01
        
        # Theta: time decay per day
        theta_dir = p - 0.5
        theta_mag = p_vol**2 / (2 * T) if T > 0 else 0
        theta = theta_dir * theta_mag / 365
        
        sign = 1 if is_yes else -1
        return Greeks(sign * delta, sign * gamma, sign * vega, sign * theta, 0)


class BrownianBridge:
    """Brownian Bridge model - better for prediction markets."""
    
    def greeks(self, p: float, sigma: float, tau: float, T: float, is_yes: bool = True) -> Greeks:
        """
        Greeks with bridge dynamics (vol decays as expiry approaches).
        sigma_eff = sigma * sqrt(tau/T)
        """
        if tau <= 1e-10:
            return Greeks(0, 0, 0, 0, 0)
        
        T = max(T, tau)
        p = np.clip(p, 0.01, 0.99)
        sigma_eff = sigma * np.sqrt(tau / T)
        p_vol = sigma_eff * p * (1 - p)
        
        delta = 0.01  # $0.01 per 1pp move
        gamma = sigma_eff * (1 - 2*p) * 0.0001
        vega = -(p - 0.5) * np.sqrt(tau / T) * p * (1 - p) * 0.01
        
        theta_dir = p - 0.5
        theta_mag = p_vol**2 / (2 * tau) if tau > 0 else 0
        theta_decay = 0.5 * sigma * p * (1-p) / np.sqrt(tau * T) if tau > 0 and T > 0 else 0
        theta = (theta_dir * theta_mag - abs(theta_dir) * theta_decay) / 365
        
        sign = 1 if is_yes else -1
        return Greeks(sign * delta, sign * gamma, sign * vega, sign * theta, 0)


# =============================================================================
# BARRIER OPTION MODEL (for markets with underlying asset and strike)
# =============================================================================

class BarrierOptionGreeks:
    """
    Digital/binary option Greeks with underlying asset.
    
    NORMALIZED for practical use:
    - Delta: $ change per 1% move in underlying
    - Gamma: Delta change per 1% move in underlying
    - Vega: $ change per 1 percentage point vol increase
    - Theta: $ change per day
    """
    
    def digital_call_greeks(self, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> Greeks:
        """
        Greeks for a digital call (pays $1 if S > K at expiry).
        
        Args:
            S: Current underlying price
            K: Strike price
            T: Time to expiry in years
            sigma: Volatility (annualized)
            r: Risk-free rate
        """
        if T <= 1e-10:
            return Greeks(0, 0, 0, 0, 0)
        
        sigma = max(sigma, 0.01)
        sqrt_T = np.sqrt(T)
        
        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        n_d2 = norm.pdf(d2)
        
        # Raw Greeks (per $1 move in underlying)
        raw_delta = n_d2 / (S * sigma * sqrt_T)
        raw_gamma = -n_d2 * (d2 + sigma * sqrt_T) / (S**2 * sigma**2 * T)
        
        # NORMALIZED Delta: $ change per 1% move in S
        # 1% move = S * 0.01, so delta_norm = raw_delta * S * 0.01
        delta = raw_delta * S * 0.01
        
        # NORMALIZED Gamma: change in delta per 1% move
        # We want d(delta_norm)/d(1% move in S)
        # delta_norm = raw_delta * S * 0.01
        # d(delta_norm)/dS = (d(raw_delta)/dS * S + raw_delta) * 0.01
        # d(delta_norm)/d(1% move) = d(delta_norm)/dS * S * 0.01
        gamma = (raw_gamma * S + raw_delta) * S * 0.0001
        
        # NORMALIZED Vega: $ change per 1pp vol increase
        d1 = d2 + sigma * sqrt_T
        raw_vega = n_d2 * d1 * sqrt_T / sigma
        vega = raw_vega * 0.01
        
        # Theta: daily decay (in $ per day)
        theta_term1 = n_d2 * sigma / (2 * sqrt_T)
        theta_term2 = n_d2 * d2 * (r - 0.5 * sigma**2) / (sigma * sqrt_T)
        theta = -(theta_term1 - theta_term2 * 0.5) / 365
        
        return Greeks(delta, gamma, vega, theta, 0)
    
    def digital_put_greeks(self, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> Greeks:
        """Greeks for a digital put (pays $1 if S < K at expiry)."""
        g = self.digital_call_greeks(S, K, T, sigma, r)
        return Greeks(-g.delta, g.gamma, -g.vega, g.theta, 0)
    
    def greeks_from_market(self, market: PolymarketMarket, current_price: float, 
                          sigma: float, is_yes: bool = True) -> Optional[Greeks]:
        """Calculate Greeks for a market with known underlying price."""
        strike = extract_strike_from_question(market.question)
        option_type = infer_option_type(market.question)
        
        if not strike or option_type == 'unknown':
            return None
        
        tau = max((market.end_date - datetime.now(timezone.utc)).total_seconds() / (365.25 * 24 * 3600), 1/365)
        
        if option_type == 'call':
            greeks = self.digital_call_greeks(current_price, strike, tau, sigma)
        else:
            greeks = self.digital_put_greeks(current_price, strike, tau, sigma)
        
        # If holding NO, flip the sign on directional greeks
        if not is_yes:
            greeks = Greeks(-greeks.delta, greeks.gamma, -greeks.vega, greeks.theta, 0)
        
        return greeks


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_strike_from_question(question: str) -> Optional[float]:
    """Extract strike price from market question."""
    patterns = [
        r'\$([0-9,]+(?:\.[0-9]+)?)',
        r'([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars|usd)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        if matches:
            try:
                return float(matches[0].replace(',', ''))
            except:
                pass
    return None


def infer_option_type(question: str) -> str:
    """Infer if market is call-like or put-like."""
    q = question.lower()
    if any(w in q for w in ['reach', 'above', 'over', 'exceed', 'higher than', 'hit (high)', 'hit(high)']):
        return 'call'
    if any(w in q for w in ['dip', 'below', 'under', 'fall', 'lower than', 'hit (low)', 'hit(low)']):
        return 'put'
    return 'unknown'


def infer_ticker_from_question(question: str) -> Optional[str]:
    """Infer Yahoo Finance ticker from market question."""
    q = question.lower()
    tickers = {
        'bitcoin': 'BTC-USD', 'btc': 'BTC-USD',
        'ethereum': 'ETH-USD', 'eth': 'ETH-USD', 'ether': 'ETH-USD',
        'solana': 'SOL-USD', 'sol': 'SOL-USD',
        'gold': 'GC=F', 'silver': 'SI=F',
        'oil': 'CL=F', 'crude': 'CL=F', 'wti': 'CL=F',
        's&p 500': '^GSPC', 'sp500': '^GSPC', 's&p': '^GSPC',
        'nasdaq': '^IXIC', 'dow': '^DJI',
        'tesla': 'TSLA', 'apple': 'AAPL', 'nvidia': 'NVDA',
        'usd index': 'DX-Y.NYB', 'dxy': 'DX-Y.NYB', 'dollar index': 'DX-Y.NYB',
    }
    for key, ticker in tickers.items():
        if key in q:
            return ticker
    return None


def fetch_yahoo_current_price(ticker: str) -> Optional[float]:
    """Fetch current price from Yahoo Finance."""
    try:
        import yfinance as yf
        data = yf.Ticker(ticker).history(period="1d")
        return float(data['Close'].iloc[-1]) if len(data) > 0 else None
    except Exception as e:
        return None


def fetch_yahoo_prices(ticker: str, period: str = "1mo") -> Optional[List[float]]:
    """Fetch historical prices from Yahoo Finance."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period)
        return hist['Close'].tolist() if len(hist) > 0 else None
    except:
        return None


def calculate_realized_vol(prices: List[float], annualize: bool = True) -> float:
    """Calculate realized volatility from price series."""
    if len(prices) < 2:
        return 0.5
    returns = np.diff(np.log(np.array(prices)))
    vol = np.std(returns)
    return vol * np.sqrt(252) if annualize else vol


# =============================================================================
# API FUNCTIONS
# =============================================================================

def load_markets_from_json(filepath: str) -> List[PolymarketMarket]:
    """Load markets from a JSON file."""
    import json
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        markets = []
        
        # Handle both event-based and direct market array formats
        if 'events' in data:
            # Event-based format - pass event data to capture relatedCommodity
            for event in data['events']:
                for market_data in event.get('markets', []):
                    try:
                        markets.append(PolymarketMarket.from_api_response(market_data, event_data=event))
                    except Exception as e:
                        print(f"Warning: Failed to parse market from JSON: {e}")
        elif isinstance(data, list):
            # Direct market array format
            for market_data in data:
                try:
                    markets.append(PolymarketMarket.from_api_response(market_data))
                except Exception as e:
                    print(f"Warning: Failed to parse market from JSON: {e}")
        else:
            print(f"Error: Unrecognized JSON format")
            return []
        
        return markets
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{filepath}': {e}")
        return []


def categorize_markets_by_asset(markets: List[PolymarketMarket]) -> Dict[str, List[PolymarketMarket]]:
    """
    Categorize markets by their underlying asset.
    Uses relatedCommodity field if available, otherwise infers from question text.
    Returns dict mapping asset name -> list of markets.
    """
    categories = {
        'bitcoin': [],
        'ethereum': [],
        'solana': [],
        'gold': [],
        'silver': [],
        'oil': [],
        'usd_index': [],
        'other': []
    }
    
    # Mapping from relatedCommodity values to our categories
    # relatedCommodity uses spaces and lowercase
    commodity_mapping = {
        'bitcoin': 'bitcoin',
        'btc': 'bitcoin',
        'ethereum': 'ethereum',
        'eth': 'ethereum',
        'solana': 'solana',
        'sol': 'solana',
        'gold': 'gold',
        'silver': 'silver',
        'oil': 'oil',
        'crude': 'oil',
        'crude oil': 'oil',
        'wti': 'oil',
        'usd index': 'usd_index',
        'dollar index': 'usd_index',
        'dxy': 'usd_index',
    }
    
    for market in markets:
        categorized = False
        
        # PRIORITY 1: Use relatedCommodity field if available
        if market.related_commodity:
            commodity_lower = market.related_commodity.lower().strip()
            if commodity_lower in commodity_mapping:
                categories[commodity_mapping[commodity_lower]].append(market)
                categorized = True
        
        # PRIORITY 2: Infer from question text (only if not already categorized)
        if not categorized:
            q_lower = market.question.lower()
            
            # Check which asset this market is about
            if any(term in q_lower for term in ['bitcoin', 'btc']):
                categories['bitcoin'].append(market)
            elif any(term in q_lower for term in ['ethereum', 'eth', 'ether']):
                categories['ethereum'].append(market)
            elif any(term in q_lower for term in ['solana', 'sol']):
                categories['solana'].append(market)
            elif 'gold' in q_lower and 'golden' not in q_lower:
                categories['gold'].append(market)
            elif 'silver' in q_lower:
                categories['silver'].append(market)
            elif any(term in q_lower for term in ['oil', 'crude', 'wti', 'brent']):
                categories['oil'].append(market)
            elif any(term in q_lower for term in ['usd index', 'dollar index', 'dxy', 'iranian rial', 'rials', 'rial']):
                categories['usd_index'].append(market)
            else:
                categories['other'].append(market)
    
    return categories
    """
    Check if search term is relevant in the question.
    Filters out false matches like 'oil' in 'Oilers' or 'gold' in 'Golden Knights'.
    """
    question_lower = question.lower()
    search_lower = search_term.lower()
    
    # Explicit sports team filters to exclude
    sports_exclusions = [
        'oilers', 'golden knights', 'golden state warriors', 
        'predators vs', 'knights vs', 'vs golden'
    ]
    
    # Check if this is a sports match
    for exclusion in sports_exclusions:
        if exclusion in question_lower:
            return False
    
    # For common assets, also check for expanded terms
    asset_expansions = {
        'oil': ['oil', 'crude', 'wti', 'brent', 'petroleum'],
        'gold': ['gold', 'xau'],
        'silver': ['silver', 'xag'],
        'bitcoin': ['bitcoin', 'btc'],
        'ethereum': ['ethereum', 'eth', 'ether'],
        'solana': ['solana', 'sol'],
    }
    
    # If it's a known asset, check for any expansion
    if search_lower in asset_expansions:
        for term in asset_expansions[search_lower]:
            if term in question_lower:
                return True
        return False
    
    # For other searches, simple substring match
    return search_lower in question_lower


def is_relevant_match(question: str, search_term: str) -> bool:
    """
    Check if search term is relevant in the question.
    Filters out false matches like 'oil' in 'Oilers' or 'gold' in 'Golden Knights'.
    """
    question_lower = question.lower()
    search_lower = search_term.lower()
    
    # Explicit sports team filters to exclude
    sports_exclusions = [
        'oilers', 'golden knights', 'golden state warriors', 
        'predators vs', 'knights vs', 'vs golden'
    ]
    
    # Check if this is a sports match
    for exclusion in sports_exclusions:
        if exclusion in question_lower:
            return False
    
    # For common assets, also check for expanded terms
    asset_expansions = {
        'oil': ['oil', 'crude', 'wti', 'brent', 'petroleum'],
        'gold': ['gold', 'xau'],
        'silver': ['silver', 'xag'],
        'bitcoin': ['bitcoin', 'btc'],
        'ethereum': ['ethereum', 'eth', 'ether'],
        'solana': ['solana', 'sol'],
        'usd_index': ['usd index', 'dollar index', 'dxy'],
    }
    
    # If it's a known asset, check for any expansion
    if search_lower in asset_expansions:
        for term in asset_expansions[search_lower]:
            if term in question_lower:
                return True
        return False
    
    # For other searches, simple substring match
    return search_lower in question_lower


def fetch_live_markets(limit: int = 10, search: str = None, debug: bool = False) -> List[PolymarketMarket]:
    """Fetch markets from Polymarket API."""
    import requests
    
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip, deflate',
        'User-Agent': 'PolymarketGreeksCalculator/1.0',
    }
    
    if search:
        # FIXED: Search through markets endpoint directly for better results
        url = "https://gamma-api.polymarket.com/markets"
        params = {
            "active": "true", 
            "closed": "false", 
            "limit": 500,  # Fetch many more to ensure we find commodity markets
            "order": "volume24hr", 
            "ascending": "false"
        }
    else:
        url = "https://gamma-api.polymarket.com/markets"
        params = {"active": "true", "closed": "false", "limit": limit, "order": "volume24hr", "ascending": "false"}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        markets = []
        
        if search:
            # FIXED: Use smart matching to avoid false positives
            filtered_count = 0
            for m in data:
                try:
                    market = PolymarketMarket.from_api_response(m)
                    # Use relevance checking
                    if is_relevant_match(market.question, search):
                        markets.append(market)
                        if len(markets) >= limit:
                            break
                    else:
                        filtered_count += 1
                        if debug and filtered_count <= 5:
                            print(f"  [DEBUG] Filtered out: {market.question[:60]}")
                except Exception as e:
                    if debug:
                        print(f"Warning: Failed to parse market: {e}")
                    continue
            if debug:
                print(f"  [DEBUG] Filtered {filtered_count} markets, kept {len(markets)}")
        else:
            for m in data:
                try:
                    markets.append(PolymarketMarket.from_api_response(m))
                except Exception as e:
                    print(f"Warning: Failed to parse market: {e}")
                    continue
        
        return markets[:limit]
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Polymarket Greeks Calculator - Live Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # From JSON file (recommended):2
  python polymarket_greeks.py -f markets.json
  python polymarket_greeks.py --file my_markets.json --asset bitcoin
  
  # From API search:
  python polymarket_greeks.py -n 20                  # Top 20 markets
  python polymarket_greeks.py -s "bitcoin"           # Bitcoin markets
  python polymarket_greeks.py -s "silver" -n 15      # Top 15 silver markets
  python polymarket_greeks.py --asset oil            # Oil markets

Popular assets: bitcoin, ethereum, solana, gold, silver, oil, usd_index
        """
    )
    parser.add_argument("-f", "--file", type=str, default=None, 
                       help="Load markets from JSON file instead of API")
    parser.add_argument("-n", "--num-markets", type=int, default=10, help="Number of markets")
    parser.add_argument("-s", "--search", type=str, default=None, 
                       help="Search term (e.g., bitcoin, gold, silver, oil, ethereum)")
    parser.add_argument("--asset", type=str, default=None, 
                       choices=['bitcoin', 'ethereum', 'solana', 'gold', 'silver', 'oil', 'usd_index'],
                       help="Filter by specific asset (works with both -f and -s)")
    parser.add_argument("--debug", action="store_true", help="Show debug info about filtered markets")
    args = parser.parse_args()
    
    # Use --asset flag if provided
    if args.asset and not args.search:
        args.search = args.asset
    
    print("=" * 85)
    print("POLYMARKET GREEKS CALCULATOR - LIVE DATA")
    print("=" * 85)
    print("\nGreeks are NORMALIZED for practical interpretation:")
    print("  Delta: $ change per 1% move in underlying (or 1pp probability)")
    print("  Gamma: Delta change per 1% move")
    print("  Vega:  $ change per 1pp vol increase")
    print("  Theta: $ change per day")
    
    # Load markets from JSON file or API
    if args.file:
        print(f"\nLoading markets from: {args.file}")
        all_markets = load_markets_from_json(args.file)
        
        if not all_markets:
            print("No markets loaded from file.")
            return
        
        # Debug: Show relatedCommodity info
        if args.debug:
            print(f"\n[DEBUG] Loaded {len(all_markets)} markets")
            print(f"[DEBUG] Markets with relatedCommodity field:")
            with_commodity = [m for m in all_markets if m.related_commodity]
            without_commodity = [m for m in all_markets if not m.related_commodity]
            print(f"  With field: {len(with_commodity)}")
            print(f"  Without field: {len(without_commodity)}")
            if with_commodity:
                print(f"\n[DEBUG] Sample relatedCommodity values:")
                for m in with_commodity[:10]:
                    print(f"  - '{m.related_commodity}' → {m.question[:50]}...")
        
        # Categorize by asset
        categorized = categorize_markets_by_asset(all_markets)
        
        if args.debug:
            print(f"\n[DEBUG] Categorization results:")
            for cat, markets_list in categorized.items():
                if markets_list:
                    print(f"  {cat}: {len(markets_list)} markets")
        
        # Filter by asset if specified
        if args.asset:
            markets = categorized.get(args.asset, [])
            print(f"Found {len(markets)} {args.asset} markets from {len(all_markets)} total markets")
        else:
            markets = all_markets
            print(f"Loaded {len(markets)} markets from file")
            print("\nMarkets by asset:")
            for asset, asset_markets in categorized.items():
                if asset_markets and asset != 'other':
                    print(f"  {asset.replace('_', ' ').capitalize()}: {len(asset_markets)} markets")
        
    else:
        # Use API search
        if args.search:
            print(f"\nSearching for markets matching: '{args.search}'...")
        else:
            print(f"\nFetching top {args.num_markets} markets by volume...")
        
        markets = fetch_live_markets(limit=args.num_markets, search=args.search, debug=args.debug)
    
    if not markets:
        print("No markets found.")
        return
    
    print(f"\nAnalyzing {len(markets)} markets\n")
    
    # Fetch Yahoo Finance data
    underlying_vols = {}
    underlying_prices = {}
    
    try:
        import yfinance
        print("Fetching underlying asset data from Yahoo Finance...")
        seen_tickers = set()
        
        for m in markets:
            ticker = infer_ticker_from_question(m.question)
            if ticker and ticker not in seen_tickers:
                seen_tickers.add(ticker)
                print(f"  {ticker}: ", end='', flush=True)
                
                price = fetch_yahoo_current_price(ticker)
                if price:
                    underlying_prices[ticker] = price
                    print(f"${price:,.2f}", end='')
                
                prices = fetch_yahoo_prices(ticker)
                if prices:
                    vol = calculate_realized_vol(prices)
                    print(f", vol={vol*100:.1f}%")
                    for market in markets:
                        if infer_ticker_from_question(market.question) == ticker:
                            underlying_vols[market.id] = vol
                else:
                    print()
        print()
    except ImportError:
        print("Note: Install yfinance for barrier option Greeks (pip install yfinance)\n")
    
    # Display all markets
    print(f"{'#':<3} {'Market':<44} {'YES':>6} {'Days':>5} {'Model':>7} {'Delta':>9} {'Gamma':>11}")
    print("-" * 92)
    
    bs = BlackScholesDigital()
    bridge = BrownianBridge()
    barrier = BarrierOptionGreeks()
    
    valid_markets = []
    market_greeks = {}
    
    for m in markets:
        tau_days = (m.end_date - datetime.now(timezone.utc)).total_seconds() / 86400
        if tau_days <= 0:
            continue
        
        valid_markets.append(m)
        tau_years = tau_days / 365.25
        sigma = underlying_vols.get(m.id, 0.5)
        
        # Try barrier model first
        ticker = infer_ticker_from_question(m.question)
        price = underlying_prices.get(ticker) if ticker else None
        strike = extract_strike_from_question(m.question)
        
        if price and strike:
            greeks = barrier.greeks_from_market(m, price, sigma, is_yes=True)
            model_name = 'Barrier'
        else:
            # Fall back to probability models
            if tau_days > 7:
                greeks = bridge.greeks(m.yes_price, sigma, tau_years, 30/365, is_yes=True)
                model_name = 'Bridge'
            else:
                greeks = bs.greeks(m.yes_price, sigma, tau_years, is_yes=True)
                model_name = 'BS'
        
        if greeks is None:
            greeks = bridge.greeks(m.yes_price, sigma, tau_years, 30/365, is_yes=True)
            model_name = 'Bridge'
        
        market_greeks[m.id] = (greeks, model_name, price, strike, sigma)
        
        q = m.question[:41] + "..." if len(m.question) > 44 else m.question
        print(f"{len(valid_markets):<3} {q:<44} {m.yes_price*100:>5.1f}% {tau_days:>4.0f}d {model_name:>7} "
              f"{greeks.delta:>+9.5f} {greeks.gamma:>+11.7f}")
    
    # Detailed breakdown for ALL markets
    print("\n" + "=" * 85)
    print("DETAILED GREEKS FOR ALL MARKETS")
    print("=" * 85)
    
    for i, m in enumerate(valid_markets):
        tau_years = (m.end_date - datetime.now(timezone.utc)).total_seconds() / (365.25 * 24 * 3600)
        tau_days = tau_years * 365.25
        greeks, model_name, price, strike, sigma = market_greeks[m.id]
        
        print(f"\n{i+1}. {m.question}")
        print(f"   YES: {m.yes_price*100:.1f}% | NO: {m.no_price*100:.1f}% | Expires: {tau_days:.0f} days | Model: {model_name}")
        print(f"   Volume: ${m.volume:,.2f} | Liquidity: ${m.liquidity:,.2f}")
        
        if price and strike:
            moneyness = (price - strike) / strike * 100
            itm_otm = "ITM" if (moneyness > 0 and infer_option_type(m.question) == 'call') or \
                               (moneyness < 0 and infer_option_type(m.question) == 'put') else "OTM"
            print(f"   Strike: ${strike:,.0f} | Spot: ${price:,.2f} | {itm_otm} by {abs(moneyness):.1f}% | Vol: {sigma*100:.0f}%")
        
        print(f"   Delta: {greeks.delta:+.6f} | Gamma: {greeks.gamma:+.8f} | Vega: {greeks.vega:+.6f} | Theta: {greeks.theta:+.8f}")
        
        # Interpretation
        if abs(greeks.delta) > 0.0001:
            direction = "up" if greeks.delta > 0 else "down"
            print(f"   → Per 1 share: ${abs(greeks.delta):.4f} gain per 1% {direction} move in underlying")
    
    # Summary statistics
    print("\n" + "=" * 85)
    print("PORTFOLIO SUMMARY (if holding 1 YES share of each)")
    print("=" * 85)
    
    total_delta = sum(g[0].delta for g in market_greeks.values())
    total_gamma = sum(g[0].gamma for g in market_greeks.values())
    total_vega = sum(g[0].vega for g in market_greeks.values())
    total_theta = sum(g[0].theta for g in market_greeks.values())
    
    print(f"\nAggregate Greeks (for 1 share each):")
    print(f"  Total Delta: {total_delta:+.6f}  (net $ exposure per 1% move)")
    print(f"  Total Gamma: {total_gamma:+.8f}  (delta sensitivity)")
    print(f"  Total Vega:  {total_vega:+.6f}  (vol sensitivity)")
    print(f"  Total Theta: {total_theta:+.8f}  (daily decay)")


if __name__ == "__main__":
    main()