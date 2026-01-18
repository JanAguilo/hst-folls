"""
Polymarket Binary Options Greeks Calculator - Live Data Version

Calculate option Greeks for live Polymarket prediction markets.
Supports both probability-based models and barrier option models for markets with strikes.

Usage:
    python polymarket_greeks.py -n 20                    # Fetch top 20 markets
    python polymarket_greeks.py -s "bitcoin"             # Search for Bitcoin markets
    python polymarket_greeks.py -s "bitcoin" -n 5        # Top 5 Bitcoin markets

Author: Pol (Polymarket Hackathon 2025)
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone
import json
import re

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
    
    @classmethod
    def from_api_response(cls, data: dict) -> 'PolymarketMarket':
        outcome_prices = [float(p) for p in json.loads(data.get('outcomePrices', '["0.5", "0.5"]'))]
        clob_token_ids = json.loads(data.get('clobTokenIds', '[]'))
        if isinstance(clob_token_ids, str):
            clob_token_ids = [clob_token_ids]
        outcomes = json.loads(data.get('outcomes', '["Yes", "No"]'))
        end_date_str = data.get('endDate') or data.get('endDateIso')
        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')) if end_date_str else datetime.now(timezone.utc)
        
        return cls(
            id=data.get('id', ''), question=data.get('question', ''), condition_id=data.get('conditionId', ''),
            slug=data.get('slug', ''), end_date=end_date, outcome_prices=outcome_prices,
            clob_token_ids=clob_token_ids, outcomes=outcomes, volume=float(data.get('volumeNum', 0) or 0),
            liquidity=float(data.get('liquidityNum', 0) or 0)
        )
    
    @property
    def yes_price(self) -> float:
        return self.outcome_prices[0] if self.outcome_prices else 0.5
    
    @property
    def no_price(self) -> float:
        return self.outcome_prices[1] if len(self.outcome_prices) > 1 else 1 - self.yes_price

@dataclass 
class Greeks:
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
        return {k: round(v, 6 if k != 'theta' else 8) for k, v in 
                [('delta', self.delta), ('gamma', self.gamma), ('vega', self.vega), 
                 ('theta', self.theta), ('rho', self.rho)]}

class BlackScholesDigital:
    def __init__(self, risk_free_rate: float = 0.0):
        self.r = risk_free_rate
    
    def greeks(self, p: float, sigma: float, T: float, is_yes: bool = True) -> Greeks:
        if T <= 1e-10:
            return Greeks(0, 0, 0, 0, 0)
        p = np.clip(p, 0.01, 0.99)
        p_vol = sigma * p * (1 - p)
        delta = 1.0
        gamma = sigma * (1 - 2*p)
        vega = -(p - 0.5) * np.sqrt(T) * p * (1 - p)
        theta = (p - 0.5) * (p_vol**2 / (2 * T)) / 365 if T > 0 else 0
        sign = 1 if is_yes else -1
        return Greeks(sign * delta, sign * gamma, sign * vega, sign * theta, 0)

class BrownianBridge:
    def effective_volatility(self, tau: float, T: float, sigma: float) -> float:
        return sigma * np.sqrt(tau / T) if T > 0 and tau > 0 else 0.0
    
    def greeks(self, p: float, sigma: float, tau: float, T: float, is_yes: bool = True) -> Greeks:
        if tau <= 1e-10:
            return Greeks(0, 0, 0, 0, 0)
        T = tau if T <= 0 else T
        p = np.clip(p, 0.01, 0.99)
        sigma_eff = self.effective_volatility(tau, T, sigma)
        p_vol = sigma_eff * p * (1 - p)
        delta = 1.0
        gamma = sigma_eff * (1 - 2*p)
        vega = -(p - 0.5) * np.sqrt(tau / T) * p * (1 - p)
        theta_dir = p - 0.5
        theta_mag = p_vol**2 / (2 * tau) if tau > 0 else 0
        theta_decay = 0.5 * sigma * p * (1-p) / np.sqrt(tau * T) if tau > 0 and T > 0 else 0
        theta = (theta_dir * theta_mag - abs(theta_dir) * theta_decay) / 365
        sign = 1 if is_yes else -1
        return Greeks(sign * delta, sign * gamma, sign * vega, sign * theta, 0)

class BarrierOptionGreeks:
    def digital_call_greeks(self, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> Greeks:
        if T <= 1e-10:
            return Greeks(0, 0, 0, 0, 0)
        sigma = max(sigma, 0.01)
        d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = norm.pdf(d2) / (S * sigma * np.sqrt(T))
        gamma = -norm.pdf(d2) * d2 / (S**2 * sigma**2 * T)
        vega = -norm.pdf(d2) * d2 * np.sqrt(T) / sigma
        theta = (-S * norm.pdf(d2) * sigma / (2 * np.sqrt(T)) + 
                norm.pdf(d2) * (r - 0.5 * sigma**2) / (sigma * np.sqrt(T))) / 365
        return Greeks(delta, gamma, vega, theta, 0)
    
    def digital_put_greeks(self, S: float, K: float, T: float, sigma: float, r: float = 0.0) -> Greeks:
        g = self.digital_call_greeks(S, K, T, sigma, r)
        return Greeks(-g.delta, -g.gamma, -g.vega, -g.theta, 0)
    
    def greeks_from_market(self, market: PolymarketMarket, current_price: float, 
                          sigma: float, is_yes: bool = True) -> Optional[Greeks]:
        strike = extract_strike_from_question(market.question)
        option_type = infer_option_type(market.question)
        if not strike or option_type == 'unknown':
            return None
        tau = max((market.end_date - datetime.now(timezone.utc)).total_seconds() / (365.25 * 24 * 3600), 1/365)
        greeks = self.digital_call_greeks(current_price, strike, tau, sigma) if option_type == 'call' else \
                 self.digital_put_greeks(current_price, strike, tau, sigma)
        return Greeks(-greeks.delta, -greeks.gamma, -greeks.vega, -greeks.theta, 0) if not is_yes else greeks

def fetch_yahoo_current_price(ticker: str) -> Optional[float]:
    try:
        import yfinance as yf
        data = yf.Ticker(ticker).history(period="1d")
        return float(data['Close'].iloc[-1]) if len(data) > 0 else None
    except Exception as e:
        print(f"      Error fetching {ticker}: {e}")
        return None

def fetch_yahoo_prices(ticker: str, period: str = "1mo") -> Optional[List[float]]:
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period=period)
        return hist['Close'].tolist() if len(hist) > 0 else None
    except:
        return None

def calculate_realized_vol(prices: List[float], annualize: bool = True) -> float:
    if len(prices) < 2:
        return 0.5
    returns = np.diff(np.log(np.array(prices)))
    vol = np.std(returns)
    return vol * np.sqrt(252) if annualize else vol

def extract_strike_from_question(question: str) -> Optional[float]:
    for pattern in [r'\$([0-9,]+(?:\.[0-9]+)?)', r'([0-9,]+(?:\.[0-9]+)?)\s*(?:dollars|usd)']:
        matches = re.findall(pattern, question, re.IGNORECASE)
        if matches:
            try:
                return float(matches[0].replace(',', ''))
            except:
                pass
    return None

def infer_option_type(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ['reach', 'above', 'over', 'exceed', 'higher than']):
        return 'call'
    if any(w in q for w in ['dip', 'below', 'under', 'fall', 'lower than']):
        return 'put'
    return 'unknown'

def infer_ticker_from_question(question: str) -> Optional[str]:
    q = question.lower()
    tickers = {
        'bitcoin': 'BTC-USD', 'btc': 'BTC-USD', 'ethereum': 'ETH-USD', 'eth': 'ETH-USD',
        'solana': 'SOL-USD', 'sol': 'SOL-USD', 'gold': 'GC=F', 'silver': 'SI=F',
        'oil': 'CL=F', 'crude': 'CL=F', 's&p 500': '^GSPC', 'sp500': '^GSPC', 's&p': '^GSPC',
        'nasdaq': '^IXIC', 'dow': '^DJI', 'tesla': 'TSLA', 'tsla': 'TSLA',
        'apple': 'AAPL', 'aapl': 'AAPL', 'nvidia': 'NVDA', 'nvda': 'NVDA'
    }
    for key, ticker in tickers.items():
        if key in q:
            return ticker
    return None

@dataclass
class Position:
    market: PolymarketMarket
    quantity: float
    is_yes: bool
    entry_price: float = 0.0
    
    @property
    def notional(self) -> float:
        price = self.market.yes_price if self.is_yes else self.market.no_price
        return abs(self.quantity) * price
    
    @property
    def pnl(self) -> float:
        current = self.market.yes_price if self.is_yes else self.market.no_price
        return self.quantity * (current - self.entry_price)

class Portfolio:
    BRIDGE_THRESHOLD_DAYS = 7
    
    def __init__(self, model: str = 'auto'):
        self.positions: List[Position] = []
        self.model = model
        self._bs = BlackScholesDigital()
        self._bridge = BrownianBridge()
    
    def add_position(self, position: Position):
        self.positions.append(position)
    
    def _time_to_expiry(self, market: PolymarketMarket) -> float:
        return max((market.end_date - datetime.now(timezone.utc)).total_seconds() / (365.25 * 24 * 3600), 1/365)
    
    def _select_model(self, market: PolymarketMarket) -> str:
        if self.model != 'auto':
            return self.model
        return 'bridge' if self._time_to_expiry(market) * 365 > self.BRIDGE_THRESHOLD_DAYS else 'bs'
    
    def calculate_greeks(self, position: Position, sigma: float = 0.5, 
                        underlying_price: Optional[float] = None, use_barrier_model: bool = True) -> Greeks:
        tau = self._time_to_expiry(position.market)
        if use_barrier_model and underlying_price:
            greeks = BarrierOptionGreeks().greeks_from_market(position.market, underlying_price, sigma, position.is_yes)
            if greeks:
                return greeks * position.quantity
        p = np.clip(position.market.yes_price if position.is_yes else position.market.no_price, 0.01, 0.99)
        model = self._select_model(position.market)
        greeks = self._bridge.greeks(p, sigma, tau, max(30/365, tau), position.is_yes) if model == 'bridge' else \
                 self._bs.greeks(p, sigma, tau, position.is_yes)
        return greeks * position.quantity
    
    def aggregate_greeks(self, volatilities: Optional[Dict[str, float]] = None,
                        underlying_prices: Optional[Dict[str, float]] = None, use_barrier_model: bool = True) -> Greeks:
        total = Greeks(0, 0, 0, 0, 0)
        for pos in self.positions:
            sigma = (volatilities or {}).get(pos.market.id, 0.5)
            ticker = infer_ticker_from_question(pos.market.question)
            price = (underlying_prices or {}).get(ticker) if ticker else None
            total = total + self.calculate_greeks(pos, sigma, price, use_barrier_model)
        return total

def fetch_live_markets(limit: int = 10, search: str = None) -> List[PolymarketMarket]:
    import requests
    headers = {'Accept': 'application/json', 'Accept-Encoding': 'gzip, deflate',
               'User-Agent': 'PolymarketGreeksCalculator/1.0'}
    
    if search:
        url, params = "https://gamma-api.polymarket.com/events", {
            "active": "true", "closed": "false", "limit": 100, "order": "volume24hr", "ascending": "false"}
    else:
        url, params = "https://gamma-api.polymarket.com/markets", {
            "active": "true", "closed": "false", "limit": limit, "order": "volume24hr", "ascending": "false"}
    
    try:
        data = requests.get(url, params=params, headers=headers, timeout=15).json()
        markets = []
        
        if search:
            for event in data:
                title = event.get('title', '') or event.get('question', '') or ''
                if search.lower() not in title.lower() and search.lower() not in event.get('description', '').lower():
                    continue
                for m in event.get('markets', []):
                    try:
                        markets.append(PolymarketMarket.from_api_response(m))
                    except:
                        continue
                if len(markets) >= limit:
                    break
        else:
            for m in data:
                try:
                    markets.append(PolymarketMarket.from_api_response(m))
                except:
                    continue
        return markets[:limit]
    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []

def fetch_price_history(token_id: str, interval: str = "1w") -> List[dict]:
    import requests
    try:
        data = requests.get("https://clob.polymarket.com/prices-history", 
                          params={"market": str(token_id).strip('[]"\''), "interval": interval, "fidelity": 60},
                          headers={'Accept': 'application/json', 'Accept-Encoding': 'gzip, deflate',
                                 'User-Agent': 'PolymarketGreeksCalculator/1.0'}, timeout=10).json()
        return data.get("history", [])
    except:
        return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Greeks Calculator - Live Data")
    parser.add_argument("-n", "--num-markets", type=int, default=10, help="Number of markets to fetch")
    parser.add_argument("-s", "--search", type=str, default=None, help="Search term to filter markets")
    args = parser.parse_args()
    
    print("=" * 70)
    print("POLYMARKET GREEKS CALCULATOR - LIVE DATA")
    print("=" * 70)
    
    if args.search:
        print(f"\nSearching for markets matching: '{args.search}'...")
    else:
        print(f"\nFetching top {args.num_markets} markets by volume...")
    
    markets = fetch_live_markets(limit=args.num_markets, search=args.search)
    
    if not markets:
        print("No markets found.")
        return
    
    print(f"Found {len(markets)} markets\n")
    
    # Fetch Yahoo Finance data
    try:
        import yfinance
        print("Fetching underlying asset data from Yahoo Finance...")
        underlying_vols, underlying_prices, seen = {}, {}, set()
        
        for m in markets:
            ticker = infer_ticker_from_question(m.question)
            if ticker and ticker not in seen:
                seen.add(ticker)
                print(f"  {ticker}: ", end='')
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
    except ImportError:
        print("yfinance not installed - using probability models only")
        underlying_vols, underlying_prices = {}, {}
    
    # Display markets
    print(f"\n{'#':<3} {'Market':<40} {'YES':>7} {'Days':>5} {'Model':>7} {'Delta':>8} {'Gamma':>10}")
    print("-" * 85)
    
    valid_markets = []
    for m in markets:
        tau_days = (m.end_date - datetime.now(timezone.utc)).total_seconds() / 86400
        if tau_days <= 0:
            continue
        valid_markets.append(m)
        
        sigma = underlying_vols.get(m.id, 0.8)
        ticker = infer_ticker_from_question(m.question)
        price = underlying_prices.get(ticker) if ticker else None
        strike = extract_strike_from_question(m.question)
        
        if price and strike:
            greeks = BarrierOptionGreeks().greeks_from_market(m, price, sigma, True)
            model = 'Barrier'
        else:
            model = 'Bridge' if tau_days > 7 else 'BS'
            greeks = (BrownianBridge().greeks(m.yes_price, sigma, tau_days/365, 30/365, True) if model == 'Bridge'
                     else BlackScholesDigital().greeks(m.yes_price, sigma, tau_days/365, True))
        
        q = m.question[:37] + "..." if len(m.question) > 40 else m.question
        print(f"{len(valid_markets):<3} {q:<40} {m.yes_price*100:>6.1f}% {tau_days:>4.0f}d {model:>7} "
              f"{greeks.delta:>+8.4f} {greeks.gamma:>+10.6f}")
    
    # Detailed breakdown
    print("\n" + "=" * 70)
    print("DETAILED GREEKS")
    print("=" * 70)
    
    for m in valid_markets[:5]:  # Show first 5 in detail
        tau = (m.end_date - datetime.now(timezone.utc)).total_seconds() / (365.25 * 24 * 3600)
        sigma = underlying_vols.get(m.id, 0.8)
        ticker = infer_ticker_from_question(m.question)
        price = underlying_prices.get(ticker) if ticker else None
        strike = extract_strike_from_question(m.question)
        
        print(f"\n📊 {m.question}")
        print(f"   YES: {m.yes_price*100:.1f}% | Expires in {tau*365:.0f} days")
        
        if price and strike:
            g = BarrierOptionGreeks().greeks_from_market(m, price, sigma, True)
            print(f"   Strike: ${strike:,.0f} | Current: ${price:,.2f} | Vol: {sigma*100:.0f}%")
            print(f"   Delta: {g.delta:.6f} | Gamma: {g.gamma:.8f} | Vega: {g.vega:.6f} | Theta: {g.theta:.8f}")
        else:
            bs_g = BlackScholesDigital().greeks(m.yes_price, sigma, tau, True)
            br_g = BrownianBridge().greeks(m.yes_price, sigma, tau, 30/365, True)
            print(f"   BS:     Δ={bs_g.delta:.4f}, Γ={bs_g.gamma:.4f}, V={bs_g.vega:.4f}, Θ={bs_g.theta:.6f}")
            print(f"   Bridge: Δ={br_g.delta:.4f}, Γ={br_g.gamma:.4f}, V={br_g.vega:.4f}, Θ={br_g.theta:.6f}")

if __name__ == "__main__":
    main()