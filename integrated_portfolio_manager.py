"""
Integrated Portfolio Manager with Correlation-Adjusted Greeks

Combines:
1. Polymarket Greeks Calculator (from polymarket_greeks.py)
2. Correlation Analysis (from correlation_analysis.py)
3. Greek Optimization Agent (from simplified_greek_optimization.py)

Allows you to:
- Load a Polymarket portfolio from JSON
- Add/remove positions
- Adjust Greeks based on cross-asset correlations
- Re-optimize portfolio to maintain target Greeks
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Import from the Greek optimization agent
import sys
sys.path.insert(0, '/home/claude')

# We'll inline the necessary components since we can't directly import

# =============================================================================
# ASSET MAPPING - Maps Polymarket assets to Yahoo Finance tickers
# =============================================================================

ASSET_TO_TICKER = {
    'bitcoin': 'BTC-USD',
    'ethereum': 'ETH-USD',
    'solana': 'SOL-USD',
    'gold': 'GC=F',
    'silver': 'SI=F',
    'oil': 'CL=F',
    'usd_index': 'DX-Y.NYB',
}

# Reverse mapping
TICKER_TO_ASSET = {v: k for k, v in ASSET_TO_TICKER.items()}

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PolymarketPosition:
    """A position in a Polymarket market"""
    market_id: str
    question: str
    asset_category: str  # bitcoin, gold, silver, oil, etc.
    quantity: float  # number of shares (positive = YES, negative = NO)
    yes_price: float
    no_price: float
    expiry_days: int
    
    # Greeks (per share)
    delta: float
    gamma: float
    vega: float
    theta: float
    
    # Optional metadata
    liquidity: float = 0.0
    volume: float = 0.0
    
    @property
    def position_value(self) -> float:
        """Total value of position"""
        return abs(self.quantity) * self.yes_price
    
    @property
    def greek_contributions(self) -> Dict[str, float]:
        """Greeks contributed by this position"""
        return {
            'delta': self.delta * self.quantity,
            'gamma': self.gamma * self.quantity,
            'vega': self.vega * self.quantity,
            'theta': self.theta * self.quantity
        }

@dataclass
class Portfolio:
    """Portfolio of Polymarket positions"""
    positions: List[PolymarketPosition]
    correlation_matrix: Optional[pd.DataFrame] = None
    
    def add_position(self, position: PolymarketPosition):
        """Add a new position"""
        self.positions.append(position)
    
    def remove_position(self, market_id: str):
        """Remove a position by market ID"""
        self.positions = [p for p in self.positions if p.market_id != market_id]
    
    def get_position(self, market_id: str) -> Optional[PolymarketPosition]:
        """Get position by market ID"""
        for pos in self.positions:
            if pos.market_id == market_id:
                return pos
        return None
    
    def calculate_greeks(self, use_correlations: bool = False) -> Dict[str, float]:
        """
        Calculate aggregate portfolio Greeks.
        
        Args:
            use_correlations: If True, adjust Greeks based on cross-correlations
        """
        if not self.positions:
            return {
                'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0,
                'total_value': 0.0, 'num_positions': 0
            }
        
        if use_correlations and self.correlation_matrix is not None:
            return self._calculate_correlated_greeks()
        else:
            return self._calculate_simple_greeks()
    
    def _calculate_simple_greeks(self) -> Dict[str, float]:
        """Simple summation of Greeks (no correlation adjustment)"""
        greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
        total_value = 0.0
        
        for pos in self.positions:
            contrib = pos.greek_contributions
            for greek in greeks:
                greeks[greek] += contrib[greek]
            total_value += pos.position_value
        
        greeks['total_value'] = total_value
        greeks['num_positions'] = len(self.positions)
        return greeks
    
    def _calculate_correlated_greeks(self) -> Dict[str, float]:
        """
        Calculate Greeks with correlation adjustments.
        
        Key insight: When you hold correlated assets, their Greeks don't simply add.
        The effective Greek exposure depends on how correlated the underlying assets are.
        
        Formula for correlated Greeks:
        Effective_Greek = Σᵢ Σⱼ (Greekᵢ * Greekⱼ * ρᵢⱼ)^0.5
        
        where ρᵢⱼ is the correlation between assets i and j
        """
        if self.correlation_matrix is None:
            return self._calculate_simple_greeks()
        
        # Group positions by asset category
        asset_groups = {}
        for pos in self.positions:
            if pos.asset_category not in asset_groups:
                asset_groups[pos.asset_category] = []
            asset_groups[pos.asset_category].append(pos)
        
        # Calculate correlated Greeks
        greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
        total_value = 0.0
        
        assets = list(asset_groups.keys())
        
        for greek_name in ['delta', 'gamma', 'vega', 'theta']:
            # Create Greek contribution matrix
            greek_matrix = np.zeros((len(assets), len(assets)))
            
            for i, asset_i in enumerate(assets):
                # Sum Greeks for all positions in asset_i
                greek_i = sum(getattr(pos, greek_name) * pos.quantity 
                             for pos in asset_groups[asset_i])
                
                for j, asset_j in enumerate(assets):
                    # Sum Greeks for all positions in asset_j
                    greek_j = sum(getattr(pos, greek_name) * pos.quantity 
                                 for pos in asset_groups[asset_j])
                    
                    # Get correlation between asset_i and asset_j
                    corr = self._get_correlation(asset_i, asset_j)
                    
                    # Correlation-adjusted Greek contribution
                    # Sign matters: if both Greeks same sign, correlation amplifies
                    # If opposite signs, correlation reduces (hedging effect)
                    greek_matrix[i, j] = greek_i * greek_j * corr
            
            # Total correlated Greek (with sign preservation)
            total_greek = np.sum(greek_matrix)
            greeks[greek_name] = total_greek
        
        # Calculate total value
        for pos in self.positions:
            total_value += pos.position_value
        
        greeks['total_value'] = total_value
        greeks['num_positions'] = len(self.positions)
        greeks['correlation_adjusted'] = True
        
        return greeks
    
    def _get_correlation(self, asset1: str, asset2: str) -> float:
        """Get correlation between two assets from correlation matrix"""
        if self.correlation_matrix is None:
            return 1.0 if asset1 == asset2 else 0.0
        
        # Get tickers
        ticker1 = ASSET_TO_TICKER.get(asset1)
        ticker2 = ASSET_TO_TICKER.get(asset2)
        
        if ticker1 is None or ticker2 is None:
            return 1.0 if asset1 == asset2 else 0.0
        
        # Try to find in correlation matrix
        # Correlation matrix has format: "Asset Name (TICKER)"
        matching_cols1 = [col for col in self.correlation_matrix.columns 
                         if ticker1 in col]
        matching_rows2 = [row for row in self.correlation_matrix.index 
                         if ticker2 in row]
        
        if matching_cols1 and matching_rows2:
            try:
                corr = self.correlation_matrix.loc[matching_rows2[0], matching_cols1[0]]
                return float(corr) if not pd.isna(corr) else (1.0 if asset1 == asset2 else 0.0)
            except:
                pass
        
        # Fallback
        return 1.0 if asset1 == asset2 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'positions': [asdict(p) for p in self.positions],
            'greeks_simple': self._calculate_simple_greeks(),
            'greeks_correlated': self._calculate_correlated_greeks() if self.correlation_matrix is not None else None
        }

# =============================================================================
# PORTFOLIO MANAGER
# =============================================================================

class IntegratedPortfolioManager:
    """
    Manages a Polymarket portfolio with correlation-adjusted Greeks
    """
    
    def __init__(
        self,
        correlation_matrix_path: Optional[str] = None,
        portfolio: Optional[Portfolio] = None
    ):
        """
        Initialize the portfolio manager.
        
        Args:
            correlation_matrix_path: Path to correlation matrix CSV
            portfolio: Existing portfolio (optional)
        """
        self.portfolio = portfolio or Portfolio(positions=[])
        
        # Load correlation matrix if provided
        if correlation_matrix_path:
            self.load_correlation_matrix(correlation_matrix_path)
        
    def load_correlation_matrix(self, path: str):
        """Load correlation matrix from CSV"""
        try:
            self.portfolio.correlation_matrix = pd.read_csv(path, index_col=0)
            print(f"✅ Loaded correlation matrix: {self.portfolio.correlation_matrix.shape}")
        except Exception as e:
            print(f"❌ Failed to load correlation matrix: {e}")
            self.portfolio.correlation_matrix = None
    
    def load_portfolio_from_json(
        self,
        json_path: str,
        markets_data: List[Dict]
    ):
        """
        Load portfolio from Polymarket JSON and markets data.
        
        Args:
            json_path: Path to JSON file with markets
            markets_data: List of market dictionaries with Greeks from polymarket_greeks.py
        """
        # Load JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Parse markets
        if 'events' in data:
            for event in data['events']:
                for market in event.get('markets', []):
                    self._add_market_from_json(market, event, markets_data)
        elif isinstance(data, list):
            for market in data:
                self._add_market_from_json(market, None, markets_data)
        
        print(f"✅ Loaded {len(self.portfolio.positions)} positions from {json_path}")
    
    def _add_market_from_json(
        self,
        market: Dict,
        event: Optional[Dict],
        markets_data: List[Dict]
    ):
        """Helper to add a single market from JSON"""
        market_id = market.get('id', '')
        
        # Find matching Greeks data
        greeks_data = None
        for m in markets_data:
            if m.get('market_id') == market_id or m.get('question') == market.get('question'):
                greeks_data = m
                break
        
        if greeks_data is None:
            print(f"⚠️ No Greeks data found for market {market_id}, skipping")
            return
        
        # Determine asset category
        asset_category = 'other'
        if event and 'relatedCommodity' in event:
            commodity = event['relatedCommodity'].lower().strip()
            if commodity == 'usd index':
                asset_category = 'usd_index'
            elif commodity in ASSET_TO_TICKER:
                asset_category = commodity
        
        # Parse prices
        outcome_prices = json.loads(market.get('outcomePrices', '["0.5", "0.5"]'))
        yes_price = float(outcome_prices[0])
        no_price = float(outcome_prices[1]) if len(outcome_prices) > 1 else 1 - yes_price
        
        # Calculate expiry days
        end_date_str = market.get('endDate') or market.get('endDateIso')
        if end_date_str:
            try:
                if end_date_str.endswith('Z'):
                    end_date_str = end_date_str[:-1] + '+00:00'
                end_date = datetime.fromisoformat(end_date_str)
                expiry_days = max(0, (end_date - datetime.now(timezone.utc)).days)
            except:
                expiry_days = 30
        else:
            expiry_days = 30
        
        # Create position (initially zero quantity)
        position = PolymarketPosition(
            market_id=market_id,
            question=market.get('question', ''),
            asset_category=asset_category,
            quantity=0.0,  # Will be set when adding position
            yes_price=yes_price,
            no_price=no_price,
            expiry_days=expiry_days,
            delta=greeks_data.get('delta', 0.0),
            gamma=greeks_data.get('gamma', 0.0),
            vega=greeks_data.get('vega', 0.0),
            theta=greeks_data.get('theta', 0.0),
            liquidity=float(market.get('liquidityNum', 0)),
            volume=float(market.get('volumeNum', 0))
        )
        
        self.portfolio.add_position(position)
    
    def add_position_to_portfolio(
        self,
        market_id: str,
        quantity: float,
        use_correlations: bool = True,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Add a position to the portfolio and recalculate Greeks.
        
        Args:
            market_id: Market ID to trade
            quantity: Number of shares (positive = buy YES, negative = buy NO)
            use_correlations: Whether to use correlation-adjusted Greeks
            verbose: Print detailed output
        
        Returns:
            Dictionary with before/after Greeks and impact analysis
        """
        # Find the position
        position = self.portfolio.get_position(market_id)
        if position is None:
            raise ValueError(f"Market {market_id} not found in portfolio")
        
        # Calculate Greeks before
        greeks_before = self.portfolio.calculate_greeks(use_correlations)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"ADDING POSITION: {quantity:+.2f} shares of {position.question[:50]}")
            print(f"{'='*80}")
            print(f"Asset Category: {position.asset_category}")
            print(f"YES Price: ${position.yes_price:.3f} | NO Price: ${position.no_price:.3f}")
            print(f"Position Value: ${abs(quantity) * position.yes_price:,.2f}")
            print(f"\nPer-Share Greeks:")
            print(f"  Delta: {position.delta:+.6f}")
            print(f"  Gamma: {position.gamma:+.6f}")
            print(f"  Vega:  {position.vega:+.6f}")
            print(f"  Theta: {position.theta:+.6f}")
        
        # Update quantity
        old_quantity = position.quantity
        position.quantity += quantity
        
        # Calculate Greeks after
        greeks_after = self.portfolio.calculate_greeks(use_correlations)
        
        # Calculate impact
        impact = {
            'market_id': market_id,
            'question': position.question,
            'asset_category': position.asset_category,
            'quantity_change': quantity,
            'old_quantity': old_quantity,
            'new_quantity': position.quantity,
            'position_value': abs(quantity) * position.yes_price,
            'greeks_before': greeks_before,
            'greeks_after': greeks_after,
            'greeks_delta': {
                k: greeks_after.get(k, 0) - greeks_before.get(k, 0)
                for k in ['delta', 'gamma', 'vega', 'theta']
            },
            'correlation_adjusted': use_correlations and self.portfolio.correlation_matrix is not None
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"PORTFOLIO IMPACT")
            print(f"{'='*80}")
            
            adjustment_label = " (Correlation-Adjusted)" if impact['correlation_adjusted'] else " (Simple Sum)"
            print(f"\nGreek Changes{adjustment_label}:")
            print(f"{'Greek':<10} {'Before':<12} {'After':<12} {'Change':<12}")
            print(f"{'-'*46}")
            
            for greek in ['delta', 'gamma', 'vega', 'theta']:
                before = greeks_before.get(greek, 0)
                after = greeks_after.get(greek, 0)
                change = impact['greeks_delta'][greek]
                
                # Color code
                if abs(change) < 0.001:
                    color = "\033[90m"  # Gray
                elif abs(change / (abs(before) + 1e-8)) < 0.05:
                    color = "\033[92m"  # Green (small change)
                elif abs(change / (abs(before) + 1e-8)) < 0.15:
                    color = "\033[93m"  # Yellow (medium change)
                else:
                    color = "\033[91m"  # Red (large change)
                
                print(f"{greek:<10} {before:+11.6f} {after:+11.6f} {color}{change:+11.6f}\033[0m")
            
            print(f"\nTotal Portfolio Value: ${greeks_after['total_value']:,.2f}")
            print(f"Number of Positions: {greeks_after['num_positions']}")
            
            if impact['correlation_adjusted']:
                print(f"\n💡 Greeks adjusted for cross-asset correlations")
                if position.asset_category in ASSET_TO_TICKER:
                    ticker = ASSET_TO_TICKER[position.asset_category]
                    print(f"   {position.asset_category.title()} ({ticker}) correlations considered")
        
        return impact
    
    def remove_position_from_portfolio(
        self,
        market_id: str,
        quantity: Optional[float] = None,
        use_correlations: bool = True,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Remove a position (or reduce quantity) from portfolio.
        
        Args:
            market_id: Market ID
            quantity: Amount to remove (None = remove all)
            use_correlations: Whether to use correlation-adjusted Greeks
            verbose: Print output
        """
        position = self.portfolio.get_position(market_id)
        if position is None:
            raise ValueError(f"Market {market_id} not found in portfolio")
        
        if quantity is None:
            quantity = position.quantity
        
        return self.add_position_to_portfolio(
            market_id, -quantity, use_correlations, verbose
        )
    
    def get_portfolio_summary(self, use_correlations: bool = True) -> Dict:
        """Get comprehensive portfolio summary"""
        greeks = self.portfolio.calculate_greeks(use_correlations)
        
        # Group by asset category
        by_asset = {}
        for pos in self.portfolio.positions:
            if abs(pos.quantity) < 1e-8:
                continue
            
            if pos.asset_category not in by_asset:
                by_asset[pos.asset_category] = {
                    'positions': [],
                    'total_value': 0.0,
                    'greeks': {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
                }
            
            by_asset[pos.asset_category]['positions'].append(pos)
            by_asset[pos.asset_category]['total_value'] += pos.position_value
            
            for greek in ['delta', 'gamma', 'vega', 'theta']:
                by_asset[pos.asset_category]['greeks'][greek] += getattr(pos, greek) * pos.quantity
        
        return {
            'total_greeks': greeks,
            'by_asset': by_asset,
            'num_positions': greeks['num_positions'],
            'total_value': greeks['total_value'],
            'correlation_adjusted': use_correlations and self.portfolio.correlation_matrix is not None
        }
    
    def print_portfolio_summary(self, use_correlations: bool = True):
        """Print formatted portfolio summary"""
        summary = self.get_portfolio_summary(use_correlations)
        
        print(f"\n{'='*80}")
        print(f"PORTFOLIO SUMMARY")
        if summary['correlation_adjusted']:
            print(f"(Greeks Adjusted for Cross-Asset Correlations)")
        print(f"{'='*80}")
        
        print(f"\nTotal Portfolio Value: ${summary['total_value']:,.2f}")
        print(f"Number of Positions: {summary['num_positions']}")
        
        print(f"\nAggregate Greeks:")
        for greek, value in summary['total_greeks'].items():
            if greek not in ['total_value', 'num_positions', 'correlation_adjusted']:
                print(f"  {greek.capitalize():<10}: {value:+.6f}")
        
        print(f"\n{'='*80}")
        print(f"BREAKDOWN BY ASSET CATEGORY")
        print(f"{'='*80}")
        
        for asset, data in summary['by_asset'].items():
            print(f"\n{asset.upper().replace('_', ' ')}")
            print(f"  Positions: {len(data['positions'])}")
            print(f"  Total Value: ${data['total_value']:,.2f}")
            print(f"  Greeks:")
            for greek, value in data['greeks'].items():
                print(f"    {greek.capitalize():<10}: {value:+.6f}")
            
            # Show top positions
            sorted_positions = sorted(data['positions'], 
                                     key=lambda p: abs(p.position_value), 
                                     reverse=True)
            print(f"  Top Positions:")
            for i, pos in enumerate(sorted_positions[:3], 1):
                action = "LONG" if pos.quantity > 0 else "SHORT"
                print(f"    {i}. {action} {abs(pos.quantity):.2f} shares: {pos.question[:40]}")
                print(f"       Value: ${pos.position_value:,.2f}")
    
    def export_to_json(self, filepath: str):
        """Export portfolio to JSON"""
        data = self.portfolio.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"✅ Portfolio exported to {filepath}")


# =============================================================================
# CLI AND TESTING
# =============================================================================

def main():
    """Demonstration of integrated portfolio manager"""
    print("=" * 80)
    print("INTEGRATED POLYMARKET PORTFOLIO MANAGER")
    print("Correlation-Adjusted Greeks System")
    print("=" * 80)
    
    # Example usage
    print("\n📋 EXAMPLE WORKFLOW:")
    print("\n1. Initialize manager with correlation matrix")
    print("2. Load Polymarket markets from JSON")
    print("3. Add positions and see correlation-adjusted Greek impacts")
    print("4. View portfolio summary")
    
    print("\n" + "=" * 80)
    print("To use this system:")
    print("=" * 80)
    print("""
# 1. Load correlation matrix
manager = IntegratedPortfolioManager(
    correlation_matrix_path='commodity_vs_core_assets_correlations.csv'
)

# 2. Load markets (with Greeks data from polymarket_greeks.py output)
markets_with_greeks = [
    {
        'market_id': '123',
        'question': 'Will gold hit $3000?',
        'delta': 0.5,
        'gamma': 0.1,
        'vega': 0.2,
        'theta': -0.02
    },
    # ... more markets
]

manager.load_portfolio_from_json(
    'markets.json',
    markets_with_greeks
)

# 3. Add a position
impact = manager.add_position_to_portfolio(
    market_id='123',
    quantity=100,  # Buy 100 YES shares
    use_correlations=True
)

# 4. View portfolio
manager.print_portfolio_summary(use_correlations=True)
    """)

if __name__ == "__main__":
    main()