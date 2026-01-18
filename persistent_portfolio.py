"""
Persistent Portfolio Tracker

Keeps track of:
- Current open positions
- Greek evolution over time
- Trade history
- Portfolio state saved to disk

All changes are permanent and tracked.
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Optional
import warnings
from io import StringIO

warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrated_portfolio_manager import (
    IntegratedPortfolioManager,
    PolymarketPosition,
    Portfolio
)

from polymarket_greeks import (
    load_markets_from_json,
    categorize_markets_by_asset,
    BrownianBridge
)

try:
    from greeksv2 import (
        SimplifiedGreekOptimizationAgent,
        create_sample_markets_from_inputs
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False


# =============================================================================
# PERSISTENT PORTFOLIO CLASS
# =============================================================================

class PersistentPortfolio:
    """
    Portfolio that saves state to disk and tracks all changes
    """
    
    def __init__(self, portfolio_file: str = "portfolio_state.json"):
        """
        Initialize persistent portfolio
        
        Args:
            portfolio_file: Path to save portfolio state
        """
        self.portfolio_file = portfolio_file
        self.history_file = portfolio_file.replace('.json', '_history.json')
        self.manager = None
        self.trade_history = []
        self.greek_history = []
        
        # Load existing portfolio if it exists
        if os.path.exists(portfolio_file):
            print(f"[LOAD] Loading existing portfolio from {portfolio_file}")
            self.load()
        else:
            print(f"[NEW] Creating new portfolio (will save to {portfolio_file})")
    
    def initialize_from_markets(
        self,
        markets_file: str,
        correlation_file: Optional[str] = None
    ):
        """
        Initialize portfolio from markets JSON
        
        Args:
            markets_file: Path to Polymarket markets JSON
            correlation_file: Optional correlation matrix CSV
        """
        print(f"\n[INIT] Initializing portfolio from {markets_file}...")
        
        # Load correlation matrix
        corr_matrix = None
        if correlation_file and os.path.exists(correlation_file):
            corr_matrix = pd.read_csv(correlation_file, index_col=0)
            print(f"   [OK] Loaded correlation matrix")
        
        # Load markets
        all_markets = load_markets_from_json(markets_file)
        print(f"   [OK] Loaded {len(all_markets)} markets")
        
        # Calculate Greeks (suppress warnings)
        print("   [CALC] Calculating Greeks...")
        markets_with_greeks = []
        
        stderr_backup = sys.stderr
        sys.stderr = StringIO()
        
        try:
            for market in all_markets:
                try:
                    tau_years = (market.end_date - datetime.now(timezone.utc)).total_seconds() / (365.25 * 24 * 3600)
                    if tau_years <= 0:
                        continue
                    
                    categorized = categorize_markets_by_asset([market])
                    asset = list(categorized.keys())[0] if categorized else 'other'
                    
                    bridge = BrownianBridge()
                    greeks = bridge.greeks(market.yes_price, 0.5, tau_years, 30/365, is_yes=True)
                    
                    markets_with_greeks.append(PolymarketPosition(
                        market_id=market.id,
                        question=market.question,
                        asset_category=asset,
                        quantity=0.0,
                        yes_price=market.yes_price,
                        no_price=market.no_price,
                        expiry_days=max(int(tau_years * 365), 1),
                        delta=greeks.delta,
                        gamma=greeks.gamma,
                        vega=greeks.vega,
                        theta=greeks.theta,
                        liquidity=market.liquidity,
                        volume=market.volume
                    ))
                except:
                    continue
        finally:
            sys.stderr = stderr_backup
        
        print(f"   [OK] Calculated Greeks for {len(markets_with_greeks)} markets")
        
        # Create manager
        portfolio = Portfolio(positions=markets_with_greeks, correlation_matrix=corr_matrix)
        self.manager = IntegratedPortfolioManager(portfolio=portfolio)
        
        # Save initial state
        self.save()
        
        print(f"[OK] Portfolio initialized and saved!")
    
    def add_position(
        self,
        market_id: str,
        quantity: float,
        use_correlations: bool = True,
        notes: str = ""
    ):
        """
        Add position and save state
        
        Args:
            market_id: Market ID
            quantity: Quantity to add
            use_correlations: Use correlation adjustments
            notes: Optional notes for this trade
        """
        if self.manager is None:
            raise RuntimeError("Portfolio not initialized. Call initialize_from_markets() first.")
        
        # Get Greeks before
        greeks_before = self.manager.portfolio.calculate_greeks(use_correlations)
        
        # Add position
        impact = self.manager.add_position_to_portfolio(
            market_id,
            quantity,
            use_correlations=use_correlations,
            verbose=True
        )
        
        greeks_after = impact['greeks_after']
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'market_id': market_id,
            'question': impact['question'],
            'asset_category': impact['asset_category'],
            'quantity': quantity,
            'position_value': impact['position_value'],
            'greeks_before': greeks_before,
            'greeks_after': greeks_after,
            'greek_changes': impact['greeks_delta'],
            'correlation_adjusted': use_correlations,
            'notes': notes
        }
        
        self.trade_history.append(trade_record)
        
        # Record Greeks snapshot
        self.greek_history.append({
            'timestamp': datetime.now().isoformat(),
            'greeks': greeks_after,
            'num_positions': greeks_after.get('num_positions', 0),
            'total_value': greeks_after.get('total_value', 0)
        })
        
        # Save to disk
        self.save()
        
        print(f"\n[OK] Position added and saved to {self.portfolio_file}")
    
    def get_current_greeks(self, use_correlations: bool = True) -> Dict:
        """Get current portfolio Greeks"""
        if self.manager is None:
            raise RuntimeError("Portfolio not initialized")
        
        return self.manager.portfolio.calculate_greeks(use_correlations)
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions (non-zero quantity)"""
        if self.manager is None:
            return []
        
        open_positions = []
        for pos in self.manager.portfolio.positions:
            if abs(pos.quantity) > 1e-8:
                open_positions.append({
                    'market_id': pos.market_id,
                    'question': pos.question,
                    'asset_category': pos.asset_category,
                    'quantity': pos.quantity,
                    'yes_price': pos.yes_price,
                    'position_value': abs(pos.quantity) * pos.yes_price,
                    'delta': pos.delta,
                    'gamma': pos.gamma,
                    'vega': pos.vega,
                    'theta': pos.theta,
                    'greek_contributions': {
                        'delta': pos.delta * pos.quantity,
                        'gamma': pos.gamma * pos.quantity,
                        'vega': pos.vega * pos.quantity,
                        'theta': pos.theta * pos.quantity
                    }
                })
        
        # Sort by position value
        open_positions.sort(key=lambda x: abs(x['position_value']), reverse=True)
        
        return open_positions
    
    def show_summary(self, use_correlations: bool = True):
        """Display portfolio summary"""
        if self.manager is None:
            print("[ERROR] Portfolio not initialized")
            return
        
        print("\n" + "=" * 80)
        print("PORTFOLIO SUMMARY")
        print("=" * 80)
        
        # Current Greeks
        greeks = self.get_current_greeks(use_correlations)
        
        print(f"\n[GREEKS] Current Greeks{' (Correlation-Adjusted)' if use_correlations else ''}:")
        print(f"   Delta:  {greeks.get('delta', 0):+.6f}")
        print(f"   Gamma:  {greeks.get('gamma', 0):+.6f}")
        print(f"   Vega:   {greeks.get('vega', 0):+.6f}")
        print(f"   Theta:  {greeks.get('theta', 0):+.6f}")
        
        print(f"\n[METRICS] Portfolio Metrics:")
        print(f"   Total Value: ${greeks.get('total_value', 0):,.2f}")
        print(f"   Open Positions: {greeks.get('num_positions', 0)}")
        
        # Open positions
        open_pos = self.get_open_positions()
        
        if open_pos:
            print(f"\n[POSITIONS] Open Positions (Top 10):")
            print("-" * 80)
            
            for i, pos in enumerate(open_pos[:10], 1):
                action = "LONG" if pos['quantity'] > 0 else "SHORT"
                print(f"\n{i}. {action} {abs(pos['quantity']):.2f} shares")
                print(f"   {pos['question'][:65]}")
                print(f"   {pos['asset_category']} | Value: ${pos['position_value']:,.2f}")
                print(f"   Greek Contrib: "
                      f"Δ:{pos['greek_contributions']['delta']:+.4f} "
                      f"Γ:{pos['greek_contributions']['gamma']:+.4f} "
                      f"V:{pos['greek_contributions']['vega']:+.4f} "
                      f"θ:{pos['greek_contributions']['theta']:+.4f}")
        else:
            print(f"\n[POSITIONS] No open positions")
        
        # Recent trades
        if self.trade_history:
            print(f"\n[TRADES] Recent Trades (Last 5):")
            print("-" * 80)
            
            for trade in self.trade_history[-5:]:
                ts = datetime.fromisoformat(trade['timestamp']).strftime('%Y-%m-%d %H:%M')
                qty = trade['quantity']
                action = "LONG" if qty > 0 else "SHORT"
                
                print(f"\n{ts} - {action} {abs(qty):.0f} shares")
                print(f"   {trade['question'][:65]}")
                delta_change = trade['greek_changes']['delta']
                print(f"   Delta Impact: {delta_change:+.4f}")
    
    def show_greek_evolution(self):
        """Show how Greeks have evolved over time"""
        if not self.greek_history:
            print("[HISTORY] No Greek history yet")
            return
        
        print("\n" + "=" * 80)
        print("GREEK EVOLUTION")
        print("=" * 80)
        
        print(f"\n{'Timestamp':<20} {'Delta':<12} {'Gamma':<12} {'Vega':<12} {'Theta':<12} {'Value':<12}")
        print("-" * 92)
        
        for snapshot in self.greek_history[-10:]:  # Last 10
            ts = datetime.fromisoformat(snapshot['timestamp']).strftime('%Y-%m-%d %H:%M')
            greeks = snapshot['greeks']
            
            print(f"{ts:<20} "
                  f"{greeks.get('delta', 0):+11.4f} "
                  f"{greeks.get('gamma', 0):+11.4f} "
                  f"{greeks.get('vega', 0):+11.4f} "
                  f"{greeks.get('theta', 0):+11.4f} "
                  f"${snapshot['total_value']:>10,.2f}")
    
    def save(self):
        """Save portfolio state to disk"""
        if self.manager is None:
            return
        
        # Save portfolio positions
        positions_data = []
        for pos in self.manager.portfolio.positions:
            positions_data.append({
                'market_id': pos.market_id,
                'question': pos.question,
                'asset_category': pos.asset_category,
                'quantity': pos.quantity,
                'yes_price': pos.yes_price,
                'no_price': pos.no_price,
                'expiry_days': pos.expiry_days,
                'delta': pos.delta,
                'gamma': pos.gamma,
                'vega': pos.vega,
                'theta': pos.theta,
                'liquidity': pos.liquidity,
                'volume': pos.volume
            })
        
        state = {
            'last_updated': datetime.now().isoformat(),
            'positions': positions_data,
            'current_greeks': self.get_current_greeks(use_correlations=True),
        }
        
        with open(self.portfolio_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        # Save history
        history = {
            'trade_history': self.trade_history,
            'greek_history': self.greek_history
        }
        
        with open(self.history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)
    
    def load(self):
        """Load portfolio state from disk"""
        try:
            with open(self.portfolio_file, 'r') as f:
                state = json.load(f)
            
            # Recreate positions
            positions = []
            for pos_data in state.get('positions', []):
                try:
                    positions.append(PolymarketPosition(**pos_data))
                except Exception as e:
                    print(f"   [WARN] Failed to load position: {e}")
                    continue
            
            # Create portfolio
            portfolio = Portfolio(positions=positions)
            self.manager = IntegratedPortfolioManager(portfolio=portfolio)
            
            # Load history
            if os.path.exists(self.history_file):
                try:
                    with open(self.history_file, 'r') as f:
                        history = json.load(f)
                        self.trade_history = history.get('trade_history', [])
                        self.greek_history = history.get('greek_history', [])
                except Exception as e:
                    print(f"   [WARN] Failed to load history: {e}")
                    self.trade_history = []
                    self.greek_history = []
            
            print(f"   [OK] Loaded {len(positions)} positions")
            print(f"   [OK] Loaded {len(self.trade_history)} historical trades")
            
            # Show summary
            greeks = state.get('current_greeks', {})
            if 'last_updated' in state:
                print(f"\n   Portfolio as of {state['last_updated']}:")
            print(f"   Delta: {greeks.get('delta', 0):+.6f}")
            print(f"   Value: ${greeks.get('total_value', 0):,.2f}")
            print(f"   Positions: {greeks.get('num_positions', 0)}")
        except Exception as e:
            print(f"   [ERROR] Failed to load portfolio: {e}")
            print(f"   [INFO] Portfolio will need to be reinitialized")
            self.manager = None
            self.trade_history = []
            self.greek_history = []


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Persistent Portfolio Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:

Initialize new portfolio:
  python persistent_portfolio.py init -f markets.json --portfolio my_portfolio.json

Add position:
  python persistent_portfolio.py add --portfolio my_portfolio.json \\
    --market "1176066" --quantity 100

Show portfolio:
  python persistent_portfolio.py show --portfolio my_portfolio.json

Show Greek evolution:
  python persistent_portfolio.py history --portfolio my_portfolio.json

Optimize to target:
  python persistent_portfolio.py optimize --portfolio my_portfolio.json \\
    --target-delta 0.0 --target-gamma 0.5 --max-investment 5000
        """
    )
    
    parser.add_argument('command', choices=['init', 'add', 'show', 'history', 'optimize'])
    parser.add_argument('--portfolio', default='portfolio_state.json',
                       help='Portfolio file path')
    
    # Init command
    parser.add_argument('-f', '--file', help='Markets JSON (for init)')
    parser.add_argument('--correlation-file', help='Correlation CSV')
    
    # Add command
    parser.add_argument('--market', help='Market ID (for add)')
    parser.add_argument('--quantity', type=float, help='Quantity (for add)')
    parser.add_argument('--notes', default='', help='Trade notes')
    
    # Optimize command
    parser.add_argument('--target-delta', type=float)
    parser.add_argument('--target-gamma', type=float)
    parser.add_argument('--target-vega', type=float)
    parser.add_argument('--target-theta', type=float)
    parser.add_argument('--max-investment', type=float, default=10000)
    
    # Common
    parser.add_argument('--no-correlations', action='store_true')
    
    args = parser.parse_args()
    
    # Create portfolio object
    portfolio = PersistentPortfolio(args.portfolio)
    
    use_corr = not args.no_correlations
    
    # Execute command
    if args.command == 'init':
        if not args.file:
            print("[ERROR] --file required for init")
            sys.exit(1)
        
        portfolio.initialize_from_markets(args.file, args.correlation_file)
        portfolio.show_summary(use_corr)
    
    elif args.command == 'add':
        if not args.market or args.quantity is None:
            print("[ERROR] --market and --quantity required")
            sys.exit(1)
        
        portfolio.add_position(args.market, args.quantity, use_corr, args.notes)
        portfolio.show_summary(use_corr)
    
    elif args.command == 'show':
        portfolio.show_summary(use_corr)
    
    elif args.command == 'history':
        portfolio.show_greek_evolution()
    
    elif args.command == 'optimize':
        if not OPTIMIZER_AVAILABLE:
            print("[ERROR] Optimizer not available")
            sys.exit(1)
        
        targets = {}
        if args.target_delta is not None:
            targets['delta'] = args.target_delta
        if args.target_gamma is not None:
            targets['gamma'] = args.target_gamma
        if args.target_vega is not None:
            targets['vega'] = args.target_vega
        if args.target_theta is not None:
            targets['theta'] = args.target_theta
        
        if not targets:
            print("[ERROR] Need at least one target Greek")
            sys.exit(1)
        
        # Import optimization function
        from enhanced_portfolio_manager import optimize_to_target_greeks
        
        result = optimize_to_target_greeks(
            portfolio.manager,
            targets,
            args.max_investment,
            use_corr
        )
        
        if result['success'] and result.get('recommendations'):
            print("\n" + "=" * 80)
            print("Execute these trades? (y/n)")
            response = input().strip().lower()
            
            if response == 'y':
                for rec in result['recommendations']:
                    # Find market by title
                    market_id = None
                    for pos in portfolio.manager.portfolio.positions:
                        if pos.question == rec['market_title']:
                            market_id = pos.market_id
                            break
                    
                    if market_id:
                        portfolio.add_position(
                            market_id,
                            rec['quantity'],
                            use_corr,
                            notes=f"Optimization trade to reach target Greeks"
                        )
                
                print("\n[OK] All trades executed and saved!")
                portfolio.show_summary(use_corr)


if __name__ == "__main__":
    main()