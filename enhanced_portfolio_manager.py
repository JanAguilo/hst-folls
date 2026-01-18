"""
Enhanced Portfolio Manager with Two Modes:

MODE 1: Position Impact Analysis
- Input: market_id + quantity to add
- Output: Visualization of Greek impact BEFORE adding

MODE 2: Greek Targeting with Optimization
- Input: current portfolio + available markets + target Greeks
- Output: Optimal hedging strategy using your friend's optimizer
"""

import json
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import warnings
from io import StringIO

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrated_portfolio_manager import (
    IntegratedPortfolioManager,
    PolymarketPosition,
    Portfolio
)

from polymarket_greeks_fixed import (
    load_markets_from_json,
    categorize_markets_by_asset,
    BrownianBridge
)

# Import optimizer from your friend's code
try:
    from simplified_greek_optimization import (
        SimplifiedGreekOptimizationAgent,
        create_sample_markets_from_inputs
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    print("⚠️  simplified_greek_optimization.py not found. Mode 2 disabled.")
    OPTIMIZER_AVAILABLE = False


# ============================================================================
# MODE 1: POSITION IMPACT ANALYSIS (BEFORE ADDING)
# ============================================================================

def analyze_position_impact(
    manager: IntegratedPortfolioManager,
    market_id: str,
    quantity: float,
    use_correlations: bool = True
):
    """
    MODE 1: Visualize Greek impact BEFORE adding position
    
    Args:
        manager: Portfolio manager
        market_id: Market to add
        quantity: Number of shares (+ YES, - NO)
        use_correlations: Use correlations
    
    Returns:
        dict with impact analysis
    """
    print("\n" + "=" * 80)
    print("MODE 1: POSITION IMPACT ANALYSIS")
    print("=" * 80)
    
    position = manager.portfolio.get_position(market_id)
    if position is None:
        raise ValueError(f"Market {market_id} not found")
    
    # Current Greeks
    greeks_before = manager.portfolio.calculate_greeks(use_correlations)
    
    print(f"\n📊 Market: {position.question}")
    print(f"   Asset: {position.asset_category}")
    print(f"   YES: ${position.yes_price:.3f} | NO: ${position.no_price:.3f}")
    print(f"   Quantity: {quantity:+.2f} shares")
    print(f"   Value: ${abs(quantity) * position.yes_price:,.2f}")
    
    print(f"\n📈 Per-Share Greeks:")
    print(f"   Δ: {position.delta:+.6f}")
    print(f"   Γ: {position.gamma:+.6f}")
    print(f"   V: {position.vega:+.6f}")
    print(f"   θ: {position.theta:+.6f}")
    
    # SIMULATE (don't actually add yet)
    original_qty = position.quantity
    position.quantity += quantity
    greeks_after = manager.portfolio.calculate_greeks(use_correlations)
    position.quantity = original_qty  # Restore
    
    # Show impact
    print(f"\n" + "=" * 80)
    print("GREEK IMPACT" + (" (Correlation-Adjusted)" if use_correlations else ""))
    print("=" * 80)
    
    print(f"\n{'Greek':<10} {'Before':<15} {'After':<15} {'Change':<15} {'% Chg':<10}")
    print("-" * 65)
    
    for greek in ['delta', 'gamma', 'vega', 'theta']:
        before = greeks_before.get(greek, 0)
        after = greeks_after.get(greek, 0)
        change = after - before
        
        if abs(before) > 1e-8:
            pct = (change / abs(before)) * 100
        else:
            pct = 0
        
        if abs(pct) < 5:
            color = "\033[92m"
        elif abs(pct) < 20:
            color = "\033[93m"
        else:
            color = "\033[91m"
        
        pct_str = f"{pct:+.1f}%" if abs(pct) < 999 else "N/A"
        print(f"{greek:<10} {before:+14.6f} {after:+14.6f} {color}{change:+14.6f}\033[0m {pct_str}")
    
    # Bar chart
    print(f"\n{'Greek':<10} Impact Bar")
    print("-" * 40)
    
    changes = {g: greeks_after.get(g, 0) - greeks_before.get(g, 0) for g in ['delta', 'gamma', 'vega', 'theta']}
    max_change = max(abs(v) for v in changes.values())
    
    if max_change > 0:
        for greek, change in changes.items():
            bar_len = int((abs(change) / max_change) * 30)
            bar = "█" * bar_len
            sign = "+" if change > 0 else "-"
            print(f"{greek:<10} {sign}{bar} {change:+.6f}")
    
    print(f"\n💡 Do you want to execute this trade? (greeks shown above)")
    
    return {
        'greeks_before': greeks_before,
        'greeks_after': greeks_after,
        'changes': changes
    }


# ============================================================================
# MODE 2: GREEK TARGETING WITH OPTIMIZATION
# ============================================================================

def optimize_to_target_greeks(
    manager: IntegratedPortfolioManager,
    target_greeks: dict,
    max_investment: float = 10000.0,
    use_correlations: bool = True
):
    """
    MODE 2: Find trades to reach target Greeks
    
    Args:
        manager: Portfolio manager
        target_greeks: {'delta': X, 'gamma': Y, ...}
        max_investment: Max $ for new trades
        use_correlations: Use correlations
    
    Returns:
        dict with recommendations
    """
    if not OPTIMIZER_AVAILABLE:
        raise RuntimeError("Optimizer not available")
    
    print("\n" + "=" * 80)
    print("MODE 2: GREEK TARGETING WITH OPTIMIZATION")
    print("=" * 80)
    
    # Current Greeks
    current = manager.portfolio.calculate_greeks(use_correlations)
    
    print(f"\n📊 Current Greeks{' (Corr-Adj)' if use_correlations else ''}:")
    for g in ['delta', 'gamma', 'vega', 'theta']:
        print(f"   {g.capitalize():<10}: {current.get(g, 0):+.6f}")
    
    print(f"\n🎯 Target Greeks:")
    for g, v in target_greeks.items():
        print(f"   {g.capitalize():<10}: {v:+.6f}")
    
    # Gap
    gap = {g: target_greeks.get(g, current.get(g, 0)) - current.get(g, 0) for g in ['delta', 'gamma', 'vega', 'theta']}
    
    print(f"\n📉 Gap:")
    for g, v in gap.items():
        if abs(v) > 1e-8:
            print(f"   {g.capitalize():<10}: {v:+.6f}")
    
    if max(abs(v) for v in gap.values()) < 0.01:
        print("\n✅ Already at target!")
        return {'success': True, 'already_at_target': True}
    
    # Prepare for optimizer
    print(f"\n⚙️  Preparing markets for optimizer...")
    
    opt_markets = []
    for pos in manager.portfolio.positions:
        if pos.expiry_days > 0:
            opt_markets.append({
                'title': pos.question,
                'yes_price': pos.yes_price,
                'no_price': pos.no_price,
                'expiry_days': pos.expiry_days,
                'delta': pos.delta,
                'gamma': pos.gamma,
                'vega': pos.vega,
                'theta': pos.theta,
                'category': pos.asset_category,
                'liquidity': pos.liquidity,
                'volume_24h': pos.volume
            })
    
    print(f"   ✅ {len(opt_markets)} markets ready")
    
    # Run optimizer
    print(f"\n🔧 Running optimization...")
    
    market_objs = create_sample_markets_from_inputs(opt_markets)
    agent = SimplifiedGreekOptimizationAgent(markets=market_objs, seed=42)
    
    result = agent.optimize(
        target_greeks=gap,  # Optimize for GAP
        max_total_investment=max_investment,
        max_position_per_market=500,
        min_position_size=1.0,
        deterministic=True
    )
    
    print(f"\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    
    print(f"\nStatus: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    print(f"Time: {result.optimization_time_ms:.1f} ms")
    print(f"Investment: ${result.total_investment:,.2f}")
    print(f"Trades: {result.num_positions}")
    
    if 'portfolio_efficiency' in result.metrics:
        print(f"Efficiency: {result.metrics['portfolio_efficiency']:.1%}")
    
    # Gap closure
    print(f"\n📊 Gap Closure:")
    print(f"{'Greek':<10} {'Gap':<12} {'Achieved':<12} {'% Closed':<10}")
    print("-" * 44)
    
    for g in ['delta', 'gamma', 'vega', 'theta']:
        gap_val = gap[g]
        achieved = result.achieved_greeks.get(g, 0)
        
        if abs(gap_val) > 1e-8:
            pct = (achieved / gap_val) * 100
        else:
            pct = 100 if abs(achieved) < 1e-6 else 0
        
        print(f"{g:<10} {gap_val:+11.6f} {achieved:+11.6f} {pct:9.1f}%")
    
    # Recommendations
    if result.recommendations:
        print(f"\n" + "=" * 80)
        print(f"RECOMMENDED TRADES")
        print("=" * 80)
        
        for i, rec in enumerate(result.recommendations[:10], 1):
            print(f"\n{i}. {rec['action']} {abs(rec['quantity']):.0f} shares")
            print(f"   {rec['market_title'][:65]}")
            print(f"   {rec['category']} | ${rec['position_value']:,.2f}")
            print(f"   Δ:{rec['greek_contributions']['delta']:+.4f} "
                  f"Γ:{rec['greek_contributions']['gamma']:+.4f} "
                  f"V:{rec['greek_contributions']['vega']:+.4f} "
                  f"θ:{rec['greek_contributions']['theta']:+.4f}")
    
    # Projected
    print(f"\n" + "=" * 80)
    print("PROJECTED GREEKS AFTER TRADES")
    print("=" * 80)
    
    print(f"\n{'Greek':<10} {'Current':<12} {'Projected':<12} {'Target':<12} {'vs Target':<12}")
    print("-" * 58)
    
    for g in ['delta', 'gamma', 'vega', 'theta']:
        curr = current.get(g, 0)
        proj = curr + result.achieved_greeks.get(g, 0)
        targ = target_greeks.get(g, curr)
        diff = proj - targ
        
        print(f"{g:<10} {curr:+11.6f} {proj:+11.6f} {targ:+11.6f} {diff:+11.6f}")
    
    return {
        'success': result.success,
        'recommendations': result.recommendations,
        'result': result
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Portfolio Manager")
    parser.add_argument('-f', '--file', required=True, help='Markets JSON')
    parser.add_argument('--mode', choices=['analyze', 'optimize'], required=True)
    
    # Mode 1
    parser.add_argument('--market', help='Market ID (mode=analyze)')
    parser.add_argument('--quantity', type=float, help='Quantity (mode=analyze)')
    
    # Mode 2
    parser.add_argument('--target-delta', type=float)
    parser.add_argument('--target-gamma', type=float)
    parser.add_argument('--target-vega', type=float)
    parser.add_argument('--target-theta', type=float)
    parser.add_argument('--max-investment', type=float, default=10000)
    
    # Common
    parser.add_argument('--no-correlations', action='store_true')
    parser.add_argument('--correlation-file', help='Correlation CSV')
    
    args = parser.parse_args()
    
    # Load correlation matrix
    corr_matrix = None
    if args.correlation_file and os.path.exists(args.correlation_file):
        corr_matrix = pd.read_csv(args.correlation_file, index_col=0)
        print(f"✅ Loaded correlations")
    
    # Load markets
    print(f"📥 Loading markets...")
    all_markets = load_markets_from_json(args.file)
    print(f"✅ {len(all_markets)} markets")
    
    # Calculate Greeks (suppress warnings)
    print("⚙️  Calculating Greeks...")
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
    
    print(f"✅ {len(markets_with_greeks)} markets with Greeks")
    
    # Create manager
    portfolio = Portfolio(positions=markets_with_greeks, correlation_matrix=corr_matrix)
    manager = IntegratedPortfolioManager(portfolio=portfolio)
    
    use_corr = not args.no_correlations and corr_matrix is not None
    
    # Execute
    if args.mode == 'analyze':
        if not args.market or args.quantity is None:
            print("❌ Need --market and --quantity")
            sys.exit(1)
        
        analyze_position_impact(manager, args.market, args.quantity, use_corr)
    
    elif args.mode == 'optimize':
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
            print("❌ Need at least one target Greek")
            sys.exit(1)
        
        optimize_to_target_greeks(manager, targets, args.max_investment, use_corr)