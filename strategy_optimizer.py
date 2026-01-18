"""
AI Strategy Optimizer for Polymarket Portfolio Greeks
Integrates with greeksv2.py to provide API-friendly optimization
"""

import numpy as np
from typing import List, Dict, Optional
from greeksv2 import SimplifiedGreekOptimizationAgent, MarketData, create_sample_markets_from_inputs
import json


def optimize_strategy(
    markets_with_greeks: List[Dict],
    target_greeks: Dict[str, float],
    max_budget: float,
    initial_greeks: Optional[Dict[str, float]] = None,
    current_portfolio: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Optimize portfolio strategy to achieve target Greeks within budget.
    
    Args:
        markets_with_greeks: List of markets with their calculated Greeks
            Format: [
                {
                    "id": str,
                    "question": str,
                    "yes_price": float,
                    "no_price": float,
                    "expiry_days": int,
                    "delta": float,
                    "gamma": float,
                    "vega": float,
                    "theta": float,
                    "liquidity": float,
                    "volume": float,
                    "relatedCommodity": str
                },
                ...
            ]
        target_greeks: Desired Greeks to optimize for
            Format: {"delta": float, "gamma": float, "vega": float, "theta": float}
        max_budget: Maximum budget to spend in USD
        initial_greeks: Current portfolio Greeks (optional)
        current_portfolio: Current positions {market_id: quantity} (optional)
    
    Returns:
        Dictionary with optimization results:
        {
            "success": bool,
            "optimal_positions": [
                {
                    "market_id": str,
                    "market_question": str,
                    "action": str,  # "BUY YES" or "BUY NO"
                    "quantity": float,
                    "position_value": float,
                    "yes_price": float,
                    "no_price": float,
                    "greek_contributions": {...}
                },
                ...
            ],
            "achieved_greeks": {...},
            "target_greeks": {...},
            "deviations": {...},
            "total_investment": float,
            "num_positions": int,
            "optimization_time_ms": float,
            "metrics": {...},
            "initial_greeks": {...} (if provided)
        }
    """
    
    # Convert markets to MarketData format
    market_data_list = []
    for market in markets_with_greeks:
        market_data = MarketData(
            id=market.get('id'),
            title=market.get('question', 'Unknown Market'),
            yes_price=float(market.get('yes_price', 0.5)),
            no_price=float(market.get('no_price', 0.5)),
            expiry_days=int(market.get('expiry_days', 30)),
            delta=float(market.get('delta', 0.0)),
            gamma=float(market.get('gamma', 0.0)),
            vega=float(market.get('vega', 0.0)),
            theta=float(market.get('theta', 0.0)),
            category=market.get('relatedCommodity', 'general'),
            liquidity=float(market.get('liquidity', 100000.0)),
            volume_24h=float(market.get('volume', 500000.0)),
            tags=[market.get('relatedCommodity')] if market.get('relatedCommodity') else []
        )
        market_data_list.append(market_data)
    
    if not market_data_list:
        return {
            "success": False,
            "error": "No markets provided for optimization",
            "optimal_positions": [],
            "achieved_greeks": {},
            "target_greeks": target_greeks,
            "deviations": {},
            "total_investment": 0.0,
            "num_positions": 0,
            "optimization_time_ms": 0.0,
            "metrics": {}
        }
    
    # Create optimization agent
    print(f"[OPTIMIZER] Creating agent with {len(market_data_list)} markets")
    
    # Calculate market Greek ranges for feasibility check
    greek_ranges = {}
    for greek in ['delta', 'gamma', 'vega', 'theta']:
        values = [abs(getattr(m, greek, 0.0)) for m in market_data_list]
        if values:
            greek_ranges[greek] = {
                'max': max(values),
                'total_positive': sum(v for m in market_data_list if (v := getattr(m, greek, 0.0)) > 0),
                'total_negative': sum(abs(v) for m in market_data_list if (v := getattr(m, greek, 0.0)) < 0)
            }
    
    print(f"[OPTIMIZER] Market Greek capabilities:")
    for greek, target in target_greeks.items():
        if greek in greek_ranges:
            r = greek_ranges[greek]
            print(f"  {greek}: target={target:.2f}, max_single={r['max']:.4f}, total_pos={r['total_positive']:.2f}, total_neg={r['total_negative']:.2f}")
    
    agent = SimplifiedGreekOptimizationAgent(
        markets=market_data_list,
        current_portfolio=current_portfolio or {},
        seed=42
    )
    
    # Calculate max position per market - be generous to allow hitting targets
    # Use 40% of budget or $2000, whichever is larger (more flexible)
    max_position_per_market = max(max_budget * 0.4, min(2000.0, max_budget))
    print(f"[OPTIMIZER] Max position per market: ${max_position_per_market:.2f} (budget: ${max_budget:.2f})")
    print(f"[OPTIMIZER] Target Greeks: {target_greeks}")
    
    # Calculate adaptive weights to balance different magnitude Greeks
    # All Greeks should have roughly equal importance in the objective function
    weights = {}
    target_magnitudes = {k: abs(v) for k, v in target_greeks.items() if abs(v) > 1e-8}
    
    if target_magnitudes:
        # Normalize so all Greeks contribute equally to objective
        max_magnitude = max(target_magnitudes.values())
        for greek in target_greeks.keys():
            if abs(target_greeks[greek]) > 1e-8:
                # Equal weighting for all non-zero targets
                weights[greek] = 1.0
            else:
                weights[greek] = 0.1  # Lower weight for zero targets
    else:
        weights = {k: 1.0 for k in target_greeks.keys()}
    
    print(f"[OPTIMIZER] Weights: {weights}")
    
    # Run optimization with MAXIMUM ACCURACY priority
    # L1 and concentration penalties are minimized to let optimizer focus on targets
    print(f"[OPTIMIZER] Starting scipy optimization...")
    result = agent.optimize(
        target_greeks=target_greeks,
        weights=weights,  # Use adaptive weights
        max_total_investment=max_budget,
        max_position_per_market=max_position_per_market,
        min_position_size=0.1,  # Very low for maximum flexibility
        l1_penalty=0.0001,  # Nearly zero - don't penalize position size
        concentration_penalty=0.0001,  # Nearly zero - allow any concentration
        max_iterations=1500,  # More iterations for convergence
        ftol=1e-9,  # Very tight tolerance
        deterministic=True
    )
    
    print(f"[OPTIMIZER] Optimization complete!")
    print(f"[OPTIMIZER] Success: {result.success}, Positions: {result.num_positions}, Investment: ${result.total_investment:.2f}")
    
    # Format response
    # Add rho: 0 to achieved_greeks for frontend compatibility
    achieved_greeks_with_rho = dict(result.achieved_greeks)
    achieved_greeks_with_rho['rho'] = 0.0
    
    response = {
        "success": result.success,
        "optimal_positions": result.recommendations,
        "achieved_greeks": achieved_greeks_with_rho,
        "target_greeks": result.target_greeks,
        "deviations": result.deviations,
        "total_investment": result.total_investment,
        "num_positions": result.num_positions,
        "optimization_time_ms": result.optimization_time_ms,
        "metrics": result.metrics
    }
    
    # Add initial Greeks if provided
    if initial_greeks:
        # Ensure initial_greeks has rho
        initial_greeks_with_rho = dict(initial_greeks)
        if 'rho' not in initial_greeks_with_rho:
            initial_greeks_with_rho['rho'] = 0.0
        
        response["initial_greeks"] = initial_greeks_with_rho
        response["greek_changes_from_initial"] = {
            greek: result.achieved_greeks.get(greek, 0.0) - initial_greeks.get(greek, 0.0)
            for greek in ['delta', 'gamma', 'vega', 'theta']
        }
        response["greek_changes_from_initial"]['rho'] = 0.0
    
    return response


def test_optimizer():
    """Test the optimizer with sample data"""
    print("Testing Strategy Optimizer...")
    
    # Sample markets
    markets = [
        {
            "id": "market_1",
            "question": "Will Bitcoin reach $100k in 2026?",
            "yes_price": 0.65,
            "no_price": 0.35,
            "expiry_days": 365,
            "delta": 1.5,
            "gamma": 0.15,
            "vega": 0.25,
            "theta": -0.02,
            "liquidity": 1000000,
            "volume": 5000000,
            "relatedCommodity": "crypto"
        },
        {
            "id": "market_2",
            "question": "Will oil prices exceed $100?",
            "yes_price": 0.42,
            "no_price": 0.58,
            "expiry_days": 90,
            "delta": 0.5,
            "gamma": 0.18,
            "vega": 0.22,
            "theta": -0.03,
            "liquidity": 600000,
            "volume": 2000000,
            "relatedCommodity": "oil"
        }
    ]
    
    # Target Greeks
    target_greeks = {
        "delta": 2.0,
        "gamma": 0.3,
        "vega": 0.5,
        "theta": -0.1
    }
    
    # Run optimization
    result = optimize_strategy(
        markets_with_greeks=markets,
        target_greeks=target_greeks,
        max_budget=5000.0
    )
    
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    test_optimizer()
