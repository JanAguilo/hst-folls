"""
Simplified Greek Optimization Agent
Inputs: title, yes_price, no_price, expiry_days, delta, gamma, vega, theta
"""

import numpy as np
from scipy.optimize import minimize, Bounds
from typing import List, Dict, Tuple, Optional, Any
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from enum import Enum
import warnings

# Suppress optimization warnings
warnings.filterwarnings('ignore', message='Values in x were outside bounds')
np.random.seed(42)  # For reproducibility

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class OptimizationObjective(Enum):
    """Optimization objectives"""
    MINIMIZE_DEVIATION = "minimize_deviation"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_HEDGE_EFFICIENCY = "maximize_hedge_efficiency"
    BALANCED = "balanced"

@dataclass
class MarketData:
    """Simplified market data structure with required fields only"""
    title: str
    yes_price: float
    no_price: float
    expiry_days: int
    delta: float
    gamma: float
    vega: float
    theta: float
    id: Optional[str] = None  # Will be auto-generated
    category: str = "general"
    current_price: Optional[float] = None  # Will be calculated from yes_price
    liquidity: float = 100000.0
    volume_24h: float = 500000.0
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.id is None:
            # Create ID from title
            self.id = "market_" + str(abs(hash(self.title)))[:8]
        if self.current_price is None:
            # Current price is the yes_price (probability of yes)
            self.current_price = self.yes_price

@dataclass 
class OptimizationResult:
    """Structured optimization result"""
    success: bool
    optimal_quantities: Dict[str, float]
    achieved_greeks: Dict[str, float]
    target_greeks: Dict[str, float]
    deviations: Dict[str, float]
    total_investment: float
    num_positions: int
    optimization_time_ms: float
    metrics: Dict[str, Any]
    recommendations: List[Dict]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str, indent=2)

# ============================================================================
# MARKET DATA CREATION FUNCTIONS
# ============================================================================

def create_sample_markets_from_inputs(market_inputs: List[Dict]) -> List[MarketData]:
    """
    Create MarketData objects from simplified inputs.
    
    Expected input format for each market:
    {
        "title": str,
        "yes_price": float,      # Price for YES outcome (0-1)
        "no_price": float,       # Price for NO outcome (0-1)
        "expiry_days": int,
        "delta": float,
        "gamma": float,
        "vega": float,
        "theta": float
    }
    """
    markets = []
    
    for i, market_input in enumerate(market_inputs):
        try:
            market = MarketData(
                id=f"market_{i+1}",
                title=market_input.get("title", f"Market {i+1}"),
                yes_price=float(market_input["yes_price"]),
                no_price=float(market_input["no_price"]),
                expiry_days=int(market_input["expiry_days"]),
                delta=float(market_input["delta"]),
                gamma=float(market_input["gamma"]),
                vega=float(market_input["vega"]),
                theta=float(market_input["theta"]),
                category=market_input.get("category", "general"),
                liquidity=market_input.get("liquidity", 100000.0),
                volume_24h=market_input.get("volume_24h", 500000.0),
                tags=market_input.get("tags", [])
            )
            markets.append(market)
        except KeyError as e:
            print(f"Warning: Missing required field {e} in market input {i+1}")
            continue
    
    return markets

def create_default_sample_markets() -> List[MarketData]:
    """Create default sample markets for testing"""
    market_inputs = [
        {
            "title": "Will Bitcoin reach $100K by end of year?",
            "yes_price": 0.65,
            "no_price": 0.35,
            "expiry_days": 180,
            "delta": 1.2,
            "gamma": 0.15,
            "vega": 0.25,
            "theta": -0.02,
            "category": "crypto",
            "liquidity": 500000.0,
            "volume_24h": 2500000.0,
            "tags": ["crypto", "bitcoin", "high-vol"]
        },
        {
            "title": "Will the Fed cut rates in Q2?",
            "yes_price": 0.45,
            "no_price": 0.55,
            "expiry_days": 90,
            "delta": -0.7,
            "gamma": 0.12,
            "vega": 0.18,
            "theta": -0.03,
            "category": "economics",
            "liquidity": 750000.0,
            "volume_24h": 1800000.0,
            "tags": ["economics", "rates", "fed"]
        },
        {
            "title": "Will S&P 500 finish positive this year?",
            "yes_price": 0.72,
            "no_price": 0.28,
            "expiry_days": 365,
            "delta": 0.9,
            "gamma": 0.08,
            "vega": 0.12,
            "theta": -0.01,
            "category": "stocks",
            "liquidity": 1200000.0,
            "volume_24h": 5000000.0,
            "tags": ["stocks", "index", "sp500"]
        },
        {
            "title": "Will Democrats win the Senate?",
            "yes_price": 0.55,
            "no_price": 0.45,
            "expiry_days": 120,
            "delta": 0.4,
            "gamma": 0.05,
            "vega": 0.08,
            "theta": -0.02,
            "category": "politics",
            "liquidity": 300000.0,
            "volume_24h": 800000.0,
            "tags": ["politics", "election", "senate"]
        },
        {
            "title": "Will inflation drop below 3%?",
            "yes_price": 0.38,
            "no_price": 0.62,
            "expiry_days": 60,
            "delta": -0.6,
            "gamma": 0.10,
            "vega": 0.15,
            "theta": -0.04,
            "category": "economics",
            "liquidity": 400000.0,
            "volume_24h": 1200000.0,
            "tags": ["economics", "inflation"]
        },
        {
            "title": "Will Tesla deliver 2M vehicles this year?",
            "yes_price": 0.58,
            "no_price": 0.42,
            "expiry_days": 120,
            "delta": 0.8,
            "gamma": 0.20,
            "vega": 0.30,
            "theta": -0.05,
            "category": "stocks",
            "liquidity": 800000.0,
            "volume_24h": 3500000.0,
            "tags": ["stocks", "tesla", "automotive"]
        },
        {
            "title": "Will oil prices exceed $100?",
            "yes_price": 0.42,
            "no_price": 0.58,
            "expiry_days": 90,
            "delta": 0.5,
            "gamma": 0.18,
            "vega": 0.22,
            "theta": -0.03,
            "category": "commodities",
            "liquidity": 600000.0,
            "volume_24h": 2000000.0,
            "tags": ["commodities", "oil", "energy"]
        },
        {
            "title": "Will AI regulation pass this year?",
            "yes_price": 0.35,
            "no_price": 0.65,
            "expiry_days": 180,
            "delta": 0.3,
            "gamma": 0.07,
            "vega": 0.10,
            "theta": -0.015,
            "category": "technology",
            "liquidity": 400000.0,
            "volume_24h": 1500000.0,
            "tags": ["technology", "ai", "regulation"]
        }
    ]
    
    return create_sample_markets_from_inputs(market_inputs)

# ============================================================================
# SIMPLIFIED GREEK OPTIMIZATION AGENT
# ============================================================================

class SimplifiedGreekOptimizationAgent:
    """
    Simplified Greek optimization agent that works with minimal inputs:
    title, yes_price, no_price, expiry_days, delta, gamma, vega, theta
    """
    
    def __init__(
        self,
        markets: List[MarketData],
        current_portfolio: Optional[Dict[str, float]] = None,
        logger: Optional[logging.Logger] = None,
        seed: int = 42
    ):
        """
        Args:
            markets: List of MarketData objects
            current_portfolio: Current positions {market_id: quantity}
            logger: Optional logger for debugging
            seed: Random seed for reproducibility
        """
        self.markets = markets
        self.current_portfolio = current_portfolio or {}
        self.market_dict = {m.id: m for m in markets}
        self.market_ids = [m.id for m in markets]
        self.n_markets = len(markets)
        self.logger = logger or logging.getLogger(__name__)
        self.seed = seed
        np.random.seed(seed)
        
        # Supported Greeks (simplified to the ones we have)
        self.all_greeks = ['delta', 'gamma', 'vega', 'theta']
        
        # Analyze market characteristics
        self._analyze_markets()
        
        # Default weights
        self.default_weights = {
            'delta': 1.0,    # Directional exposure
            'gamma': 1.0,    # Convexity
            'vega': 1.0,     # Volatility exposure
            'theta': 1.0     # Time decay
        }
        
        # Default risk limits
        self.risk_limits = self._calculate_default_limits()
        
    def _analyze_markets(self):
        """Analyze market characteristics for adaptive scaling"""
        self.greek_stats = {}
        
        for greek in self.all_greeks:
            values = [getattr(m, greek, 0.0) for m in self.markets]
            abs_values = [abs(v) for v in values]
            
            if values:
                self.greek_stats[greek] = {
                    'max': max(abs_values),
                    'min': min(abs_values) if min(abs_values) > 0 else 0.001,
                    'mean': np.mean(abs_values),
                    'median': np.median(abs_values),
                    'range': max(abs_values) - min(abs_values) if len(abs_values) > 1 else 1.0
                }
            else:
                self.greek_stats[greek] = {
                    'max': 1.0, 'min': 0.001, 'mean': 0.5, 'median': 0.5, 'range': 1.0
                }
        
        # Price statistics (using yes_price as current price)
        prices = [m.yes_price for m in self.markets]
        self.price_stats = {
            'max': max(prices) if prices else 1.0,
            'min': min(prices) if prices else 0.01,
            'mean': np.mean(prices) if prices else 1.0,
            'median': np.median(prices) if prices else 1.0
        }
        
    def _calculate_default_limits(self):
        """Calculate default risk limits based on market characteristics"""
        limits = {}
        
        # Scale limits by market characteristics
        for greek in self.all_greeks:
            stat = self.greek_stats[greek]
            if greek == 'delta':
                limits[f'max_{greek}'] = max(10.0, stat['max'] * 20)
            elif greek == 'gamma':
                limits[f'max_{greek}'] = max(5.0, stat['max'] * 10)
            elif greek == 'vega':
                limits[f'max_{greek}'] = max(5.0, stat['max'] * 10)
            elif greek == 'theta':
                limits[f'max_{greek}'] = max(2.0, stat['max'] * 5)
        
        limits['max_position_value'] = self.price_stats['mean'] * 200
        limits['max_total_investment'] = self.price_stats['mean'] * len(self.markets) * 100
        
        return limits
    
    def calculate_portfolio_greeks(
        self, 
        quantities: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate aggregated Greeks for all positions.
        """
        portfolio_greeks = {
            # All Greeks we support
            'delta': 0.0,
            'gamma': 0.0,
            'vega': 0.0,
            'theta': 0.0,
            
            # Investment metrics
            'total_investment': 0.0,
            'num_positions': 0,
            'long_exposure': 0.0,
            'short_exposure': 0.0,
            'max_position_value': 0.0,
            'avg_position_value': 0.0,
            'greek_concentration': 0.0
        }
        
        position_values = []
        greek_contributions = {greek: [] for greek in self.all_greeks}
        
        for market_id, qty in quantities.items():
            market = self.market_dict.get(market_id)
            if not market or abs(qty) < 1e-10:
                continue
            
            # Calculate all Greeks
            for greek in self.all_greeks:
                greek_value = getattr(market, greek, 0.0)
                contribution = greek_value * qty
                portfolio_greeks[greek] += contribution
                greek_contributions[greek].append(abs(contribution))
            
            # Investment metrics - using yes_price as price
            position_value = abs(qty) * market.yes_price
            portfolio_greeks['total_investment'] += position_value
            position_values.append(position_value)
            
            if abs(qty) > 1e-8:
                portfolio_greeks['num_positions'] += 1
            
            # Exposure metrics
            if qty > 0:
                portfolio_greeks['long_exposure'] += position_value
            else:
                portfolio_greeks['short_exposure'] += position_value
        
        # Additional metrics
        if position_values:
            portfolio_greeks['max_position_value'] = max(position_values)
            portfolio_greeks['avg_position_value'] = sum(position_values) / len(position_values)
        
        # Calculate Greek concentration
        for greek in self.all_greeks:
            contributions = greek_contributions[greek]
            if contributions and sum(contributions) > 0:
                squared_shares = [(c / sum(contributions)) ** 2 for c in contributions]
                portfolio_greeks['greek_concentration'] += sum(squared_shares) / len(self.all_greeks)
        
        portfolio_greeks['net_exposure'] = (
            portfolio_greeks['long_exposure'] - portfolio_greeks['short_exposure']
        )
        
        return portfolio_greeks
    
    def _create_objective_function(
        self,
        target_greeks: Dict[str, float],
        weights: Optional[Dict[str, float]] = None,
        l1_penalty: float = 0.001,
        concentration_penalty: float = 0.0005
    ):
        """Create objective function with proper scaling"""
        weights = weights or self.default_weights
        
        # Calculate adaptive scaling factors
        scaling_factors = {}
        for greek in self.all_greeks:
            if greek in target_greeks:
                target_mag = abs(target_greeks[greek])
                market_max = self.greek_stats[greek]['max']
                # Use geometric mean for better scaling
                scaling_factors[greek] = 1.0 / np.sqrt(max(target_mag, 1e-8) * max(market_max, 1e-8))
            else:
                scaling_factors[greek] = 1.0 / max(self.greek_stats[greek]['max'], 1e-8)
        
        def objective_function(x: np.ndarray) -> float:
            quantities = dict(zip(self.market_ids, x))
            current_greeks = self.calculate_portfolio_greeks(quantities)
            
            total_error = 0.0
            
            # 1. TARGET DEVIATION ERROR (main objective)
            for greek, target in target_greeks.items():
                if greek not in current_greeks:
                    continue
                
                weight = weights.get(greek, 1.0)
                current = current_greeks[greek]
                
                # Adaptive error calculation
                if abs(target) > 1e-8:
                    # Relative squared error for non-zero targets
                    rel_error = (current - target) / target
                    error_term = (rel_error ** 2)
                else:
                    # Scaled absolute error for zero targets
                    scale = scaling_factors.get(greek, 1.0)
                    error_term = (current ** 2) * scale
                
                total_error += weight * error_term * 1000.0
            
            # 2. POSITION COST PENALTY (scaled by price)
            position_magnitude = np.sum(np.abs(x))
            price_array = np.array([self.market_dict[mid].yes_price for mid in self.market_ids])
            price_weighted = np.sum(np.abs(x) * price_array / self.price_stats['mean'])
            
            total_error += l1_penalty * price_weighted
            
            # 3. CONCENTRATION PENALTY (encourage diversification)
            non_zero = np.sum(np.abs(x) > 1e-6)
            if non_zero > 0:
                # Penalize too few positions
                if non_zero < 3:
                    total_error += 0.01 * (3 - non_zero)
                
                # Penalize concentration in top positions
                sorted_abs = np.sort(np.abs(x))[::-1]
                if len(sorted_abs) >= 3:
                    top3_concentration = np.sum(sorted_abs[:3]) / (np.sum(sorted_abs) + 1e-10)
                    total_error += concentration_penalty * top3_concentration
            
            return total_error
        
        return objective_function
    
    def optimize(
        self,
        target_greeks: Dict[str, float],
        max_position_per_market: float = 1000.0,
        max_total_investment: float = 10000.0,
        min_position_size: float = 0.1,
        weights: Optional[Dict[str, float]] = None,
        risk_tolerance: Optional[Dict[str, float]] = None,
        enforce_integer_positions: bool = False,
        l1_penalty: float = 0.001,
        concentration_penalty: float = 0.0005,
        max_iterations: int = 2000,
        ftol: float = 1e-10,
        deterministic: bool = True
    ) -> OptimizationResult:
        """
        Optimize portfolio to target Greeks.
        """
        start_time = datetime.now()
        
        if deterministic:
            np.random.seed(self.seed)
        
        # Validate targets
        for greek in target_greeks:
            if greek not in self.all_greeks:
                print(f"Warning: Greek '{greek}' not supported. Supported Greeks: {self.all_greeks}")
        
        # Set up bounds
        bounds = Bounds(
            lb=-max_position_per_market,
            ub=max_position_per_market
        )
        
        # Constraints
        constraints = []
        
        # 1. Budget constraint
        price_array = np.array([self.market_dict[mid].yes_price for mid in self.market_ids])
        
        def budget_constraint(x):
            total_inv = np.sum(np.abs(x) * price_array)
            return max_total_investment - total_inv
        
        constraints.append({'type': 'ineq', 'fun': budget_constraint})
        
        # 2. Risk constraints
        risk_limits = risk_tolerance or self.risk_limits
        
        def risk_constraint_factory(greek_name, limit):
            def constraint(x):
                total_greek = 0.0
                for i, mid in enumerate(self.market_ids):
                    market = self.market_dict[mid]
                    greek_value = getattr(market, greek_name, 0.0)
                    total_greek += greek_value * x[i]
                return limit - abs(total_greek)
            return constraint
        
        for constraint_name, limit in risk_limits.items():
            if constraint_name.startswith('max_'):
                greek_name = constraint_name[4:]
                if greek_name in self.all_greeks:
                    constraints.append({
                        'type': 'ineq',
                        'fun': risk_constraint_factory(greek_name, limit)
                    })
        
        # 3. Create objective function
        objective_fn = self._create_objective_function(
            target_greeks=target_greeks,
            weights=weights,
            l1_penalty=l1_penalty,
            concentration_penalty=concentration_penalty
        )
        
        # 4. Intelligent initial guess
        x0 = np.zeros(len(self.market_ids))
        
        # Use current portfolio if exists
        for i, mid in enumerate(self.market_ids):
            x0[i] = self.current_portfolio.get(mid, 0.0)
        
        # If no current positions, create smart initial guess
        if np.all(np.abs(x0) < 1e-8):
            # For each Greek target, find best markets
            for greek, target in target_greeks.items():
                if abs(target) < 1e-8:
                    continue
                
                # Score markets for this Greek
                market_scores = []
                for i, market in enumerate(self.markets):
                    greek_value = getattr(market, greek, 0.0)
                    if abs(greek_value) > 1e-8:
                        # Score considers Greek magnitude and price efficiency
                        efficiency = abs(greek_value) / (market.yes_price + 1e-8)
                        alignment = 1.0 if greek_value * target > 0 else 0.5
                        score = efficiency * alignment
                        market_scores.append((i, score, greek_value))
                
                # Sort and allocate
                market_scores.sort(key=lambda x: x[1], reverse=True)
                num_to_use = min(3, len(market_scores))
                
                for rank, (i, score, greek_value) in enumerate(market_scores[:num_to_use]):
                    # Distributed allocation
                    qty_estimate = target / (greek_value * (num_to_use - rank))
                    
                    # Budget constraint
                    max_qty = max_total_investment / (len(target_greeks) * market.yes_price)
                    qty_estimate = np.clip(qty_estimate, -max_qty, max_qty)
                    
                    x0[i] += qty_estimate
        
        # 5. Run optimization with multiple starting points
        best_result = None
        best_x = None
        best_fun = float('inf')
        
        # Generate deterministic starting points
        starting_points = [x0]
        
        # Add perturbed versions
        if deterministic:
            seeds = [self.seed + i * 100 for i in range(1, 4)]
            for seed_idx, seed in enumerate(seeds):
                np.random.seed(seed)
                perturbation = 0.1 * (seed_idx + 1)
                perturbed = x0 + np.random.normal(0, perturbation, size=len(x0))
                perturbed = np.clip(perturbed, -max_position_per_market * 0.5, max_position_per_market * 0.5)
                starting_points.append(perturbed)
        
        # Try each starting point
        for start_idx, initial_guess in enumerate(starting_points):
            try:
                result = minimize(
                    objective_fn,
                    initial_guess,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': max_iterations,
                        'ftol': ftol,
                        'eps': 1e-8,
                        'disp': False
                    }
                )
                
                if result.success and result.fun < best_fun:
                    best_result = result
                    best_x = result.x
                    best_fun = result.fun
                    
            except Exception as e:
                if self.logger:
                    self.logger.debug(f"Start point {start_idx} failed: {e}")
        
        # 6. Process results
        if best_result is not None and best_result.success:
            success = True
            message = best_result.message
            
            raw_quantities = dict(zip(self.market_ids, best_x))
            
            # Apply minimum position size
            optimal_quantities = {}
            for mid, qty in raw_quantities.items():
                if abs(qty) >= min_position_size:
                    if enforce_integer_positions:
                        optimal_quantities[mid] = np.round(qty)
                    else:
                        optimal_quantities[mid] = float(qty)
            
            # Calculate final Greeks
            final_greeks = self.calculate_portfolio_greeks(optimal_quantities)
            
            # Calculate deviations
            deviations = {}
            relative_deviations = {}
            for greek in target_greeks:
                achieved = final_greeks.get(greek, 0.0)
                target = target_greeks[greek]
                deviations[greek] = achieved - target
                if abs(target) > 1e-8:
                    relative_deviations[greek] = deviations[greek] / abs(target)
                else:
                    relative_deviations[greek] = achieved
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                optimal_quantities, target_greeks, final_greeks
            )
            
            # Calculate metrics
            metrics = {
                'optimization_success': success,
                'optimization_message': message,
                'objective_value': float(best_fun),
                'iterations': best_result.nit if best_result else 0,
                'execution_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'convergence_status': 'converged',
                'num_starting_points': len(starting_points),
                'relative_deviations': relative_deviations,
                'portfolio_efficiency': self._calculate_efficiency(target_greeks, final_greeks)
            }
            
        else:
            # Fallback to deterministic heuristic
            success = False
            message = "Optimization failed, using heuristic"
            optimal_quantities, final_greeks = self._deterministic_heuristic(
                target_greeks, max_total_investment, min_position_size
            )
            deviations = {}
            recommendations = []
            metrics = {
                'optimization_success': False,
                'optimization_message': message,
                'fallback_used': True
            }
        
        # Create result object
        optimization_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return OptimizationResult(
            success=success,
            optimal_quantities=optimal_quantities,
            achieved_greeks=final_greeks,
            target_greeks=target_greeks,
            deviations=deviations,
            total_investment=final_greeks.get('total_investment', 0.0),
            num_positions=final_greeks.get('num_positions', 0),
            optimization_time_ms=optimization_time,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _deterministic_heuristic(
        self,
        target_greeks: Dict[str, float],
        max_total_investment: float,
        min_position_size: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Deterministic heuristic for portfolio construction"""
        quantities = {}
        remaining_budget = max_total_investment
        
        # Score markets based on ability to meet targets
        market_scores = []
        for market in self.markets:
            score = 0.0
            for greek, target in target_greeks.items():
                greek_value = getattr(market, greek, 0.0)
                if abs(greek_value) > 1e-8 and abs(target) > 1e-8:
                    # Positive score for aligned Greeks
                    alignment = greek_value * target
                    efficiency = abs(greek_value) / market.yes_price
                    score += efficiency * (1.0 if alignment > 0 else 0.3)
            
            if score > 0:
                market_scores.append((market, score))
        
        # Sort by score (deterministic)
        market_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate to top markets
        for market, score in market_scores[:5]:  # Use top 5 markets
            if remaining_budget <= 0:
                break
            
            # Try to find optimal quantity for this market
            best_qty = 0.0
            best_error = float('inf')
            
            # Test a few quantity points
            test_quantities = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            for test_qty in test_quantities:
                # Test both long and short
                for sign in [1, -1]:
                    qty = test_qty * sign
                    
                    # Check budget
                    cost = abs(qty) * market.yes_price
                    if cost > remaining_budget:
                        continue
                    
                    # Calculate Greek contributions
                    total_error = 0.0
                    for greek, target in target_greeks.items():
                        greek_value = getattr(market, greek, 0.0)
                        contribution = greek_value * qty
                        
                        # Calculate error (use relative if target non-zero)
                        if abs(target) > 1e-8:
                            error = abs((contribution - target) / target)
                        else:
                            error = abs(contribution)
                        
                        total_error += error
                    
                    if total_error < best_error:
                        best_error = total_error
                        best_qty = qty
            
            if abs(best_qty) >= min_position_size:
                quantities[market.id] = best_qty
                remaining_budget -= abs(best_qty) * market.yes_price
        
        final_greeks = self.calculate_portfolio_greeks(quantities)
        return quantities, final_greeks
    
    def _calculate_efficiency(
        self,
        target_greeks: Dict[str, float],
        achieved_greeks: Dict[str, float]
    ) -> float:
        """Calculate portfolio efficiency (0-1)"""
        if not target_greeks:
            return 0.0
        
        total_efficiency = 0.0
        for greek, target in target_greeks.items():
            achieved = achieved_greeks.get(greek, 0.0)
            if abs(target) > 1e-8:
                # Efficiency based on relative error
                rel_error = abs(achieved - target) / abs(target)
                efficiency = max(0.0, 1.0 - min(rel_error, 1.0))
            else:
                # For zero targets, efficiency based on absolute value
                efficiency = max(0.0, 1.0 - min(abs(achieved), 1.0))
            
            total_efficiency += efficiency
        
        return total_efficiency / len(target_greeks)
    
    def _generate_recommendations(
        self,
        quantities: Dict[str, float],
        target_greeks: Dict[str, float],
        achieved_greeks: Dict[str, float]
    ) -> List[Dict]:
        """Generate detailed trading recommendations"""
        recommendations = []
        
        for market_id, qty in quantities.items():
            market = self.market_dict.get(market_id)
            if not market or abs(qty) < 1e-8:
                continue
            
            position_value = abs(qty) * market.yes_price
            
            recommendation = {
                'market_id': market.id,
                'market_title': market.title,
                'category': market.category,
                'yes_price': float(market.yes_price),
                'no_price': float(market.no_price),
                'expiry_days': market.expiry_days,
                'quantity': float(qty),
                'action': 'BUY YES' if qty > 0 else 'BUY NO',
                'position_value': float(position_value),
                'percentage_of_budget': float(position_value / achieved_greeks.get('total_investment', 1.0) * 100),
                'greek_contributions': {
                    'delta': float(market.delta * qty),
                    'gamma': float(market.gamma * qty),
                    'vega': float(market.vega * qty),
                    'theta': float(market.theta * qty)
                },
                'market_characteristics': {
                    'liquidity': float(market.liquidity),
                    'volume_24h': float(market.volume_24h),
                    'expiry_days': market.expiry_days,
                    'implied_probability': float(market.yes_price * 100)
                }
            }
            recommendations.append(recommendation)
        
        # Sort by position value
        recommendations.sort(key=lambda x: abs(x['position_value']), reverse=True)
        
        return recommendations

# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================

def test_simplified_greeks():
    """Test optimization with simplified inputs"""
    print("\n" + "=" * 70)
    print("SIMPLIFIED GREEK OPTIMIZATION TEST")
    print("=" * 70)
    
    # Create markets from default data
    markets = create_default_sample_markets()
    
    # Display market information
    print(f"\nLoaded {len(markets)} markets:")
    for market in markets[:3]:  # Show first 3
        print(f"  {market.title[:40]:40} | "
              f"YES: ${market.yes_price:.3f} NO: ${market.no_price:.3f} | "
              f"Δ:{market.delta:5.2f} Γ:{market.gamma:5.2f} V:{market.vega:5.2f} θ:{market.theta:5.2f}")
    if len(markets) > 3:
        print(f"  ... and {len(markets) - 3} more markets")
    
    agent = SimplifiedGreekOptimizationAgent(markets, seed=42)
    
    # Test case with all available Greeks
    targets = {
        'delta': 2.5,      # Moderate directional exposure
        'gamma': 0.3,      # Some convexity
        'vega': 1.2,       # Volatility exposure
        'theta': -0.4      # Negative time decay (typical)
    }
    
    print(f"\nTarget Greeks:")
    for greek, value in targets.items():
        print(f"  {greek:10}: {value:8.4f}")
    
    print("\nRunning optimization...")
    result = agent.optimize(
        target_greeks=targets,
        max_total_investment=20000,
        max_position_per_market=500,
        min_position_size=0.5,
        l1_penalty=0.001,
        deterministic=True
    )
    
    print(f"\nResults:")
    print(f"Success: {result.success}")
    print(f"Total investment: ${result.total_investment:,.2f}")
    print(f"Positions: {result.num_positions}")
    print(f"Execution time: {result.optimization_time_ms:.1f} ms")
    print(f"Efficiency: {result.metrics.get('portfolio_efficiency', 0):.2%}")
    
    print(f"\nGreek Performance:")
    print(f"{'Greek':10} {'Target':>10} {'Achieved':>10} {'Deviation':>12} {'% Error':>10}")
    print(f"{'-'*60}")
    
    for greek in targets:
        target = targets[greek]
        achieved = result.achieved_greeks.get(greek, 0)
        deviation = result.deviations.get(greek, 0)
        
        if abs(target) > 1e-8:
            error_pct = abs(deviation) / abs(target) * 100
        else:
            error_pct = abs(achieved) * 100
        
        # Color coding
        if error_pct < 5:
            color = "\033[92m"
            status = "✓"
        elif error_pct < 15:
            color = "\033[93m"
            status = "~"
        elif error_pct < 30:
            color = "\033[33m"
            status = "-"
        else:
            color = "\033[91m"
            status = "✗"
        
        print(f"{greek:10} {target:10.4f} {achieved:10.4f} "
              f"{color}{deviation:12.4f}\033[0m {error_pct:9.1f}% {status}")
    
    if result.recommendations:
        print(f"\nTop Positions:")
        for i, rec in enumerate(result.recommendations[:5]):
            action_color = "\033[92m" if rec['action'] == 'BUY YES' else "\033[91m"
            action_text = "YES" if rec['action'] == 'BUY YES' else "NO"
            print(f"  {i+1:2}. {action_color}{rec['action']:8}\033[0m "
                  f"{abs(rec['quantity']):8.2f} shares of {rec['market_title'][:25]}")
            print(f"      Value: ${rec['position_value']:10,.2f} | "
                  f"YES: ${rec['yes_price']:.3f} NO: ${rec['no_price']:.3f}")
            print(f"      Greek contributions: "
                  f"Δ: {rec['greek_contributions']['delta']:7.3f} "
                  f"Γ: {rec['greek_contributions']['gamma']:7.3f} "
                  f"V: {rec['greek_contributions']['vega']:7.3f} "
                  f"θ: {rec['greek_contributions']['theta']:7.3f}")
    
    return result

def main():
    """Main demonstration"""
    print("=" * 70)
    print("SIMPLIFIED GREEK OPTIMIZATION AGENT")
    print("Inputs: title, yes_price, no_price, expiry_days, delta, gamma, vega, theta")
    print("=" * 70)
    
    # Test with simplified inputs
    result = test_simplified_greeks()
    
    # Save results
    results_dict = {
        'simplified_test': result.to_dict(),
        'timestamp': datetime.now().isoformat(),
        'notes': 'Simplified Greek optimization with minimal inputs'
    }
    
    with open('simplified_greek_results.json', 'w') as f:
        json.dump(results_dict, f, default=str, indent=2)
    
    print(f"\nResults saved to 'simplified_greek_results.json'")

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description='Simplified Greek Optimization Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         # Run demonstration with default markets
  %(prog)s --delta 2.5 --gamma 0.3 --vega 1.2 --theta -0.4 --budget 20000
  %(prog)s --delta 0.0 --gamma 0.2 --vega 0.5                       # Hedge portfolio
  %(prog)s --delta 10 --gamma 2 --vega 5 --theta -2                 # Extreme values
  
Input Format (for custom markets via JSON):
  [
    {
      "title": "Market title",
      "yes_price": 0.65,
      "no_price": 0.35,
      "expiry_days": 30,
      "delta": 1.2,
      "gamma": 0.15,
      "vega": 0.25,
      "theta": -0.02
    },
    ...
  ]
        """
    )
    
    # Greek parameters
    parser.add_argument('--delta', type=float, help='Target delta value')
    parser.add_argument('--gamma', type=float, help='Target gamma value')
    parser.add_argument('--vega', type=float, help='Target vega value')
    parser.add_argument('--theta', type=float, help='Target theta value')
    
    # Optimization parameters
    parser.add_argument('--budget', type=float, default=10000, help='Maximum total investment')
    parser.add_argument('--max-pos', type=float, default=500, help='Maximum position per market')
    parser.add_argument('--min-size', type=float, default=0.1, help='Minimum position size')
    parser.add_argument('--l1-penalty', type=float, default=0.001, help='L1 regularization penalty')
    
    # Input/output parameters
    parser.add_argument('--markets-file', type=str, help='JSON file with custom market data')
    parser.add_argument('--output', type=str, default='greek_result.json', help='Output JSON file')
    
    # Control parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--no-deterministic', action='store_true', help='Disable deterministic mode')
    
    args = parser.parse_args()
    
    try:
        if any([args.delta, args.gamma, args.vega, args.theta]):
            # Run custom optimization
            print(f"\nRunning custom Greek optimization (seed={args.seed})...")
            
            # Build targets
            targets = {}
            if args.delta is not None:
                targets['delta'] = args.delta
            if args.gamma is not None:
                targets['gamma'] = args.gamma
            if args.vega is not None:
                targets['vega'] = args.vega
            if args.theta is not None:
                targets['theta'] = args.theta
            
            if not targets:
                print("Error: No Greek targets specified!")
                parser.print_help()
                exit(1)
            
            # Load markets
            if args.markets_file:
                # Load from JSON file
                try:
                    with open(args.markets_file, 'r') as f:
                        market_inputs = json.load(f)
                    markets = create_sample_markets_from_inputs(market_inputs)
                    print(f"Loaded {len(markets)} markets from {args.markets_file}")
                except Exception as e:
                    print(f"Error loading markets file: {e}")
                    print("Using default markets instead.")
                    markets = create_default_sample_markets()
            else:
                # Use default markets
                markets = create_default_sample_markets()
                print(f"Using {len(markets)} default markets")
            
            # Create agent
            agent = SimplifiedGreekOptimizationAgent(markets, seed=args.seed)
            
            # Run optimization
            result = agent.optimize(
                target_greeks=targets,
                max_total_investment=args.budget,
                max_position_per_market=args.max_pos,
                min_position_size=args.min_size,
                l1_penalty=args.l1_penalty,
                deterministic=not args.no_deterministic
            )
            
            # Display results
            print(f"\nOptimization Results:")
            print(f"Success: {result.success}")
            print(f"Total investment: ${result.total_investment:,.2f}")
            print(f"Positions: {result.num_positions}")
            print(f"Execution time: {result.optimization_time_ms:.1f} ms")
            
            if 'portfolio_efficiency' in result.metrics:
                print(f"Portfolio efficiency: {result.metrics['portfolio_efficiency']:.2%}")
            
            print(f"\nGreek Results:")
            for greek, target in targets.items():
                achieved = result.achieved_greeks.get(greek, 0)
                deviation = result.deviations.get(greek, 0)
                
                if abs(target) > 1e-8:
                    error_pct = abs(deviation) / abs(target) * 100
                else:
                    error_pct = abs(achieved) * 100
                
                if error_pct < 5:
                    color = "\033[92m"
                    grade = "A"
                elif error_pct < 15:
                    color = "\033[93m"
                    grade = "B"
                elif error_pct < 30:
                    color = "\033[33m"
                    grade = "C"
                else:
                    color = "\033[91m"
                    grade = "D"
                
                print(f"  {greek:10}: {color}{grade}\033[0m "
                      f"Target: {target:10.4f} | "
                      f"Achieved: {achieved:10.4f} | "
                      f"Error: {error_pct:6.1f}%")
            
            if result.recommendations:
                print(f"\nRecommended Positions (top {min(5, len(result.recommendations))}):")
                for i, rec in enumerate(result.recommendations[:5]):
                    action_color = "\033[92m" if rec['action'] == 'BUY YES' else "\033[91m"
                    print(f"  {i+1:2}. {action_color}{rec['action']:8}\033[0m "
                          f"{abs(rec['quantity']):8.2f} shares of {rec['market_title'][:30]}")
                    print(f"      Value: ${rec['position_value']:10,.2f} | "
                          f"YES: ${rec['yes_price']:.3f} NO: ${rec['no_price']:.3f}")
                    print(f"      Greek contributions: "
                          f"Δ: {rec['greek_contributions']['delta']:7.3f} "
                          f"Γ: {rec['greek_contributions']['gamma']:7.3f} "
                          f"V: {rec['greek_contributions']['vega']:7.3f} "
                          f"θ: {rec['greek_contributions']['theta']:7.3f}")
            
            # Save to file
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, default=str, indent=2)
            print(f"\nResults saved to {args.output}")
            
        else:
            # No arguments - run demo
            print("No specific targets provided. Running demonstration...\n")
            main()
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)