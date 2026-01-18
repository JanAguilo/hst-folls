from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime, timezone

# Add parent directory to path to import portfolio_to_greeks and strategy_optimizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from portfolio_to_greeks import calculate_portfolio_greeks
from strategy_optimizer import optimize_strategy
from polymarket_greeks import PolymarketMarket, Greeks, BlackScholesDigital, BrownianBridge

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load data files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMODITY_MARKETS_PATH = os.path.join(BASE_DIR, 'commodity_markets.json')
COMMODITY_MAPPING_PATH = os.path.join(BASE_DIR, 'commodity_to_main_asset_mapping.json')

# Initialize Greeks calculators
bs_calculator = BlackScholesDigital()
bridge_calculator = BrownianBridge()

# Load markets data
with open(COMMODITY_MARKETS_PATH, 'r', encoding='utf-8') as f:
    markets_data = json.load(f)

# Load commodity mapping
with open(COMMODITY_MAPPING_PATH, 'r', encoding='utf-8') as f:
    commodity_mapping = json.load(f)

print(f"[OK] Loaded {len(markets_data.get('events', []))} events from commodity_markets.json")
print(f"[OK] Loaded {len(commodity_mapping)} commodity mappings")

# Map commodity variations to standard names used in relatedCommodity field
COMMODITY_ALIASES = {
    # Oil variations
    'crude oil': 'oil',
    'crude oil (cl=f)': 'oil',
    'brent oil': 'oil',
    'brent oil (bz=f)': 'oil',
    'natural gas': 'oil',
    'natural gas (ng=f)': 'oil',
    'heating oil': 'oil',
    'heating oil (ho=f)': 'oil',
    'rbob gasoline': 'oil',
    'rbob gasoline (rb=f)': 'oil',
    # Gold variations
    'gold': 'gold',
    'gold (gc=f)': 'gold',
    # Silver variations
    'silver': 'silver',
    'silver (si=f)': 'silver',
    # USD Index variations
    'usd index': 'usd index',
    'usd index (dx-y.nyb)': 'usd index',
    'dx-y.nyb': 'usd index',
}


def calculate_market_greeks(yes_price: float, days_to_expiry: int) -> Greeks:
    """
    Calculate Greeks for a Polymarket market using simplified Black-Scholes Digital model.
    
    Args:
        yes_price: Price of YES outcome (0-1)
        days_to_expiry: Days until market expiration
    
    Returns:
        Greeks object with delta, gamma, vega, theta, rho
    """
    # Convert days to years
    T = max(days_to_expiry / 365.0, 0.001)  # Minimum 0.001 years
    
    # Use a default volatility estimate (can be improved with historical data)
    sigma = 0.5  # 50% annualized volatility
    
    # Calculate Greeks using Black-Scholes Digital model
    greeks = bs_calculator.greeks(p=yes_price, sigma=sigma, T=T, is_yes=True)
    
    return greeks


def normalize_commodity_name(commodity):
    """Normalize commodity name for comparison"""
    return commodity.lower().strip()


def get_commodity_search_term(commodity_input):
    """
    Convert commodity input to search term that matches relatedCommodity field.
    Handles variations like "Crude Oil (CL=F)" -> "oil"
    """
    normalized = normalize_commodity_name(commodity_input)
    
    # Check aliases first
    if normalized in COMMODITY_ALIASES:
        return COMMODITY_ALIASES[normalized]
    
    # Extract base name from ticker format (e.g., "Crude Oil (CL=F)" -> "crude oil")
    if '(' in normalized and ')' in normalized:
        base_name = normalized.split('(')[0].strip()
        if base_name in COMMODITY_ALIASES:
            return COMMODITY_ALIASES[base_name]
        # Try without common words
        base_name_clean = base_name.replace(' crude', '').replace(' brent', '')
        if base_name_clean in COMMODITY_ALIASES:
            return COMMODITY_ALIASES[base_name_clean]
    
    # Check if it contains keywords
    if 'oil' in normalized or 'crude' in normalized or 'brent' in normalized:
        return 'oil'
    if 'gold' in normalized:
        return 'gold'
    if 'silver' in normalized:
        return 'silver'
    if 'usd' in normalized and 'index' in normalized:
        return 'usd index'
    
    # Return normalized as-is if no match
    return normalized


def find_markets_by_commodity(commodity_name):
    """
    Find markets related to a specific commodity.
    Returns markets where relatedCommodity matches the commodity name.
    Handles variations like "Crude Oil (CL=F)" -> searches for "oil"
    """
    # Convert input to search term (e.g., "Crude Oil (CL=F)" -> "oil")
    search_term = get_commodity_search_term(commodity_name)
    matching_events = []
    
    for event in markets_data.get('events', []):
        related_commodity = event.get('relatedCommodity', '')
        
        if related_commodity and normalize_commodity_name(related_commodity) == search_term:
            # Extract relevant market information
            event_info = {
                'id': event.get('id'),
                'ticker': event.get('ticker'),
                'slug': event.get('slug'),
                'title': event.get('title'),
                'description': event.get('description'),
                'startDate': event.get('startDate'),
                'endDate': event.get('endDate'),
                'image': event.get('image'),
                'active': event.get('active'),
                'liquidity': event.get('liquidity'),
                'volume': event.get('volume'),
                'relatedCommodity': event.get('relatedCommodity'),
                'markets': []
            }
            
            # Add individual markets within the event
            for market in event.get('markets', []):
                market_info = {
                    'id': market.get('id'),
                    'question': market.get('question'),
                    'slug': market.get('slug'),
                    'endDate': market.get('endDate'),
                    'liquidity': float(market.get('liquidity', 0)) if market.get('liquidity') else 0,
                    'volume': float(market.get('volume', 0)) if market.get('volume') else 0,
                    'active': market.get('active'),
                    'outcomePrices': market.get('outcomePrices'),
                    'outcomes': market.get('outcomes'),
                }
                
                # Parse outcome prices
                try:
                    if market_info['outcomePrices']:
                        prices = json.loads(market_info['outcomePrices'])
                        market_info['yesPrice'] = float(prices[0]) if len(prices) > 0 else 0.5
                        market_info['noPrice'] = float(prices[1]) if len(prices) > 1 else 0.5
                    else:
                        market_info['yesPrice'] = 0.5
                        market_info['noPrice'] = 0.5
                except (json.JSONDecodeError, ValueError, TypeError):
                    market_info['yesPrice'] = 0.5
                    market_info['noPrice'] = 0.5
                
                event_info['markets'].append(market_info)
            
            matching_events.append(event_info)
    
    return matching_events


def get_correlated_commodity(commodity_name):
    """
    Get the correlated main commodity for a given commodity.
    Returns the main commodity it maps to, or None if it's already a main commodity.
    """
    # Check if commodity exists in mapping
    for key, value in commodity_mapping.items():
        if normalize_commodity_name(key) == normalize_commodity_name(commodity_name):
            # If it maps to itself, it's a main commodity
            if normalize_commodity_name(key) == normalize_commodity_name(value):
                return None
            # Otherwise return the mapped commodity
            return value
    
    # If not in mapping, return None
    return None


@app.route('/api/search-markets', methods=['POST'])
def search_markets():
    """
    Search for markets related to a commodity.
    
    Request body:
    {
        "commodity": "Gold (GC=F)" or "gold" or "Wheat (ZW=F)"
    }
    
    Response:
    {
        "commodity": "gold",
        "directResults": [...],  // Markets directly related to the commodity
        "correlatedCommodity": "crude oil",  // The correlated commodity (if applicable)
        "correlatedResults": [...],  // Markets related to the correlated commodity
        "message": "..."  // Information message about the search
    }
    """
    try:
        data = request.get_json()
        commodity = data.get('commodity', '').strip()
        
        if not commodity:
            return jsonify({'error': 'Commodity parameter is required'}), 400
        
        # Use the original commodity name for search (will be normalized internally)
        # Search for direct markets using the smart alias matching
        direct_results = find_markets_by_commodity(commodity)
        
        # Check if we need to find correlated commodity
        correlated_commodity = get_correlated_commodity(commodity)
        correlated_results = []
        message = ""
        
        if direct_results:
            # Found direct markets
            message = f"Found {len(direct_results)} event(s) directly related to {commodity}."
        else:
            # No direct markets, try correlated commodity
            if correlated_commodity:
                # Search for correlated commodity markets (handles aliases automatically)
                correlated_results = find_markets_by_commodity(correlated_commodity)
                
                if correlated_results:
                    message = (
                        f"No markets found directly related to {commodity} on Polymarket. "
                        f"However, here are {len(correlated_results)} event(s) related to {correlated_commodity}, "
                        f"which is the commodity most correlated with {commodity}."
                    )
                else:
                    message = (
                        f"No markets found for {commodity} or its correlated commodity {correlated_commodity}."
                    )
            else:
                message = f"No markets found related to {commodity} on Polymarket."
        
        # Flatten events into individual markets for easier frontend consumption
        direct_markets = []
        for event in direct_results:
            for market in event['markets']:
                direct_markets.append({
                    **market,
                    'eventTitle': event['title'],
                    'eventId': event['id'],
                    'eventImage': event['image'],
                    'relatedCommodity': event['relatedCommodity']
                })
        
        correlated_markets = []
        for event in correlated_results:
            for market in event['markets']:
                correlated_markets.append({
                    **market,
                    'eventTitle': event['title'],
                    'eventId': event['id'],
                    'eventImage': event['image'],
                    'relatedCommodity': event['relatedCommodity']
                })
        
        response = {
            'commodity': commodity,
            'directResults': direct_markets[:20],  # Limit to top 20 markets
            'correlatedCommodity': correlated_commodity,
            'correlatedResults': correlated_markets[:20],  # Limit to top 20 markets
            'message': message
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"Error in search_markets: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/portfolio/initial-greeks', methods=['POST'])
def calculate_initial_greeks():
    """
    Calculate initial portfolio Greeks from commodities with quantities.
    
    Request body:
    {
        "commodities": [
            {"commodity": "Gold (GC=F)", "quantity": 10000},
            {"commodity": "Silver (SI=F)", "quantity": 5000}
        ]
    }
    
    Response:
    {
        "greeks": {
            "delta": 258.5919,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        }
    }
    """
    try:
        data = request.get_json()
        commodities = data.get('commodities', [])
        
        if not commodities:
            return jsonify({'error': 'Commodities array is required'}), 400
        
        # Validate input format
        if not isinstance(commodities, list):
            return jsonify({'error': 'Commodities must be an array'}), 400
        
        for item in commodities:
            if not isinstance(item, dict) or 'commodity' not in item or 'quantity' not in item:
                return jsonify({'error': 'Each commodity must have "commodity" and "quantity" fields'}), 400
        
        # Calculate Greeks using portfolio_to_greeks
        greeks = calculate_portfolio_greeks(commodities)
        
        return jsonify({
            'greeks': greeks
        }), 200
        
    except Exception as e:
        print(f"Error in calculate_initial_greeks: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/strategy/optimize', methods=['POST'])
def optimize_portfolio_strategy():
    """
    Optimize portfolio strategy using AI to achieve target Greeks.
    
    Request body:
    {
        "commodities": [
            {"commodity": "Gold (GC=F)", "quantity": 10000},
            ...
        ],
        "target_greeks": {
            "delta": 300.0,
            "gamma": 0.5,
            "vega": 1.0,
            "theta": -0.5
        },
        "max_budget": 5000.0,
        "selected_commodities": ["Gold (GC=F)", "Silver (SI=F)"]
    }
    
    Response:
    {
        "success": bool,
        "optimal_positions": [...],
        "achieved_greeks": {...},
        "target_greeks": {...},
        "deviations": {...},
        "total_investment": float,
        "num_positions": int,
        "initial_greeks": {...},
        "greek_changes_from_initial": {...}
    }
    """
    try:
        data = request.get_json()
        commodities = data.get('commodities', [])
        target_greeks = data.get('target_greeks', {})
        max_budget = float(data.get('max_budget', 10000.0))
        selected_commodities = data.get('selected_commodities', [])
        
        if not target_greeks:
            return jsonify({'error': 'Target Greeks are required'}), 400
        
        if max_budget <= 0:
            return jsonify({'error': 'Max budget must be positive'}), 400
        
        # Calculate initial portfolio Greeks
        initial_greeks = None
        if commodities:
            try:
                initial_greeks = calculate_portfolio_greeks(commodities)
            except Exception as e:
                print(f"Warning: Could not calculate initial Greeks: {e}")
        
        # Get markets for selected commodities
        all_markets_with_greeks = []
        
        for commodity in selected_commodities:
            # Search for markets related to this commodity
            direct_results = find_markets_by_commodity(commodity)
            
            # Process each event and its markets
            for event in direct_results:
                for market in event.get('markets', []):
                    try:
                        # Calculate Greeks for this market
                        yes_price = market.get('yesPrice', 0.5)
                        no_price = market.get('noPrice', 0.5)
                        
                        # Calculate days to expiry
                        end_date_str = market.get('endDate')
                        if end_date_str:
                            try:
                                if isinstance(end_date_str, str):
                                    if end_date_str.endswith('Z'):
                                        end_date_str = end_date_str[:-1] + '+00:00'
                                    from datetime import datetime
                                    end_date = datetime.fromisoformat(end_date_str)
                                    days_to_expiry = max(1, (end_date - datetime.now(end_date.tzinfo)).days)
                                else:
                                    days_to_expiry = 30
                            except:
                                days_to_expiry = 30
                        else:
                            days_to_expiry = 30
                        
                        # Create PolymarketMarket object for Greeks calculation
                        pm_market = PolymarketMarket(
                            id=market.get('id', ''),
                            question=market.get('question', ''),
                            condition_id='',
                            slug=market.get('slug', ''),
                            end_date=None,  # Not needed for Greeks calc
                            outcome_prices=[yes_price, no_price],
                            clob_token_ids=[],
                            outcomes=['Yes', 'No'],
                            volume=float(market.get('volume', 0)),
                            liquidity=float(market.get('liquidity', 0)),
                            related_commodity=event.get('relatedCommodity')
                        )
                        
                        # Calculate Greeks
                        greeks = calculate_market_greeks(yes_price, days_to_expiry)
                        
                        # Add market with Greeks to list
                        market_with_greeks = {
                            'id': market.get('id'),
                            'question': market.get('question'),
                            'yes_price': yes_price,
                            'no_price': no_price,
                            'expiry_days': days_to_expiry,
                            'delta': greeks.delta,
                            'gamma': greeks.gamma,
                            'vega': greeks.vega,
                            'theta': greeks.theta,
                            'liquidity': float(market.get('liquidity', 0)),
                            'volume': float(market.get('volume', 0)),
                            'relatedCommodity': event.get('relatedCommodity'),
                            'eventTitle': event.get('title'),
                            'slug': market.get('slug')
                        }
                        
                        all_markets_with_greeks.append(market_with_greeks)
                        
                    except Exception as e:
                        print(f"Error calculating Greeks for market {market.get('id')}: {e}")
                        continue
        
        if not all_markets_with_greeks:
            return jsonify({
                'success': False,
                'error': 'No markets found for optimization',
                'optimal_positions': [],
                'achieved_greeks': {},
                'target_greeks': target_greeks
            }), 200
        
        print(f"Found {len(all_markets_with_greeks)} markets for optimization")
        
        # Run optimization
        result = optimize_strategy(
            markets_with_greeks=all_markets_with_greeks,
            target_greeks=target_greeks,
            max_budget=max_budget,
            initial_greeks=initial_greeks
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"Error in optimize_portfolio_strategy: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'total_events': len(markets_data.get('events', [])),
        'total_commodities_in_mapping': len(commodity_mapping)
    }), 200


if __name__ == '__main__':
    print("\n[START] Starting Polymarket Portfolio Greeks Backend...")
    print(f"[DATA] Markets data loaded from: {COMMODITY_MARKETS_PATH}")
    print(f"[DATA] Commodity mapping loaded from: {COMMODITY_MAPPING_PATH}")
    print("\n[READY] Server ready on http://localhost:5000")
    print("[API] API Endpoints:")
    print("   - POST /api/search-markets")
    print("   - POST /api/portfolio/initial-greeks")
    print("   - POST /api/strategy/optimize")
    print("   - GET  /api/health\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
