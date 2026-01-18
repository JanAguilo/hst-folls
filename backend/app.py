from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load data files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMODITY_MARKETS_PATH = os.path.join(BASE_DIR, 'commodity_markets.json')
COMMODITY_MAPPING_PATH = os.path.join(BASE_DIR, 'commodity_to_main_asset_mapping.json')

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
    print("   - GET  /api/health\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
