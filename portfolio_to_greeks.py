import pandas as pd
from typing import List, Dict
import os

# --- CONFIGURATION ---
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CORRELATION_FILE = os.path.join(SCRIPT_DIR, "commodity_vs_core_assets_correlations.csv")

# Core assets that we calculate exposure to
CORE_ASSETS = [
    "Gold (GC=F)",
    "Silver (SI=F)",
    "Crude Oil (CL=F)",
    "Bitcoin (BTC-USD)",
    "Ethereum (ETH-USD)",
    "USD Index (DX-Y.NYB)"
]

def calculate_portfolio_greeks(commodities_with_quantities: List[Dict[str, any]]) -> Dict[str, float]:
    """
    Calculate portfolio Greeks from initial commodity positions.
    
    Args:
        commodities_with_quantities: List of dicts with 'commodity' (str) and 'quantity' (float in USD)
            Example: [
                {"commodity": "Gold (GC=F)", "quantity": 10000},
                {"commodity": "Silver (SI=F)", "quantity": 5000}
            ]
    
    Returns:
        Dictionary with Greeks: {"delta": float, "gamma": float, "vega": float, "theta": float, "rho": float}
    """
    # Load correlation matrix
    try:
        correlation_df = pd.read_csv(CORRELATION_FILE, index_col=0)
        correlation_df.index = correlation_df.index.str.strip()
        correlation_df.columns = correlation_df.columns.str.strip()
    except Exception as e:
        raise FileNotFoundError(f"Failed to load correlation file: {e}")
    
    if not commodities_with_quantities:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        }
    
    # Build portfolio dictionary: {commodity_name: quantity_in_usd}
    portfolio = {}
    for item in commodities_with_quantities:
        commodity = item.get("commodity", "").strip()
        quantity = float(item.get("quantity", 0))
        
        if commodity and quantity != 0:
            # Use commodity name as-is (should match CSV row names)
            portfolio[commodity] = quantity
    
    if not portfolio:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        }
    
    # Get commodities that exist in correlation matrix
    available_commodities = [c for c in portfolio.keys() if c in correlation_df.index]
    
    if not available_commodities:
        # Try to find commodities by partial match or return zeros
        print(f"Warning: None of the commodities found in correlation matrix: {list(portfolio.keys())}")
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        }
    
    # Select correlation rows for commodities in portfolio
    relevant_corrs = correlation_df.loc[available_commodities]
    
    # Get positions (quantities in USD) for available commodities
    positions = pd.Series({c: portfolio[c] for c in available_commodities})
    
    # Calculate exposure to each core asset
    # Delta = sum(quantity * correlation) for each core asset
    # This represents how much the portfolio moves with each core asset
    exposure_to_core_assets = {}
    
    for core_asset in CORE_ASSETS:
        if core_asset in relevant_corrs.columns:
            # Multiply correlation by position size, then sum
            exposure = (relevant_corrs[core_asset] * positions).sum()
            exposure_to_core_assets[core_asset] = exposure
    
    # Calculate total portfolio Delta
    # Delta represents overall directional exposure
    # We sum exposures to all core assets (weighted by their importance)
    # For simplicity, we'll use a weighted average or sum
    total_delta = sum(exposure_to_core_assets.values())
    
    # Normalize delta by total portfolio value for better interpretation
    total_portfolio_value = sum(positions.values)
    if total_portfolio_value > 0:
        # Delta as percentage of portfolio value
        normalized_delta = total_delta / total_portfolio_value
        # Scale back to dollar terms (multiply by portfolio value)
        # This gives us delta in $ per 1% move
        portfolio_delta = normalized_delta * total_portfolio_value * 0.01
    else:
        portfolio_delta = 0.0
    
    # For now, set other Greeks to 0 (can be enhanced later)
    # Gamma: rate of change of delta
    portfolio_gamma = 0.0
    
    # Vega: sensitivity to volatility (not applicable for spot positions)
    portfolio_vega = 0.0
    
    # Theta: time decay (not applicable for spot positions)
    portfolio_theta = 0.0
    
    # Rho: sensitivity to interest rates
    portfolio_rho = 0.0
    
    return {
        "delta": float(round(portfolio_delta, 4)),
        "gamma": float(round(portfolio_gamma, 4)),
        "vega": float(round(portfolio_vega, 4)),
        "theta": float(round(portfolio_theta, 4)),
        "rho": float(round(portfolio_rho, 4))
    }


def calculate_portfolio_greeks_detailed(commodities_with_quantities: List[Dict[str, any]]) -> Dict:
    """
    Calculate detailed portfolio Greeks with breakdown by commodity.
    
    Returns:
        Dictionary with:
        - "greeks": Total portfolio Greeks
        - "breakdown": Per-commodity breakdown
        - "exposure": Exposure to each core asset
    """
    # Load correlation matrix
    try:
        correlation_df = pd.read_csv(CORRELATION_FILE, index_col=0)
        correlation_df.index = correlation_df.index.str.strip()
        correlation_df.columns = correlation_df.columns.str.strip()
    except Exception as e:
        raise FileNotFoundError(f"Failed to load correlation file: {e}")
    
    if not commodities_with_quantities:
        return {
            "greeks": {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0
            },
            "breakdown": {},
            "exposure": {}
        }
    
    # Build portfolio
    portfolio = {}
    for item in commodities_with_quantities:
        commodity = item.get("commodity", "").strip()
        quantity = float(item.get("quantity", 0))
        if commodity and quantity != 0:
            portfolio[commodity] = quantity
    
    if not portfolio:
        return {
            "greeks": {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0
            },
            "breakdown": {},
            "exposure": {}
        }
    
    # Get available commodities
    available_commodities = [c for c in portfolio.keys() if c in correlation_df.index]
    positions = pd.Series({c: portfolio[c] for c in available_commodities})
    relevant_corrs = correlation_df.loc[available_commodities]
    
    # Calculate exposure to core assets
    exposure_to_core_assets = {}
    for core_asset in CORE_ASSETS:
        if core_asset in relevant_corrs.columns:
            exposure = (relevant_corrs[core_asset] * positions).sum()
            exposure_to_core_assets[core_asset] = float(exposure)
    
    # Calculate total delta
    total_delta = sum(exposure_to_core_assets.values())
    total_portfolio_value = sum(positions.values)
    
    if total_portfolio_value > 0:
        normalized_delta = total_delta / total_portfolio_value
        portfolio_delta = normalized_delta * total_portfolio_value * 0.01
    else:
        portfolio_delta = 0.0
    
    # Per-commodity breakdown
    breakdown = {}
    for commodity in available_commodities:
        commodity_exposure = {}
        for core_asset in CORE_ASSETS:
            if core_asset in relevant_corrs.columns:
                exposure = relevant_corrs.loc[commodity, core_asset] * positions[commodity]
                commodity_exposure[core_asset] = float(exposure)
        
        breakdown[commodity] = {
            "quantity": float(positions[commodity]),
            "exposure": commodity_exposure,
            "total_exposure": float(sum(commodity_exposure.values()))
        }
    
    return {
        "greeks": {
            "delta": float(round(portfolio_delta, 4)),
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "rho": 0.0
        },
        "breakdown": breakdown,
        "exposure": exposure_to_core_assets
    }


# Legacy function for command-line use (kept for backward compatibility)
def get_user_inputs():
    """Legacy function - kept for backward compatibility."""
    user_portfolio = {}
    
    print("--- Enter Your Positions ---")
    print("Format: Commodity Amount, Commodity Amount, ...")
    print("Example: Gold (GC=F) 1000, Silver (SI=F) 5000")
    print("\n")
    
    user_input = input("Enter positions: ").strip()
    
    if not user_input:
        print("No positions entered.")
        return []
    
    commodities_with_quantities = []
    entries = user_input.split(',')
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.rsplit(maxsplit=1)
        if len(parts) != 2:
            print(f"Invalid format: '{entry}' - skipping")
            continue
        commodity_name, amount_str = parts
        commodity_name = commodity_name.strip()
        try:
            amount = float(amount_str)
            if amount != 0:
                commodities_with_quantities.append({
                    "commodity": commodity_name,
                    "quantity": amount
                })
        except ValueError:
            print(f"Invalid amount for {commodity_name}: '{amount_str}' - skipping")
    
    return commodities_with_quantities


def run_analysis():
    """Legacy command-line interface."""
    commodities_with_quantities = get_user_inputs()
    if not commodities_with_quantities:
        print("Portfolio is empty.")
        return
    
    result = calculate_portfolio_greeks_detailed(commodities_with_quantities)
    
    print("\n--- PORTFOLIO GREEKS ---")
    print(result["greeks"])
    
    print("\n--- EXPOSURE TO CORE ASSETS ---")
    for asset, exposure in result["exposure"].items():
        print(f"{asset}: {exposure:.4f}")
    
    print("\n--- PER-COMMODITY BREAKDOWN ---")
    for commodity, data in result["breakdown"].items():
        print(f"\n{commodity}:")
        print(f"  Quantity: ${data['quantity']:,.2f}")
        print(f"  Total Exposure: {data['total_exposure']:.4f}")


if __name__ == "__main__":
    run_analysis()
