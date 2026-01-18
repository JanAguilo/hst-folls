import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

# ========== PARAMETERS ==========
start_date = "2015-01-01"  # Crypto data only goes back to ~2014
end_date = "2025-01-17"

# ========== BASKET 1: Polymarket Portfolio Basket ==========
# These are the core assets we'll correlate against
core_portfolio_assets = {
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F",
    "Crude Oil (CL=F)": "CL=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "USD Index (DX-Y.NYB)": "DX-Y.NYB"
}

# Additional fiat currencies for context
fiat_currencies = {
    "EUR/USD": "EURUSD=X",
    "JPY/USD": "JPY=X",
    "GBP/USD": "GBPUSD=X",
    "CNY/USD": "CNY=X",
    "CHF/USD": "CHF=X",
}

# ========== BASKET 2: All Actively Traded Commodity Futures ==========
commodity_futures_tickers = {
    # Precious Metals
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F",
    "Platinum (PL=F)": "PL=F",
    "Palladium (PA=F)": "PA=F",
    
    # Energy
    "Crude Oil (CL=F)": "CL=F",
    "Brent Oil (BZ=F)": "BZ=F",
    "Natural Gas (NG=F)": "NG=F",
    "Heating Oil (HO=F)": "HO=F",
    "RBOB Gasoline (RB=F)": "RB=F",
    
    # Base Metals
    "Copper (HG=F)": "HG=F",
    "Aluminum (ALI=F)": "ALI=F",
    
    # Agriculture - Grains
    "Corn (ZC=F)": "ZC=F",
    "Wheat (ZW=F)": "ZW=F",
    "Soybeans (ZS=F)": "ZS=F",
    "Soybean Oil (ZL=F)": "ZL=F",
    "Soybean Meal (ZM=F)": "ZM=F",
    "Oats (ZO=F)": "ZO=F",
    "Rice (ZR=F)": "ZR=F",
    
    # Agriculture - Softs
    "Coffee (KC=F)": "KC=F",
    "Sugar (SB=F)": "SB=F",
    "Cotton (CT=F)": "CT=F",
    "Cocoa (CC=F)": "CC=F",
    "Orange Juice (OJ=F)": "OJ=F",
    
    # Livestock
    "Live Cattle (LE=F)": "LE=F",
    "Feeder Cattle (GF=F)": "GF=F",
    "Lean Hogs (HE=F)": "HE=F",
    
    # Lumber
    "Lumber (LBS=F)": "LBS=F"
}

# ========== SAFE DOWNLOAD ==========
def safe_download(tickers_dict, start, end):
    """Download adjusted close prices robustly."""
    data = yf.download(
        list(tickers_dict.values()), 
        start=start, 
        end=end,
        progress=False, 
        group_by='ticker', 
        auto_adjust=True
    )
    close_prices = pd.DataFrame()

    for name, symbol in tickers_dict.items():
        try:
            if symbol in data.columns.get_level_values(0):
                close_prices[name] = data[symbol]["Close"]
            elif "Close" in data.columns:
                # Single ticker case
                close_prices[name] = data["Close"]
            else:
                print(f"⚠️ Skipping {name}: no Close data found")
        except Exception as e:
            print(f"⚠️ Skipping {name}: {e}")
    
    return close_prices.dropna(how="all")

# ========== DOWNLOAD DATA ==========
print(f"📥 Downloading Core Portfolio Assets ({len(core_portfolio_assets)} assets)...")
core_data = safe_download(core_portfolio_assets, start_date, end_date)
print(f"✅ Downloaded {len(core_data.columns)} core assets.\n")

print(f"📥 Downloading Fiat Currencies ({len(fiat_currencies)} pairs)...")
fiat_data = safe_download(fiat_currencies, start_date, end_date)
print(f"✅ Downloaded {len(fiat_data.columns)} fiat pairs.\n")

print(f"📥 Downloading Commodity Futures ({len(commodity_futures_tickers)} contracts)...")
commodity_data = safe_download(commodity_futures_tickers, start_date, end_date)
print(f"✅ Downloaded {len(commodity_data.columns)} commodity futures.\n")

# ========== CALCULATE RETURNS ==========
core_returns = np.log(core_data / core_data.shift(1)).dropna()
fiat_returns = np.log(fiat_data / fiat_data.shift(1)).dropna()
commodity_returns = np.log(commodity_data / commodity_data.shift(1)).dropna()

# ========== INDIVIDUAL CORRELATIONS ==========
print("\n" + "="*80)
print("📊 INDIVIDUAL CORRELATIONS: Each Commodity vs Each Core Asset")
print("="*80 + "\n")

# Create correlation matrix: commodities (rows) vs core assets (columns)
correlation_matrix = pd.DataFrame(
    index=commodity_returns.columns,
    columns=core_returns.columns
)

# Calculate correlations
for commodity in commodity_returns.columns:
    for core_asset in core_returns.columns:
        correlation_matrix.loc[commodity, core_asset] = \
            commodity_returns[commodity].corr(core_returns[core_asset])

# Convert to float
correlation_matrix = correlation_matrix.astype(float)

# Display full correlation matrix
print("CORRELATION MATRIX (Commodities vs Core Assets):")
print(correlation_matrix.round(3))
print("\n")

# ========== TOP 10 BY ABSOLUTE VALUE FOR EACH CORE ASSET ==========
for core_asset in core_returns.columns:
    print(f"\n{'='*80}")
    print(f"Top 10 Most Correlated Commodities with {core_asset} (by absolute value)")
    print(f"{'='*80}")
    
    # Get correlations for this core asset
    correlations = correlation_matrix[core_asset].copy()
    
    # Exclude self-correlation if the commodity is also in core assets
    if core_asset in correlations.index:
        correlations = correlations.drop(core_asset)
    
    # Sort by absolute value but keep original sign
    correlations_sorted = correlations.iloc[correlations.abs().argsort()[::-1]]
    
    # Display top 10
    print(correlations_sorted.head(10))
    print("\n")

# ========== VISUALIZATIONS ==========

# 1️⃣ INDIVIDUAL BAR CHARTS for each core asset (sorted by absolute value)
fig, axes = plt.subplots(3, 2, figsize=(18, 20))
axes = axes.flatten()

for idx, core_asset in enumerate(core_returns.columns):
    correlations = correlation_matrix[core_asset].copy()
    
    # Exclude self-correlation
    if core_asset in correlations.index:
        correlations = correlations.drop(core_asset)
    
    # Sort by absolute value
    correlations_sorted = correlations.iloc[correlations.abs().argsort()[::-1]]
    
    colors = ['green' if x > 0 else 'red' for x in correlations_sorted.values]
    
    axes[idx].barh(range(len(correlations_sorted)), correlations_sorted.values, color=colors)
    axes[idx].set_yticks(range(len(correlations_sorted)))
    axes[idx].set_yticklabels(correlations_sorted.index, fontsize=8)
    axes[idx].set_xlabel("Correlation", fontsize=10)
    axes[idx].set_title(f"Correlations with {core_asset} (sorted by |correlation|)", 
                       fontsize=12, fontweight='bold')
    axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    axes[idx].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.show()

# 2️⃣ FULL CORRELATION HEATMAP (all commodities vs all core assets)
plt.figure(figsize=(10, 16))
sns.heatmap(correlation_matrix, 
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"})
plt.title("All Commodities Correlation with Core Portfolio Assets", 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Core Portfolio Assets", fontsize=12)
plt.ylabel("All Commodities", fontsize=12)
plt.tight_layout()
plt.show()

# ========== EXPORT DATA ==========
# Save correlation matrix
correlation_matrix.to_csv('commodity_vs_core_assets_correlations.csv')
print("\n✅ Correlation matrix saved to 'commodity_vs_core_assets_correlations.csv'")

# Save top correlations for each core asset (by absolute value)
with open('top_correlations_summary.txt', 'w') as f:
    for core_asset in core_returns.columns:
        f.write(f"\n{'='*80}\n")
        f.write(f"Top 10 Most Correlated Commodities with {core_asset} (by |correlation|)\n")
        f.write(f"{'='*80}\n\n")
        
        correlations = correlation_matrix[core_asset].copy()
        
        # Exclude self-correlation
        if core_asset in correlations.index:
            correlations = correlations.drop(core_asset)
        
        # Sort by absolute value, keep sign
        correlations_sorted = correlations.iloc[correlations.abs().argsort()[::-1]]
        
        f.write(correlations_sorted.head(10).to_string())
        f.write("\n\n")

print("✅ Top correlations summary saved to 'top_correlations_summary.txt'")

# ========== STATISTICS SUMMARY ==========
print("\n" + "="*80)
print("📈 CORRELATION STATISTICS SUMMARY")
print("="*80 + "\n")

for core_asset in core_returns.columns:
    correlations = correlation_matrix[core_asset].copy()
    
    # Exclude self-correlation
    if core_asset in correlations.index:
        correlations = correlations.drop(core_asset)
    
    print(f"{core_asset}:")
    print(f"  Mean Correlation: {correlations.mean():.3f}")
    print(f"  Mean |Correlation|: {correlations.abs().mean():.3f}")
    print(f"  Std Dev: {correlations.std():.3f}")
    
    # Max by absolute value
    max_abs_idx = correlations.abs().idxmax()
    max_abs_val = correlations[max_abs_idx]
    print(f"  Highest |Correlation|: {max_abs_val:.3f} ({max_abs_idx})")
    
    print()

print("\n✅ Full analysis complete!")