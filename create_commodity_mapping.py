import csv
import json

# Read the CSV file
commodity_to_main_asset = {}

with open('commodity_vs_core_assets_correlations.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    # Get the main asset column names (excluding the first column which is the commodity name)
    main_assets = [col for col in reader.fieldnames if col != '']
    
    for row in reader:
        commodity_name = row['']  # The first column (commodity name)
        if not commodity_name:
            continue
            
        # Find the main asset with the highest absolute correlation
        max_corr = -1
        best_main_asset = None
        
        for main_asset in main_assets:
            try:
                corr_value = abs(float(row[main_asset]))
                if corr_value > max_corr:
                    max_corr = corr_value
                    best_main_asset = main_asset
            except (ValueError, KeyError):
                continue
        
        if best_main_asset:
            commodity_to_main_asset[commodity_name] = best_main_asset

# Save to JSON file
with open('commodity_to_main_asset_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(commodity_to_main_asset, f, indent=2, ensure_ascii=False)

print(f"Created mapping for {len(commodity_to_main_asset)} commodities")
print(f"Mapping saved to commodity_to_main_asset_mapping.json")
print("\nSample mappings:")
for i, (commodity, main_asset) in enumerate(list(commodity_to_main_asset.items())[:5]):
    print(f"  {commodity} -> {main_asset}")
