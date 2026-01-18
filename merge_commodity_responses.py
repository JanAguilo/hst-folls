import json
import re

def detect_related_commodities(event):
    """
    Detect which commodities (gold, silver, oil, usd index) an event is related to.
    Checks title, description, ticker, tags, and market questions.
    Returns a list of commodity names.
    """
    related = set()
    
    # Get text fields to search
    title = event.get("title", "").lower()
    description = event.get("description", "").lower()
    ticker = event.get("ticker", "").lower()
    
    # Check tags
    tags_text = ""
    if "tags" in event:
        tags_text = " ".join([tag.get("label", "").lower() for tag in event.get("tags", [])])
    
    # Combine all searchable text
    searchable_text = f"{title} {description} {ticker} {tags_text}"
    
    # Check markets questions and descriptions
    if "markets" in event:
        for market in event["markets"]:
            question = market.get("question", "").lower()
            market_desc = market.get("description", "").lower()
            searchable_text += f" {question} {market_desc}"
    
    # Patterns for each commodity
    # Gold patterns: "gold", "gc", "gc)", but avoid "golden" in team names
    if re.search(r'\b(gold|gc)\b', searchable_text, re.IGNORECASE):
        # Avoid false positives like "golden knights", "golden state"
        if not re.search(r'(golden knights|golden state)', searchable_text, re.IGNORECASE):
            related.add("gold")
    
    # Silver patterns: "silver", "si", "si)"
    if re.search(r'\b(silver|si)\b', searchable_text, re.IGNORECASE):
        related.add("silver")
    
    # Crude Oil patterns: "crude oil", "oil", "cl", "cl)", "wti", "brent", "nymex crude"
    if re.search(r'\b(crude oil|oil|cl\b|wti|brent|nymex crude)', searchable_text, re.IGNORECASE):
        # Avoid false positives like "oil" in "Oilers" team name
        if not re.search(r'\boilers?\b', searchable_text, re.IGNORECASE):
            related.add("oil")
    
    # USD Index patterns: "usd index", "dx", "dx)", "us dollar index", "dxy"
    if re.search(r'\b(usd index|dx\b|us dollar index|dxy)', searchable_text, re.IGNORECASE):
        related.add("usd index")
    
    return sorted(list(related))

import os

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load all JSON files
print("Loading response files...")

print("  Loading usd-index.json...")
usd_index_path = os.path.join(script_dir, "comm_responses", "usd-index.json")
with open(usd_index_path, "r", encoding="utf-8") as f:
    usd_index_data = json.load(f)

print("  Loading gold.json...")
gold_path = os.path.join(script_dir, "comm_responses", "gold.json")
with open(gold_path, "r", encoding="utf-8") as f:
    gold_data = json.load(f)

print("  Loading silver.json...")
silver_path = os.path.join(script_dir, "comm_responses", "silver.json")
with open(silver_path, "r", encoding="utf-8") as f:
    silver_data = json.load(f)

print("  Loading oil.json...")
oil_path = os.path.join(script_dir, "comm_responses", "oil.json")
with open(oil_path, "r", encoding="utf-8") as f:
    oil_data = json.load(f)

# Extract events from each file
usd_index_events = usd_index_data.get("events", [])
gold_events = gold_data.get("events", [])
silver_events = silver_data.get("events", [])
oil_events = oil_data.get("events", [])

print(f"\nOriginal counts:")
print(f"  USD Index: {len(usd_index_events)} events")
print(f"  Gold: {len(gold_events)} events")
print(f"  Silver: {len(silver_events)} events")
print(f"  Oil: {len(oil_events)} events")

# Filter gold events - remove those with "golden" or "Golden" in title
print("\nFiltering gold events...")
filtered_gold_events = []
skipped_gold = 0
for event in gold_events:
    title = event.get("title", "")
    if "golden" in title.lower():
        skipped_gold += 1
        continue
    filtered_gold_events.append(event)

print(f"  Skipped {skipped_gold} gold events with 'golden' in title")
print(f"  Gold after filtering: {len(filtered_gold_events)} events")

# Merge all events and deduplicate by ID
print("\nMerging and deduplicating events...")
seen_event_ids = set()
merged_events = []

# Process each source
sources = [
    ("USD Index", usd_index_events),
    ("Gold", filtered_gold_events),
    ("Silver", silver_events),
    ("Oil", oil_events)
]

for source_name, events in sources:
    added = 0
    duplicates = 0
    # Map source name to commodity value (lowercase, with spaces)
    commodity_map = {
        "USD Index": "usd index",
        "Gold": "gold",
        "Silver": "silver",
        "Oil": "oil"
    }
    commodity_value = commodity_map.get(source_name, source_name.lower())
    
    for event in events:
        event_id = event.get("id")
        if event_id:
            if event_id not in seen_event_ids:
                seen_event_ids.add(event_id)
                # Add relatedCommodity field based on source file
                event["relatedCommodity"] = commodity_value
                # Add related commodities field (detected)
                event["relatedCommodities"] = detect_related_commodities(event)
                merged_events.append(event)
                added += 1
            else:
                duplicates += 1
        else:
            # Events without ID are still added (shouldn't happen but be safe)
            event["relatedCommodity"] = commodity_value
            event["relatedCommodities"] = detect_related_commodities(event)
            merged_events.append(event)
            added += 1
    
    print(f"  {source_name}: Added {added} events ({duplicates} duplicates skipped)")

# Create merged results structure
merged_results = {
    "events": merged_events,
    "pagination": {
        "hasMore": False,
        "totalResults": len(merged_events)
    }
}

# Save to commodity_markets.json
output_path = os.path.join(script_dir, "commodity_markets.json")
print(f"\nSaving {len(merged_events)} events to commodity_markets.json...")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_results, f, indent=2, ensure_ascii=False)

print(f"\n=== Summary ===")
print(f"Total merged events: {len(merged_events)}")
print(f"Search results saved to commodity_markets.json")
