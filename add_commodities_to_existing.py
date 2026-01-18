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

# Read existing file
print("Reading commodity_markets.json...")
with open("commodity_markets.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Process each event
events = data.get("events", [])
updated_count = 0

print(f"Processing {len(events)} events...")
for event in events:
    # Always update/ensure relatedCommodities field exists
    if "relatedCommodities" not in event:
        updated_count += 1
    event["relatedCommodities"] = detect_related_commodities(event)

# Save updated file
print(f"Updating {updated_count} events...")
with open("commodity_markets.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Updated commodity_markets.json with relatedCommodities field")
print(f"  Total events processed: {len(events)}")
print(f"\nSample events with related commodities:")
for i, event in enumerate(events[:5]):
    print(f"  - {event.get('title', 'N/A')[:60]}: {event.get('relatedCommodities', [])}")
