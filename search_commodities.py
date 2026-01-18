import requests
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

# Helper function to add events from a search result
def add_events_from_search(search_results, seen_event_ids, merged_events, search_name=""):
    """Add unique events from search results to merged_events list."""
    count = 0
    skipped_count = 0
    duplicate_count = 0
    
    if "events" in search_results:
        events_list = search_results["events"]
        print(f"  Processing {len(events_list)} events from {search_name}...")
        
        for event in events_list:
            event_id = event.get("id")
            title = event.get("title", "")
            
            # Skip events with "golden" or "Golden" in title (football teams, etc.)
            if "golden" in title.lower():
                skipped_count += 1
                continue
            
            if event_id:
                # Add ALL events, even if duplicate across searches (deduplicate only within each search)
                if event_id not in seen_event_ids:
                    seen_event_ids.add(event_id)
                    # Add related commodities field
                    event["relatedCommodities"] = detect_related_commodities(event)
                    merged_events.append(event)
                    count += 1
                else:
                    duplicate_count += 1
            else:
                # If no ID, still add it (shouldn't happen but be safe)
                event["relatedCommodities"] = detect_related_commodities(event)
                merged_events.append(event)
                count += 1
    
    if skipped_count > 0:
        print(f"    Skipped {skipped_count} events with 'golden' in title")
    if duplicate_count > 0:
        print(f"    Skipped {duplicate_count} duplicate events (already in merged list)")
    
    return count

# Function to fetch search results from API (single call - API returns all results)
def fetch_search_results(url):
    """Fetch search results from Polymarket API. Returns all results in a single call."""
    try:
        print(f"  Calling API: {url}")
        response = requests.get(url)
        response.raise_for_status()
        results = response.json()
        
        events = results.get("events", [])
        pagination = results.get("pagination", {})
        total_results = pagination.get("totalResults", len(events))
        
        print(f"  Retrieved {len(events)} events (API reports {total_results} total)")
        
        return results
        
    except Exception as e:
        print(f"  Error fetching results: {e}")
        import traceback
        traceback.print_exc()
        return {
            "events": [],
            "pagination": {
                "hasMore": False,
                "totalResults": 0
            }
        }

# Make the 4 API calls as specified (API returns all results in a single call)
print("Fetching USD Index events...")
usd_index_url = "https://gamma-api.polymarket.com/public-search?q=usd%20index&events_status=active"
usd_index_results = fetch_search_results(usd_index_url)

print("\nFetching Gold events...")
gold_url = "https://gamma-api.polymarket.com/public-search?q=gold&events_status=active"
gold_results = fetch_search_results(gold_url)

print("\nFetching Silver events...")
silver_url = "https://gamma-api.polymarket.com/public-search?q=silver&events_status=active"
silver_results = fetch_search_results(silver_url)

print("\nFetching Oil events...")
oil_url = "https://gamma-api.polymarket.com/public-search?q=oil&events_status=active"
oil_results = fetch_search_results(oil_url)

# Merge results - combine events from all searches
# Use a set to track event IDs to avoid duplicates
seen_event_ids = set()
merged_events = []

# Add events from all searches
print("\n=== Processing Events ===")
print(f"USD Index: {len(usd_index_results.get('events', []))} events from API")
usd_count = add_events_from_search(usd_index_results, seen_event_ids, merged_events, "USD Index")

print(f"Gold: {len(gold_results.get('events', []))} events from API")
gold_count = add_events_from_search(gold_results, seen_event_ids, merged_events, "Gold")

print(f"Silver: {len(silver_results.get('events', []))} events from API")
silver_count = add_events_from_search(silver_results, seen_event_ids, merged_events, "Silver")

print(f"Oil: {len(oil_results.get('events', []))} events from API")
oil_count = add_events_from_search(oil_results, seen_event_ids, merged_events, "Oil")

# Create merged results structure
merged_results = {
    "events": merged_events
}

# Copy other fields from the first search result if they exist
for key in usd_index_results:
    if key != "events":
        merged_results[key] = usd_index_results[key]

# Save results to JSON file
with open("commodity_markets.json", "w", encoding="utf-8") as f:
    json.dump(merged_results, f, indent=2, ensure_ascii=False)

print(f"\n=== Summary ===")
print(f"USD Index search: {len(usd_index_results.get('events', []))} total events - {usd_count} unique added")
print(f"Gold search: {len(gold_results.get('events', []))} total events - {gold_count} unique added")
print(f"Silver search: {len(silver_results.get('events', []))} total events - {silver_count} unique added")
print(f"Oil search: {len(oil_results.get('events', []))} total events - {oil_count} unique added")
print(f"Total merged events: {len(merged_events)}")
print(f"Search results saved to commodity_markets.json")
