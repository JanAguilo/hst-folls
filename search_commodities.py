import requests
import json

# Search for commodities
commodities_url = "https://gamma-api.polymarket.com/public-search?q=commodities&events_status=active"
commodities_response = requests.get(commodities_url)
commodities_results = commodities_response.json()

# Search for USD index
usd_index_url = "https://gamma-api.polymarket.com/public-search?q=usd%20index&event_status=active"
usd_index_response = requests.get(usd_index_url)
usd_index_results = usd_index_response.json()

# Merge results - combine events from both searches
# Use a set to track event IDs to avoid duplicates
seen_event_ids = set()
merged_events = []

# Add commodities events
if "events" in commodities_results:
    for event in commodities_results["events"]:
        event_id = event.get("id")
        if event_id and event_id not in seen_event_ids:
            seen_event_ids.add(event_id)
            merged_events.append(event)

# Add USD index events
if "events" in usd_index_results:
    for event in usd_index_results["events"]:
        event_id = event.get("id")
        if event_id and event_id not in seen_event_ids:
            seen_event_ids.add(event_id)
            merged_events.append(event)

# Create merged results structure
merged_results = {
    "events": merged_events
}

# Copy other fields from commodities_results if they exist
for key in commodities_results:
    if key != "events":
        merged_results[key] = commodities_results[key]

# Save results to JSON file
with open("commodity_markets.json", "w", encoding="utf-8") as f:
    json.dump(merged_results, f, indent=2, ensure_ascii=False)

print(f"Commodities search: {len(commodities_results.get('events', []))} events (status: {commodities_response.status_code})")
print(f"USD Index search: {len(usd_index_results.get('events', []))} events (status: {usd_index_response.status_code})")
print(f"Total merged events: {len(merged_events)}")
print(f"Search results saved to commodity_markets.json")
