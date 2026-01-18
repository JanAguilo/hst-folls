import requests
import json

url = "https://gamma-api.polymarket.com/public-search?q=commodities&events_status=active"
response = requests.get(url)
results = response.json()

# Save results to JSON file
with open("commodity_markets.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Search results saved to commodities_search.json")
print(f"Response status: {response.status_code}")
