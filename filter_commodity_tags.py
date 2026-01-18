import json

# Commodity keywords to filter tags
COMMODITY_KEYWORDS = [
    # Precious metals
    "gold", "silver", "platinum", "palladium",
    # Energy
    "oil", "crude", "petroleum", "gas", "gasoline", "wti", "brent",
    # Agriculture
    "wheat", "corn", "soybean", "coffee", "sugar", "cocoa",
    # Base metals
    "copper", "aluminum", "steel", "iron",
    # Organizations/Indicators
    "opec", "fed", "federal reserve", "cpi", "inflation", "commodity",
    # Commodity indices
    "dxy", "dollar index", "crb", "commodity index",
    # Economy-related (often related to commodities)
    "economy", "economic", "energy", "interest rates", "monetary policy",
    "opec", "opec+", "opec+ agreement", "opec+ agreement 2026", "opec+ agreement 2027", "opec+ agreement 2028", "opec+ agreement 2029", "opec+ agreement 2030", "opec+ agreement 2031", "opec+ agreement 2032", "opec+ agreement 2033", "opec+ agreement 2034", "opec+ agreement 2035", "opec+ agreement 2036", "opec+ agreement 2037", "opec+ agreement 2038", "opec+ agreement 2039", "opec+ agreement 2040",
]

# Exclude false positives (sports teams, universities, etc.)
EXCLUDE_KEYWORDS = [
    "steelers", "oilers", "raiders", "cornell", "girona", 
    "confederate", "goldy", "edmonton", "pittsburgh"
]

# Load all tags
with open("tags_simplified.json", "r", encoding="utf-8") as f:
    all_tags = json.load(f)

# Filter tags related to commodities
commodity_tags = []
for tag in all_tags:
    label = tag.get("label", "").lower() if tag.get("label") else ""
    slug = tag.get("slug", "").lower() if tag.get("slug") else ""
    
    # Skip if it matches exclusion patterns
    text = f"{label} {slug}"
    if any(exclude.lower() in text for exclude in EXCLUDE_KEYWORDS):
        continue
    
    # Check if any keyword matches in label or slug
    if any(keyword.lower() in text for keyword in COMMODITY_KEYWORDS):
        commodity_tags.append(tag)

# Save filtered tags
with open("commodity_tags.json", "w", encoding="utf-8") as f:
    json.dump(commodity_tags, f, indent=2, ensure_ascii=False)

print(f"Found {len(commodity_tags)} commodity-related tags out of {len(all_tags)} total tags")
print(f"Commodity tags saved to commodity_tags.json")

# Print some examples
print("\nCommodity tags found:")
for tag in commodity_tags:
    print(f"  - ID: {tag.get('id')}, Label: {tag.get('label')}, Slug: {tag.get('slug')}")
