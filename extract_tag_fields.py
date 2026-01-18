import json

# Load all tags
with open("all_tags.json", "r") as f:
    all_tags = json.load(f)

# Extract only id, slug, and label
simplified_tags = []
for tag in all_tags:
    simplified_tag = {
        "id": tag.get("id"),
        "slug": tag.get("slug"),
        "label": tag.get("label")
    }
    simplified_tags.append(simplified_tag)

# Save to new JSON file
with open("tags_simplified.json", "w", encoding="utf-8") as f:
    json.dump(simplified_tags, f, indent=2, ensure_ascii=False)

print(f"Extracted {len(simplified_tags)} tags with id, slug, and label")
print("Saved to tags_simplified.json")
