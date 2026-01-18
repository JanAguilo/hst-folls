# Polymarket Portfolio Greeks - Backend

Flask backend API for searching commodity-related Polymarket markets and calculating portfolio Greeks.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### POST /api/search-markets

Search for Polymarket markets related to a specific commodity.

**Request Body:**
```json
{
  "commodity": "Gold (GC=F)"
}
```

**Response:**
```json
{
  "commodity": "Gold (GC=F)",
  "directResults": [...],
  "correlatedCommodity": null,
  "correlatedResults": [],
  "message": "Found 5 event(s) directly related to Gold (GC=F)."
}
```

**Features:**
- Returns direct markets if available
- Falls back to correlated commodity markets if no direct markets found
- Uses `commodity_to_main_asset_mapping.json` for correlation lookup
- Returns up to 20 markets per search

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "total_events": 15357,
  "total_commodities_in_mapping": 28
}
```

## Data Files

The backend reads from:
- `commodity_markets.json` - All Polymarket events with commodity tags
- `commodity_to_main_asset_mapping.json` - Commodity correlation mappings
