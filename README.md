# Polymarket Portfolio Greeks - Full Stack Application

Complete application for managing commodity portfolio risk using Polymarket markets and calculating Greeks.

## 🚀 Quick Start

### 1. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the backend server
python app.py
```

The backend will start on `http://localhost:5000`

**Test the backend:**
```bash
python backend/test_api.py
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (if not already done)
npm install

# Start the development server
npm run dev
```

The frontend will start on `http://localhost:5174` (or next available port)

## 📁 Project Structure

```
hst-folls/
├── backend/
│   ├── app.py                 # Flask backend API
│   ├── test_api.py           # API test suite
│   ├── requirements.txt      # Python dependencies
│   └── README.md             # Backend documentation
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main app with 2-step wizard flow
│   │   ├── components/
│   │   │   ├── Commodity/
│   │   │   │   └── CommoditySelector.tsx   # Step 1: Select commodities
│   │   │   ├── Markets/
│   │   │   │   ├── MarketSearch.tsx        # Step 2: Search & display markets
│   │   │   │   └── MarketCard.tsx          # Individual market card
│   │   │   └── Greeks/
│   │   │       └── GreeksDisplay.tsx       # Real-time Greeks visualization
│   │   ├── services/
│   │   │   └── api.ts        # API client (calls backend)
│   │   └── types/
│   │       └── index.ts      # TypeScript interfaces
│   └── ...
│
├── commodity_markets.json                   # All Polymarket events (15K+ events)
├── commodity_to_main_asset_mapping.json    # Commodity correlation mapping
└── commodity_vs_core_assets_correlations.csv # Historical correlation data
```

## 🎯 User Flow

### Step 1: Select Commodities to Hedge
- Choose from popular commodities (Gold, Silver, Oil, etc.)
- Or add custom commodities
- Continue to market selection

### Step 2: View Markets & Add Hypothetical Positions
- **Auto-search** for markets related to selected commodities
- View **direct markets** (if available) or **correlated markets**
- Add hypothetical positions (YES/NO, custom size)
- **Real-time Greeks** update as positions are added/modified

## 🔌 API Endpoints

### POST `/api/search-markets`

Search for markets related to a commodity.

**Request:**
```json
{
  "commodity": "Gold (GC=F)"
}
```

**Response:**
```json
{
  "commodity": "Gold (GC=F)",
  "directResults": [
    {
      "id": "1032223",
      "question": "Will Gold (GC) settle at <$4,350 in January?",
      "yesPrice": 0.06,
      "noPrice": 0.94,
      "volume": 48759.22,
      "liquidity": 8144.24,
      "relatedCommodity": "gold"
    }
  ],
  "correlatedCommodity": null,
  "correlatedResults": [],
  "message": "Found 5 event(s) directly related to Gold (GC=F)."
}
```

**Correlated Search Example:**

If searching for "Wheat (ZW=F)" with no direct markets:

```json
{
  "commodity": "Wheat (ZW=F)",
  "directResults": [],
  "correlatedCommodity": "Crude Oil (CL=F)",
  "correlatedResults": [...],
  "message": "No markets found directly related to Wheat (ZW=F) on Polymarket. However, here are 8 event(s) related to Crude Oil (CL=F), which is the commodity most correlated with Wheat (ZW=F)."
}
```

### GET `/api/health`

Health check endpoint.

## 🧪 Testing

### Backend Tests
```bash
python backend/test_api.py
```

Tests:
- ✅ Health endpoint
- ✅ Direct commodity search (Gold, Silver)
- ✅ Correlated commodity fallback (Wheat → Oil)

### Manual Testing Flow
1. Start backend: `python backend/app.py`
2. Start frontend: `cd frontend && npm run dev`
3. Open `http://localhost:5174`
4. Select "Gold" and "Wheat" commodities
5. Click "Continue to Market Selection"
6. See Gold direct markets + Wheat correlated markets (Oil)
7. Add hypothetical positions
8. Watch Greeks update in real-time

## 📊 Data Files

### `commodity_markets.json`
- 15,357 Polymarket events
- Each event has `relatedCommodity` field
- Contains market prices, volumes, liquidity

### `commodity_to_main_asset_mapping.json`
- Maps 28 commodities to main assets
- Used for correlation fallback
- Example: "Wheat (ZW=F)" → "Crude Oil (CL=F)"

### `commodity_vs_core_assets_correlations.csv`
- Historical correlation data
- Used to determine commodity relationships

## 🎨 Features

✅ **Commodity-First Workflow** - Start with what you want to hedge  
✅ **Smart Search** - Direct + correlated market results  
✅ **Real-Time Greeks** - Update instantly as positions change  
✅ **Hypothetical Positions** - Experiment before trading  
✅ **Position Management** - Add, remove, adjust sizes inline  
✅ **Visual Feedback** - Progress indicator, badges, color coding  
✅ **Error Handling** - Graceful fallbacks if backend unavailable  

## 🛠️ Technologies

**Backend:**
- Flask 3.0
- Flask-CORS
- Python 3.x

**Frontend:**
- React 18
- TypeScript
- Vite
- Tailwind CSS
- Recharts (for Greeks visualization)

## 📝 Notes

- Backend must be running for market search to work
- Frontend has fallback mock data if backend is unavailable
- Greeks calculation currently uses mock calculation (to be enhanced)
- Supports up to 20 markets per commodity search

## 🎯 Next Steps

- [ ] Add real Greeks calculation algorithm
- [ ] Add market data caching
- [ ] Add user portfolio persistence
- [ ] Add historical correlation visualization
- [ ] Add export functionality for positions
