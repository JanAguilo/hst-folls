# Polymarket Portfolio Greeks - Frontend

A modern, visually rich React application for managing Polymarket commodity portfolio risk through real-time Greeks calculations.

## Features

### 🎯 Portfolio Management
- Add and track commodity-related Polymarket positions
- Real-time P&L tracking
- Support for YES/NO positions

### 📊 Advanced Visualizations
- **Radar Charts**: Visualize Greeks distribution
- **Bar Charts**: Compare Greek magnitudes
- **Line Charts**: Track Greeks evolution
- Beautiful, animated UI components

### 🔍 Smart Market Discovery
- Search for commodity-related markets
- Automatic correlation mapping
- Display correlated markets when direct matches aren't found
- Uses `commodity_to_main_asset_mapping.json` for intelligent suggestions

### 🎲 Portfolio Simulation
- Simulate adding new positions before execution
- Real-time Greeks impact calculation
- Risk analysis (hedging vs concentrating)
- Visual comparison of current vs simulated Greeks

### 📈 Greeks Calculation
- **Delta**: $ change per 1% move in underlying
- **Gamma**: Delta change per 1% move
- **Vega**: $ change per 1% volatility increase
- **Theta**: $ change per day
- **Rho**: $ change per 1% rate change

## Technology Stack

- **React 18** with TypeScript
- **Vite** for fast development
- **Tailwind CSS** for styling
- **Recharts** for data visualization
- **Lucide React** for icons

## Getting Started

### Install Dependencies
```bash
npm install
```

### Run Development Server
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Build for Production
```bash
npm run build
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── Portfolio/        # Portfolio input & list
│   │   ├── Markets/          # Market search & cards
│   │   ├── Greeks/           # Greeks visualizations
│   │   └── Simulation/       # Simulation panel
│   ├── services/
│   │   └── api.ts           # Mock API (to be replaced with real backend)
│   ├── types/
│   │   └── index.ts         # TypeScript type definitions
│   ├── App.tsx              # Main application
│   ├── main.tsx             # Entry point
│   └── index.css            # Global styles
├── public/
├── package.json
├── tailwind.config.js
└── vite.config.ts
```

## Usage Flow

1. **Add Positions**: Start by adding your existing Polymarket positions
2. **View Greeks**: See real-time Greeks calculated for your portfolio
3. **Search Markets**: Find related markets by commodity (gold, silver, oil, etc.)
4. **Simulate**: Add markets to simulation to see risk impact before trading
5. **Analyze**: Review how simulated positions affect portfolio Greeks
6. **Iterate**: Try different combinations to find optimal risk balance

## Key Features Explained

### Market Search
- Enter a commodity name (e.g., "wheat", "gold")
- System searches for direct matches
- If no matches, displays correlated commodity markets
- Correlation data from `commodity_vs_core_assets_correlations.csv`

### Greeks Visualization
- **Radar Chart**: Shows relative magnitudes of all Greeks
- **Bar Chart**: Detailed breakdown with positive/negative coloring
- **Delta Display**: Real-time risk metrics

### Simulation Panel
- Add multiple positions to test
- Calculate combined impact
- See before/after comparison
- Risk indicator (hedge/concentrate/neutral)

## Next Steps

### Backend Integration
Replace `src/services/api.ts` mock functions with real API calls to:
- Polymarket API for live market data
- Backend service for Greeks calculation
- Portfolio persistence

### Enhancements
- WebSocket for real-time price updates
- Historical Greeks tracking
- Portfolio optimization suggestions
- Export/import portfolio data

## Design Philosophy

The UI emphasizes:
- **Visual Clarity**: Dark theme with high contrast
- **Data Visualization**: Charts and graphs for complex data
- **Smooth Animations**: Fade-ins, slide-ups for better UX
- **Responsive Design**: Works on desktop and tablet
- **Professional Look**: Modern gradient accents, glassmorphism

## Styling

Custom Tailwind classes available:
- `.card` - Standard card container
- `.card-hover` - Interactive card with hover effects
- `.btn-primary` - Primary action button
- `.btn-secondary` - Secondary action button
- `.input-field` - Form input styling
- `.greek-card` - Specialized card for Greeks display

## Development Notes

- Mock API includes realistic delays to simulate network requests
- Sample data includes Gold, Silver, and Oil markets
- Greeks calculations are simplified for demonstration
- Real implementation should use `polymarket_greeks.py` logic

---

Built for Polymarket Hackathon 2025 🚀
