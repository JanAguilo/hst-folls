import type { Position, Greeks, PolymarketMarket, GreeksDelta } from '../types';

// Mock data for development
const mockMarkets: PolymarketMarket[] = [
  {
    id: '1',
    question: 'Will Gold (GC) hit $2,800 by end of January?',
    slug: 'gold-2800-jan',
    endDate: '2026-01-31T18:30:00Z',
    yesPrice: 0.65,
    noPrice: 0.35,
    volume: 125000,
    liquidity: 45000,
    relatedCommodity: 'gold',
    relatedCommodities: ['gold']
  },
  {
    id: '2',
    question: 'Will Silver (SI) hit $33 by end of January?',
    slug: 'silver-33-jan',
    endDate: '2026-01-31T18:30:00Z',
    yesPrice: 0.42,
    noPrice: 0.58,
    volume: 89000,
    liquidity: 32000,
    relatedCommodity: 'silver',
    relatedCommodities: ['silver']
  },
  {
    id: '3',
    question: 'Will Crude Oil (CL) hit $75 by end of January?',
    slug: 'oil-75-jan',
    endDate: '2026-01-31T19:30:00Z',
    yesPrice: 0.58,
    noPrice: 0.42,
    volume: 156000,
    liquidity: 52000,
    relatedCommodity: 'oil',
    relatedCommodities: ['oil']
  },
];

const mockPositions: Position[] = [
  {
    id: 'pos1',
    marketId: '1',
    marketQuestion: 'Will Gold (GC) hit $2,800 by end of January?',
    commodity: 'gold',
    side: 'YES',
    size: 1000,
    entryPrice: 0.60,
    currentPrice: 0.65,
    pnl: 50
  }
];

// Mock API functions
export const api = {
  // Portfolio endpoints
  getPortfolio: async (): Promise<Position[]> => {
    await delay(500);
    return [...mockPositions];
  },

  addPosition: async (position: Omit<Position, 'id' | 'pnl'>): Promise<Position> => {
    await delay(300);
    const newPosition: Position = {
      ...position,
      id: `pos${Date.now()}`,
      pnl: (position.currentPrice - position.entryPrice) * position.size
    };
    mockPositions.push(newPosition);
    return newPosition;
  },

  removePosition: async (id: string): Promise<void> => {
    await delay(200);
    const index = mockPositions.findIndex(p => p.id === id);
    if (index > -1) {
      mockPositions.splice(index, 1);
    }
  },

  // Market search endpoints
  searchMarkets: async (commodity: string) => {
    try {
      const response = await fetch('http://localhost:5000/api/search-markets', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ commodity })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Transform backend data to match frontend PolymarketMarket interface
      const transformMarket = (market: any): PolymarketMarket => ({
        id: market.id,
        question: market.question,
        slug: market.slug,
        endDate: market.endDate,
        yesPrice: market.yesPrice || 0.5,
        noPrice: market.noPrice || 0.5,
        volume: market.volume || 0,
        liquidity: market.liquidity || 0,
        relatedCommodity: market.relatedCommodity
      });

      return {
        commodity: data.commodity,
        directResults: data.directResults.map(transformMarket),
        correlatedCommodity: data.correlatedCommodity,
        correlatedResults: data.correlatedResults.map(transformMarket),
        message: data.message
      };
    } catch (error) {
      console.error('Error searching markets:', error);
      // Fallback to mock data if backend is unavailable
      return {
        commodity,
        directResults: [],
        correlatedCommodity: undefined,
        correlatedResults: [],
        message: 'Error connecting to backend. Please ensure the backend server is running on port 5000.'
      };
    }
  },

  // Greeks calculation
  calculateGreeks: async (positions: Position[]): Promise<Greeks> => {
    await delay(400);
    
    // Mock Greeks calculation based on positions
    const totalDelta = positions.reduce((sum, pos) => {
      const sign = pos.side === 'YES' ? 1 : -1;
      return sum + (sign * pos.size * 0.01); // Simplified calculation
    }, 0);

    return {
      delta: totalDelta,
      gamma: totalDelta * 0.05,
      vega: totalDelta * 0.8,
      theta: -totalDelta * 0.02,
      rho: totalDelta * 0.1
    };
  },

  calculateGreeksDelta: async (
    currentPositions: Position[],
    simulatedPositions: Position[]
  ): Promise<GreeksDelta> => {
    await delay(500);
    
    const current = await api.calculateGreeks(currentPositions);
    const simulated = await api.calculateGreeks([...currentPositions, ...simulatedPositions]);
    
    const delta: Greeks = {
      delta: simulated.delta - current.delta,
      gamma: simulated.gamma - current.gamma,
      vega: simulated.vega - current.vega,
      theta: simulated.theta - current.theta,
      rho: simulated.rho - current.rho
    };

    // Determine risk change
    let riskChange: 'hedge' | 'concentrate' | 'neutral' = 'neutral';
    const deltaAbsChange = Math.abs(simulated.delta) - Math.abs(current.delta);
    
    if (deltaAbsChange < -10) {
      riskChange = 'hedge';
    } else if (deltaAbsChange > 10) {
      riskChange = 'concentrate';
    }

    return {
      current,
      simulated,
      delta,
      riskChange
    };
  },

  // Fetch market details
  getMarket: async (marketId: string): Promise<PolymarketMarket | null> => {
    // For now, we'll use a simplified version since we get full market data from search
    // In a real implementation, this could fetch from a separate endpoint
    await delay(100);
    
    // Check if market exists in mock data first
    const mockMarket = mockMarkets.find(m => m.id === marketId);
    if (mockMarket) {
      return mockMarket;
    }
    
    // If not found in mock data, return null
    // The frontend should already have the market data from the search results
    return null;
  }
};

// Helper functions
function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function getCorrelatedCommodity(commodity: string): string | undefined {
  // Simplified mapping - in real app, load from commodity_to_main_asset_mapping.json
  const mapping: Record<string, string> = {
    'wheat': 'oil',
    'corn': 'oil',
    'soybeans': 'oil',
    'copper': 'silver',
    'platinum': 'silver'
  };
  return mapping[commodity.toLowerCase()];
}
