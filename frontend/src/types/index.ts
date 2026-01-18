export interface Position {
  id: string;
  marketId: string;
  marketQuestion: string;
  commodity: string;
  side: 'YES' | 'NO';
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
}

export interface Portfolio {
  positions: Position[];
  totalValue: number;
  totalPnL: number;
  greeks: Greeks;
}

export interface Greeks {
  delta: number;
  gamma: number;
  vega: number;
  theta: number;
  rho: number;
}

export interface PolymarketMarket {
  id: string;
  question: string;
  slug: string;
  endDate: string;
  yesPrice: number;
  noPrice: number;
  volume: number;
  liquidity: number;
  relatedCommodity?: string;
  relatedCommodities?: string[];
}

export interface SimulatedPosition {
  market: PolymarketMarket;
  side: 'YES' | 'NO';
  size: number;
}

export interface GreeksDelta {
  current: Greeks;
  simulated: Greeks;
  delta: Greeks;
  riskChange: 'hedge' | 'concentrate' | 'neutral';
}

export interface CommoditySearch {
  commodity: string;
  directResults: PolymarketMarket[];
  correlatedCommodity?: string;
  correlatedResults: PolymarketMarket[];
}

export interface HypotheticalPosition {
  market: PolymarketMarket;
  side: 'YES' | 'NO';
  size: number;
}

export interface CommoditySelection {
  commodity: string;
  displayName: string;
}

export interface CommodityWithQuantity {
  commodity: string;
  quantity: number;
}