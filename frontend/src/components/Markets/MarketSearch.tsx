import React, { useState, useEffect } from 'react';
import { Search, Loader, Package, X, Plus, Minus } from 'lucide-react';
import type { PolymarketMarket, HypotheticalPosition } from '../../types';
import { api } from '../../services/api';
import { MarketCard } from './MarketCard';

interface MarketSearchProps {
  selectedCommodities: string[];
  hypotheticalPositions: HypotheticalPosition[];
  onSelectMarket: (market: PolymarketMarket, side: 'YES' | 'NO', size: number) => void;
  onRemovePosition: (marketId: string, side: 'YES' | 'NO') => void;
  onUpdatePositionSize: (marketId: string, side: 'YES' | 'NO', newSize: number) => void;
}

export const MarketSearch: React.FC<MarketSearchProps> = ({
  selectedCommodities,
  hypotheticalPositions,
  onSelectMarket,
  onRemovePosition,
  onUpdatePositionSize
}) => {
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activeSearch, setActiveSearch] = useState<string | null>(null);
  const [activeCommodityTab, setActiveCommodityTab] = useState<string | 'all'>('all');

  // Automatically search for all selected commodities when component mounts or commodities change
  useEffect(() => {
    if (selectedCommodities.length > 0) {
      searchAllCommodities();
    }
  }, [selectedCommodities]);

  const searchAllCommodities = async () => {
    setIsLoading(true);
    try {
      const allResults = await Promise.all(
        selectedCommodities.map(async (commodity) => {
          const results = await api.searchMarkets(commodity);
          return {
            commodity,
            ...results
          };
        })
      );
      setSearchResults(allResults);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Get filtered results based on active tab
  const getFilteredResults = () => {
    if (activeCommodityTab === 'all') {
      return searchResults;
    }
    return searchResults.filter(result => result.commodity === activeCommodityTab);
  };

  const filteredResults = getFilteredResults();

  const getPositionForMarket = (marketId: string, side: 'YES' | 'NO'): HypotheticalPosition | undefined => {
    return hypotheticalPositions.find(hp => hp.market.id === marketId && hp.side === side);
  };

  if (isLoading) {
    return (
      <div className="card text-center py-12">
        <Loader className="w-12 h-12 animate-spin text-primary-400 mx-auto mb-4" />
        <p className="text-slate-400">Searching for related markets...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Selected Commodities Overview */}
      <div className="card bg-gradient-to-br from-primary-500/10 to-blue-500/10 border-primary-500/30">
        <div className="flex items-center gap-3 mb-4">
          <Package className="w-6 h-6 text-primary-400" />
          <h3 className="text-xl font-bold">Hedging Commodities ({selectedCommodities.length})</h3>
        </div>
        
        {/* Tabs for toggling between commodities (only show if more than 1) */}
        {selectedCommodities.length > 1 ? (
          <div className="flex flex-wrap gap-2 mb-4 pb-4 border-b border-primary-500/20">
            <button
              onClick={() => setActiveCommodityTab('all')}
              className={`px-4 py-2 rounded-full font-medium transition-all ${
                activeCommodityTab === 'all'
                  ? 'bg-primary-500 text-white border-2 border-primary-400'
                  : 'bg-primary-500/20 text-primary-300 border border-primary-500/30 hover:bg-primary-500/30'
              }`}
            >
              All ({selectedCommodities.length})
            </button>
            {selectedCommodities.map(commodity => (
              <button
                key={commodity}
                onClick={() => setActiveCommodityTab(commodity)}
                className={`px-4 py-2 rounded-full font-medium transition-all ${
                  activeCommodityTab === commodity
                    ? 'bg-primary-500 text-white border-2 border-primary-400'
                    : 'bg-primary-500/20 text-primary-300 border border-primary-500/30 hover:bg-primary-500/30'
                }`}
              >
                {commodity}
              </button>
            ))}
          </div>
        ) : (
          <div className="flex flex-wrap gap-2">
            {selectedCommodities.map(commodity => (
              <div
                key={commodity}
                className="px-4 py-2 bg-primary-500/20 text-primary-300 rounded-full border border-primary-500/30 font-medium"
              >
                {commodity}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Hypothetical Positions Summary */}
      {hypotheticalPositions.length > 0 && (
        <div className="card bg-slate-800/50">
          <h3 className="text-xl font-bold mb-4">Hypothetical Positions ({hypotheticalPositions.length})</h3>
          <div className="space-y-3">
            {hypotheticalPositions.map(hp => (
              <div
                key={`${hp.market.id}-${hp.side}`}
                className="flex items-center justify-between p-4 bg-slate-900/50 rounded-lg border border-slate-700"
              >
                <div className="flex-1">
                  <div className="font-semibold text-white mb-1">{hp.market.question}</div>
                  <div className="flex items-center gap-4 text-sm text-slate-400">
                    <span className={`px-2 py-1 rounded ${hp.side === 'YES' ? 'bg-success-500/20 text-success-400' : 'bg-danger-500/20 text-danger-400'}`}>
                      {hp.side}
                    </span>
                    <span>Price: ${hp.side === 'YES' ? hp.market.yesPrice.toFixed(2) : hp.market.noPrice.toFixed(2)}</span>
                  </div>
                </div>
                
                <div className="flex items-center gap-3">
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => onUpdatePositionSize(hp.market.id, hp.side, hp.size - 100)}
                      className="p-1 hover:bg-slate-700 rounded transition-colors"
                      disabled={hp.size <= 100}
                    >
                      <Minus className="w-4 h-4" />
                    </button>
                    <input
                      type="number"
                      value={hp.size}
                      onChange={(e) => onUpdatePositionSize(hp.market.id, hp.side, parseFloat(e.target.value) || 0)}
                      className="w-24 px-2 py-1 bg-slate-800 border border-slate-600 rounded text-center"
                    />
                    <button
                      onClick={() => onUpdatePositionSize(hp.market.id, hp.side, hp.size + 100)}
                      className="p-1 hover:bg-slate-700 rounded transition-colors"
                    >
                      <Plus className="w-4 h-4" />
                    </button>
                  </div>
                  
                  <button
                    onClick={() => onRemovePosition(hp.market.id, hp.side)}
                    className="p-2 hover:bg-danger-500/20 text-danger-400 rounded transition-colors"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Search Results by Commodity (filtered by active tab) */}
      {filteredResults.map((result) => (
        <div key={result.commodity} className="space-y-4">
          {/* Direct Results */}
          {result.directResults.length > 0 && (
            <div className="card">
              <h4 className="text-xl font-bold mb-4">
                {result.commodity} Markets ({result.directResults.length})
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {result.directResults.map((market: PolymarketMarket) => (
                  <MarketCard
                    key={market.id}
                    market={market}
                    onSelect={onSelectMarket}
                    existingPosition={hypotheticalPositions.find(hp => hp.market.id === market.id)}
                  />
                ))}
              </div>
            </div>
          )}

          {/* No Direct Results Message */}
          {result.directResults.length === 0 && (
            <div className="card text-center py-6 bg-slate-800/30">
              <div className="text-slate-400">
                No active markets found for <span className="font-semibold">{result.commodity}</span>
              </div>
              {result.correlatedCommodity && (
                <div className="text-sm text-slate-500 mt-2">
                  ↓ Showing correlated markets below for <span className="font-semibold">{result.commodity}</span>
                </div>
              )}
            </div>
          )}

          {/* Correlated Results */}
          {result.correlatedCommodity && result.correlatedResults.length > 0 && (
            <div className="card border-2 border-amber-500/30 bg-amber-500/5">
              <div className="flex items-center gap-3 mb-4">
                <div className="px-3 py-1 bg-amber-500/20 text-amber-400 rounded-full text-sm font-semibold">
                  Correlated
                </div>
                <h4 className="text-xl font-bold">
                  {result.correlatedCommodity} Markets ({result.correlatedResults.length})
                </h4>
              </div>
              <p className="text-slate-400 mb-4">
                These markets are statistically correlated with {result.commodity} based on historical data
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {result.correlatedResults.map((market: PolymarketMarket) => (
                  <MarketCard
                    key={market.id}
                    market={market}
                    onSelect={onSelectMarket}
                    isCorrelated
                    existingPosition={hypotheticalPositions.find(hp => hp.market.id === market.id)}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      ))}

      {searchResults.length === 0 && !isLoading && (
        <div className="card text-center py-12">
          <Search className="w-16 h-16 text-slate-600 mx-auto mb-4" />
          <h3 className="text-xl font-bold text-slate-400 mb-2">
            No commodities selected
          </h3>
          <p className="text-slate-500">
            Go back and select commodities to see related markets
          </p>
        </div>
      )}
    </div>
  );
};
