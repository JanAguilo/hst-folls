import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, Loader } from 'lucide-react';
import type { Greeks, HypotheticalPosition, PolymarketMarket, CommodityWithQuantity } from './types';
import { api } from './services/api';
import { CommoditySelector } from './components/Commodity/CommoditySelector';
import { GreeksDisplay } from './components/Greeks/GreeksDisplay';
import { MarketSearch } from './components/Markets/MarketSearch';

type Step = 'select-commodities' | 'select-markets';

function App() {
  const [currentStep, setCurrentStep] = useState<Step>('select-commodities');
  const [selectedCommodities, setSelectedCommodities] = useState<CommodityWithQuantity[]>([]);
  const [hypotheticalPositions, setHypotheticalPositions] = useState<HypotheticalPosition[]>([]);
  const [greeks, setGreeks] = useState<Greeks | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);

  // Calculate Greeks whenever hypothetical positions change
  useEffect(() => {
    if (hypotheticalPositions.length > 0) {
      calculateGreeks();
    } else {
      setGreeks(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hypotheticalPositions]);

  const handleAddCommodity = (commodity: string) => {
    if (!selectedCommodities.some(c => c.commodity === commodity)) {
      setSelectedCommodities([...selectedCommodities, { commodity, quantity: 0 }]);
    }
  };

  const handleRemoveCommodity = (commodity: string) => {
    setSelectedCommodities(selectedCommodities.filter(c => c.commodity !== commodity));
  };

  const handleUpdateQuantity = (commodity: string, quantity: number) => {
    setSelectedCommodities(
      selectedCommodities.map(c =>
        c.commodity === commodity ? { ...c, quantity } : c
      )
    );
  };

  const handleContinueToMarkets = () => {
    if (selectedCommodities.length > 0) {
      setCurrentStep('select-markets');
    }
  };

  // Get commodity names for market search
  const commodityNames = selectedCommodities.map(c => c.commodity);

  const handleBackToCommodities = () => {
    setCurrentStep('select-commodities');
  };

  const calculateGreeks = async () => {
    if (hypotheticalPositions.length === 0) {
      setGreeks(null);
      return;
    }

    setIsCalculating(true);
    try {
      // Convert hypothetical positions to Position format for Greeks calculation
      const positions = hypotheticalPositions.map(hp => ({
        id: `hyp-${hp.market.id}`,
        marketId: hp.market.id,
        marketQuestion: hp.market.question,
        commodity: hp.market.relatedCommodity || 'unknown',
        side: hp.side,
        size: hp.size,
        entryPrice: hp.side === 'YES' ? hp.market.yesPrice : hp.market.noPrice,
        currentPrice: hp.side === 'YES' ? hp.market.yesPrice : hp.market.noPrice,
        pnl: 0
      }));

      const calculatedGreeks = await api.calculateGreeks(positions);
      setGreeks(calculatedGreeks);
    } catch (error) {
      console.error('Error calculating Greeks:', error);
    } finally {
      setIsCalculating(false);
    }
  };

  const handleAddHypotheticalPosition = (market: PolymarketMarket, side: 'YES' | 'NO', size: number) => {
    try {
      const existing = hypotheticalPositions.find(hp => hp.market.id === market.id && hp.side === side);
      if (existing) {
        // Update size if position already exists
        setHypotheticalPositions(
          hypotheticalPositions.map(hp =>
            hp.market.id === market.id && hp.side === side
              ? { ...hp, size: hp.size + size }
              : hp
          )
        );
      } else {
        // Add new position
        const newPosition: HypotheticalPosition = { market, side, size };
        setHypotheticalPositions([...hypotheticalPositions, newPosition]);
      }
    } catch (error) {
      console.error('Error adding hypothetical position:', error);
    }
  };

  const handleRemoveHypotheticalPosition = (marketId: string, side: 'YES' | 'NO') => {
    setHypotheticalPositions(
      hypotheticalPositions.filter(hp => !(hp.market.id === marketId && hp.side === side))
    );
  };

  const handleUpdatePositionSize = (marketId: string, side: 'YES' | 'NO', newSize: number) => {
    if (newSize <= 0) {
      handleRemoveHypotheticalPosition(marketId, side);
    } else {
      setHypotheticalPositions(
        hypotheticalPositions.map(hp =>
          hp.market.id === marketId && hp.side === side
            ? { ...hp, size: newSize }
            : hp
        )
      );
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Activity className="w-8 h-8 text-primary-500" />
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-primary-400 to-blue-500 bg-clip-text text-transparent">
                  Han Solo Tech
                </h1>
                <p className="text-sm text-slate-400">Risk management for commodity markets</p>
              </div>
            </div>

            {/* Display current step and Greeks summary */}
            <div className="flex items-center gap-4 text-sm">
              {currentStep === 'select-markets' && (
                <>
                  <button
                    onClick={handleBackToCommodities}
                    className="text-slate-400 hover:text-primary-400 transition-colors font-medium"
                  >
                    ← Back to Commodities
                  </button>
                  
                  {selectedCommodities.length > 0 && (
                    <div className="text-center border-l border-slate-700 pl-4">
                      <div className="text-slate-400">Portfolio Value</div>
                      <div className="text-xl font-bold text-primary-400">
                        ${selectedCommodities.reduce((sum, c) => sum + c.quantity, 0).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
                      </div>
                    </div>
                  )}
                  
                  {greeks && (
                    <>
                      <div className="text-center border-l border-slate-700 pl-4">
                        <div className="text-slate-400">Portfolio Delta</div>
                        <div className={`text-xl font-bold ${greeks.delta >= 0 ? 'text-success-500' : 'text-danger-500'}`}>
                          {greeks.delta >= 0 ? '+' : ''}{greeks.delta.toFixed(2)}
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-slate-400">Positions</div>
                        <div className="text-xl font-bold text-primary-400">{hypotheticalPositions.length}</div>
                      </div>
                    </>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Progress Indicator */}
      <div className="border-b border-slate-700 bg-slate-900/30">
        <div className="max-w-7xl mx-auto px-6 py-3">
          <div className="flex items-center gap-4">
            <div className={`flex items-center gap-2 ${currentStep === 'select-commodities' ? 'text-primary-400' : 'text-success-500'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold border-2 ${
                currentStep === 'select-commodities' ? 'border-primary-400 bg-primary-500/20' : 'border-success-500 bg-success-500/20'
              }`}>
                1
              </div>
              <span className="font-semibold">Select Commodities</span>
            </div>
            
            <div className={`flex-1 h-0.5 ${currentStep === 'select-markets' ? 'bg-primary-500' : 'bg-slate-700'}`} />
            
            <div className={`flex items-center gap-2 ${currentStep === 'select-markets' ? 'text-primary-400' : 'text-slate-500'}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold border-2 ${
                currentStep === 'select-markets' ? 'border-primary-400 bg-primary-500/20' : 'border-slate-700 bg-slate-800'
              }`}>
                2
              </div>
              <span className="font-semibold">Select Markets & View Greeks</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {currentStep === 'select-commodities' && (
          <CommoditySelector
            selectedCommodities={selectedCommodities}
            onAddCommodity={handleAddCommodity}
            onRemoveCommodity={handleRemoveCommodity}
            onUpdateQuantity={handleUpdateQuantity}
            onContinue={handleContinueToMarkets}
          />
        )}

        {currentStep === 'select-markets' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
             {/* Left Column: Commodities & Markets */}
             <div className="space-y-6">
               {/* Display Selected Commodities with Quantities */}
               <div className="card bg-gradient-to-br from-primary-500/10 to-blue-500/10 border-primary-500/30">
                 <h3 className="text-xl font-bold mb-4">Your Portfolio Values (USD)</h3>
                 {selectedCommodities.length > 0 ? (
                   <>
                     <div className="space-y-2">
                       {selectedCommodities.map(({ commodity, quantity }) => (
                         <div key={commodity} className="flex items-center justify-between p-3 bg-slate-900/50 rounded-lg">
                           <span className="font-medium text-slate-300">{commodity}</span>
                           <span className="text-lg font-bold text-primary-400">
                             ${quantity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                           </span>
                         </div>
                       ))}
                     </div>
                     <div className="mt-4 pt-4 border-t border-primary-500/20">
                       <div className="flex items-center justify-between">
                         <span className="font-semibold text-white">Total Portfolio Value:</span>
                         <span className="text-xl font-bold text-primary-400">
                           ${selectedCommodities.reduce((sum, c) => sum + (c.quantity || 0), 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                         </span>
                       </div>
                     </div>
                   </>
                 ) : (
                   <div className="text-center py-8 text-slate-400">
                     No commodities selected
                   </div>
                 )}
               </div>

               <MarketSearch
                 selectedCommodities={commodityNames}
                 hypotheticalPositions={hypotheticalPositions}
                 onSelectMarket={handleAddHypotheticalPosition}
                 onRemovePosition={handleRemoveHypotheticalPosition}
                 onUpdatePositionSize={handleUpdatePositionSize}
               />
             </div>

            {/* Right Column: Greeks Visualization */}
            <div className="space-y-6 sticky top-24 h-fit">
              {hypotheticalPositions.length > 0 ? (
                <>
                  {greeks ? (
                    <div className="animate-fade-in">
                      <GreeksDisplay greeks={greeks} title="Real-Time Portfolio Greeks" />
                      
                      <div className="card mt-6 bg-gradient-to-br from-primary-500/10 to-blue-500/10 border-primary-500/30">
                        <h3 className="text-xl font-bold mb-3">Portfolio Analysis</h3>
                        <div className="space-y-2 text-slate-300">
                          <p>
                            <span className="font-semibold text-white">Risk Exposure:</span>{' '}
                            {Math.abs(greeks.delta) < 20 ? 'Low' : Math.abs(greeks.delta) < 50 ? 'Moderate' : 'High'}
                          </p>
                          <p>
                            <span className="font-semibold text-white">Delta Direction:</span>{' '}
                            {greeks.delta > 0 ? 'Bullish (increases with price increases)' : 'Bearish (increases with price decreases)'}
                          </p>
                          <p className="text-sm text-slate-400 mt-4">
                            💡 These are hypothetical positions. Adjust sizes and sides to optimize your risk profile before executing real trades.
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="card text-center py-12 bg-slate-800/30">
                      <Loader className="w-8 h-8 animate-spin text-primary-400 mx-auto mb-4" />
                      <p className="text-slate-400">Calculating Greeks...</p>
                    </div>
                  )}
                </>
              ) : (
                <div className="card text-center py-12 bg-slate-800/30 border-dashed">
                  <TrendingUp className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                  <h3 className="text-xl font-bold text-slate-400 mb-2">
                    No Positions Selected Yet
                  </h3>
                  <p className="text-slate-500">
                    Select markets from the left to see real-time Greeks here
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="mt-20 border-t border-slate-700 bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-6 py-6 text-center text-slate-400 text-sm">
          <p>Han Solo Tech</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
