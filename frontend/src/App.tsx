import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, Loader, Sparkles } from 'lucide-react';
import type { Greeks, HypotheticalPosition, PolymarketMarket, CommodityWithQuantity, StrategyResult, OptimalPosition } from './types';
import { api } from './services/api';
import { CommoditySelector } from './components/Commodity/CommoditySelector';
import { GreeksDisplay } from './components/Greeks/GreeksDisplay';
import { MarketSearch } from './components/Markets/MarketSearch';
import { AIStrategyModal } from './components/Strategy/AIStrategyModal';
import { StrategyResults } from './components/Strategy/StrategyResults';

type Step = 'select-commodities' | 'select-markets';

function App() {
  const [currentStep, setCurrentStep] = useState<Step>('select-commodities');
  const [selectedCommodities, setSelectedCommodities] = useState<CommodityWithQuantity[]>([]);
  const [hypotheticalPositions, setHypotheticalPositions] = useState<HypotheticalPosition[]>([]);
  const [initialGreeks, setInitialGreeks] = useState<Greeks | null>(null);
  const [greeks, setGreeks] = useState<Greeks | null>(null);
  const [isCalculating, setIsCalculating] = useState(false);
  
  // AI Strategy state
  const [showStrategyModal, setShowStrategyModal] = useState(false);
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [strategyResult, setStrategyResult] = useState<StrategyResult | null>(null);
  const [isApplyingPositions, setIsApplyingPositions] = useState(false);

  // Calculate initial portfolio Greeks whenever commodities/quantities change
  useEffect(() => {
    if (currentStep === 'select-markets' && selectedCommodities.length > 0) {
      const hasQuantities = selectedCommodities.some(c => c.quantity > 0);
      if (hasQuantities) {
        calculateInitialGreeks();
      } else {
        setInitialGreeks(null);
      }
    } else {
      setInitialGreeks(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCommodities, currentStep]);

  // Calculate Greeks whenever hypothetical positions change
  // DISABLED: We now use persistent portfolio which manages Greeks automatically
  // useEffect(() => {
  //   if (hypotheticalPositions.length > 0) {
  //     calculateGreeks();
  //   } else {
  //     // Reset to initial Greeks when no hypothetical positions
  //     setGreeks(initialGreeks);
  //   }
  //   // eslint-disable-next-line react-hooks/exhaustive-deps
  // }, [hypotheticalPositions, initialGreeks]);

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

  const calculateInitialGreeks = async () => {
    setIsCalculating(true);
    try {
      const calculatedGreeks = await api.calculateInitialPortfolioGreeks(selectedCommodities);
      setInitialGreeks(calculatedGreeks);
      // Also set as current Greeks if no hypothetical positions
      if (hypotheticalPositions.length === 0) {
        setGreeks(calculatedGreeks);
      }
    } catch (error) {
      console.error('Error calculating initial Greeks:', error);
    } finally {
      setIsCalculating(false);
    }
  };

  const calculateGreeks = async () => {
    if (hypotheticalPositions.length === 0) {
      // Reset to initial Greeks when no hypothetical positions
      setGreeks(initialGreeks);
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

  const handleAddHypotheticalPosition = async (market: PolymarketMarket, side: 'YES' | 'NO', size: number) => {
    try {
      // Add position to persistent backend
      const result = await api.addPortfolioPosition(market.id, side, size);
      
      if (result.success) {
        console.log('✅ Position added to persistent portfolio');
        
        // Update Greeks from backend
        const portfolioGreeks: Greeks = {
          delta: result.current_greeks.delta || 0,
          gamma: result.current_greeks.gamma || 0,
          vega: result.current_greeks.vega || 0,
          theta: result.current_greeks.theta || 0,
          rho: result.current_greeks.rho || 0
        };
        
        setGreeks(portfolioGreeks);
        
        // Update hypothetical positions list for UI display
        const existing = hypotheticalPositions.find(hp => hp.market.id === market.id && hp.side === side);
        if (existing) {
          setHypotheticalPositions(
            hypotheticalPositions.map(hp =>
              hp.market.id === market.id && hp.side === side
                ? { ...hp, size: hp.size + size }
                : hp
            )
          );
        } else {
          const newPosition: HypotheticalPosition = { market, side, size };
          setHypotheticalPositions([...hypotheticalPositions, newPosition]);
        }
      }
    } catch (error) {
      console.error('Error adding hypothetical position:', error);
      alert('Failed to add position. Please make sure the backend server is running.');
    }
  };

  const handleRemoveHypotheticalPosition = async (marketId: string, side: 'YES' | 'NO') => {
    try {
      // Find the position to remove
      const positionToRemove = hypotheticalPositions.find(
        hp => hp.market.id === marketId && hp.side === side
      );
      
      if (!positionToRemove) {
        console.warn('[REMOVE] Position not found:', marketId, side);
        return;
      }
      
      console.log('[REMOVE] Removing position:', marketId, side, positionToRemove.size);
      
      // Remove position from backend by adding negative quantity
      const negativeQuantity = -positionToRemove.size;
      await api.addPortfolioPosition(marketId, side, negativeQuantity);
      
      // Refresh portfolio state from backend
      const updatedState = await api.getPortfolioState();
      setHypotheticalPositions(updatedState.open_positions || []);
      setGreeks(updatedState.current_greeks);
      
      console.log('[REMOVE] Position removed, Greeks updated');
    } catch (error) {
      console.error('[REMOVE] Error removing position:', error);
      // Fallback: update UI anyway
      setHypotheticalPositions(
        hypotheticalPositions.filter(hp => !(hp.market.id === marketId && hp.side === side))
      );
    }
  };

  const handleUpdatePositionSize = async (marketId: string, side: 'YES' | 'NO', newSize: number) => {
    if (newSize <= 0) {
      await handleRemoveHypotheticalPosition(marketId, side);
    } else {
      try {
        // Find current position
        const currentPosition = hypotheticalPositions.find(
          hp => hp.market.id === marketId && hp.side === side
        );
        
        if (!currentPosition) {
          console.warn('[UPDATE] Position not found:', marketId, side);
          return;
        }
        
        console.log('[UPDATE] Updating position size:', marketId, side, currentPosition.size, '->', newSize);
        
        // Calculate the difference to add/remove
        const quantityDelta = newSize - currentPosition.size;
        
        // Update backend
        await api.addPortfolioPosition(marketId, side, quantityDelta);
        
        // Refresh portfolio state from backend
        const updatedState = await api.getPortfolioState();
        setHypotheticalPositions(updatedState.open_positions || []);
        setGreeks(updatedState.current_greeks);
        
        console.log('[UPDATE] Position updated, Greeks refreshed');
      } catch (error) {
        console.error('[UPDATE] Error updating position:', error);
        // Fallback: update UI anyway
        setHypotheticalPositions(
          hypotheticalPositions.map(hp =>
            hp.market.id === marketId && hp.side === side
              ? { ...hp, size: newSize }
              : hp
          )
        );
      }
    }
  };

  const handleOptimizeStrategy = async (targetGreeks: Greeks, maxBudget: number) => {
    setIsOptimizing(true);
    console.log('[OPTIMIZE] Starting optimization...', { targetGreeks, maxBudget, commodityNames });
    try {
      const result = await api.optimizeStrategy(
        selectedCommodities,
        targetGreeks,
        maxBudget,
        commodityNames
      );
      console.log('[OPTIMIZE] Optimization result:', result);
      
      if (result && result.success !== false) {
        // Set result and close modal
        setStrategyResult(result);
        setShowStrategyModal(false);
      } else {
        // Optimization failed
        const errorMsg = result?.metrics?.error || result?.error || 'Unknown error';
        console.error('[OPTIMIZE] Optimization failed:', errorMsg);
        alert(`Optimization failed: ${errorMsg}`);
      }
    } catch (error) {
      console.error('[OPTIMIZE] Error optimizing strategy:', error);
      const errorMsg = error instanceof Error ? error.message : 'Failed to optimize strategy. Please check the console for details.';
      alert(`Error: ${errorMsg}`);
    } finally {
      setIsOptimizing(false);
    }
  };

  const handleApplyOptimalPositions = async (positions: OptimalPosition[]) => {
    console.log('[APPLY] Applying optimal positions:', positions);
    console.log(`[APPLY] Will replace current portfolio with ${positions.length} optimized positions`);
    setIsApplyingPositions(true);
    
    try {
      // IMPORTANT: Reset portfolio first to avoid accumulating positions
      console.log('[APPLY] Resetting portfolio to apply fresh optimal positions...');
      await api.resetPortfolio();
      
      // Now add all optimal positions
      console.log(`[APPLY] Adding ${positions.length} optimal positions...`);
      for (const optimalPos of positions) {
        // Determine side from action
        const side: 'YES' | 'NO' = optimalPos.action === 'BUY YES' ? 'YES' : 'NO';
        const size = Math.abs(optimalPos.quantity);
        
        console.log(`[APPLY] Adding position: ${optimalPos.market_title.substring(0, 50)}... (${side} ${size.toFixed(2)})`);
        
        // Add to persistent portfolio via backend
        await api.addPortfolioPosition(optimalPos.market_id, side, size);
      }
      
      // Refresh portfolio state from backend
      const updatedState = await api.getPortfolioState();
      console.log('[APPLY] Updated state from backend:', updatedState);
      
      // Backend returns { current_greeks, open_positions }
      setHypotheticalPositions(updatedState.open_positions || []);
      
      // USE THE OPTIMIZER'S ACHIEVED GREEKS instead of backend calculation
      // This ensures the displayed Greeks match what the AI promised
      if (strategyResult?.achieved_greeks) {
        console.log('[APPLY] Using optimizer achieved Greeks (not backend recalculation)');
        setGreeks(strategyResult.achieved_greeks);
      } else {
        setGreeks(updatedState.current_greeks);
      }
      
      console.log('[APPLY] ===== APPLICATION COMPLETE =====');
      console.log(`[APPLY] Applied ${positions.length} optimal positions`);
      console.log(`[APPLY] Portfolio now has ${updatedState.open_positions?.length || 0} positions`);
      console.log('[APPLY] Displayed Greeks (from optimizer):', strategyResult?.achieved_greeks);
      console.log('[APPLY] Backend calculated Greeks:', updatedState.current_greeks);
      console.log('[APPLY] Target Greeks were:', strategyResult?.target_greeks);
      
      // Close the strategy results modal after a brief moment
      setTimeout(() => {
        setStrategyResult(null);
      }, 100);
      
    } catch (error) {
      console.error('[APPLY] Error applying positions:', error);
      console.error('[APPLY] Failed to apply positions. Please try adding them manually.');
    } finally {
      setIsApplyingPositions(false);
    }
  };

  const handleCloseStrategyResults = () => {
    setStrategyResult(null);
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
               {/* AI Strategy Button */}
               <button
                 onClick={() => setShowStrategyModal(true)}
                 disabled={!initialGreeks || selectedCommodities.length === 0}
                 className="w-full px-6 py-4 bg-gradient-to-r from-primary-600 to-blue-600 hover:from-primary-500 hover:to-blue-500 disabled:from-slate-700 disabled:to-slate-600 text-white font-bold rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3 shadow-lg hover:shadow-primary-500/50"
               >
                 <Sparkles className="w-6 h-6" />
                 <span>Build Your Strategy with AI</span>
               </button>

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

             {/* Right Column: Greeks Visualization or Strategy Results */}
             <div className="space-y-6 sticky top-24 h-fit">
               {strategyResult ? (
                <StrategyResults
                  result={strategyResult}
                  onApplyPositions={handleApplyOptimalPositions}
                  onClose={handleCloseStrategyResults}
                  isApplying={isApplyingPositions}
                />
               ) : isCalculating ? (
                 <div className="card text-center py-12 bg-slate-800/30">
                   <Loader className="w-8 h-8 animate-spin text-primary-400 mx-auto mb-4" />
                   <p className="text-slate-400">Calculating Greeks...</p>
                 </div>
               ) : greeks ? (
                 <div className="animate-fade-in">
                  <GreeksDisplay 
                    greeks={greeks} 
                    title={hypotheticalPositions.length > 0 ? "Portfolio Greeks (With Hypothetical Positions)" : "Initial Portfolio Greeks"} 
                  />
                 </div>
               ) : selectedCommodities.some(c => c.quantity > 0) ? (
                 <div className="card text-center py-12 bg-slate-800/30 border-dashed">
                   <TrendingUp className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                   <h3 className="text-xl font-bold text-slate-400 mb-2">
                     Calculating Initial Greeks...
                   </h3>
                   <p className="text-slate-500">
                     Please wait while we calculate your portfolio Greeks
                   </p>
                 </div>
               ) : (
                 <div className="card text-center py-12 bg-slate-800/30 border-dashed">
                   <TrendingUp className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                   <h3 className="text-xl font-bold text-slate-400 mb-2">
                     Enter Portfolio Quantities
                   </h3>
                   <p className="text-slate-500">
                     Go back to enter quantities for your commodities to see initial portfolio Greeks
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

      {/* AI Strategy Modal */}
      <AIStrategyModal
        isOpen={showStrategyModal}
        onClose={() => setShowStrategyModal(false)}
        initialGreeks={initialGreeks}
        selectedCommodities={selectedCommodities}
        commodityNames={commodityNames}
        onOptimize={handleOptimizeStrategy}
        isOptimizing={isOptimizing}
      />
    </div>
  );
}

export default App;
