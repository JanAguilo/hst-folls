import React, { useState, useEffect } from 'react';
import { Search, Plus, X, TrendingUp, DollarSign } from 'lucide-react';
import type { CommodityWithQuantity } from '../../types';

interface CommoditySelectorProps {
  selectedCommodities: CommodityWithQuantity[];
  onAddCommodity: (commodity: string) => void;
  onRemoveCommodity: (commodity: string) => void;
  onUpdateQuantity: (commodity: string, quantity: number) => void;
  onContinue: () => void;
}

const COMMON_COMMODITIES = [
  { name: 'Gold', symbol: 'Gold (GC=F)' },
  { name: 'Silver', symbol: 'Silver (SI=F)' },
  { name: 'Crude Oil', symbol: 'Crude Oil (CL=F)' },
  { name: 'Natural Gas', symbol: 'Natural Gas (NG=F)' },
  { name: 'Wheat', symbol: 'Wheat (ZW=F)' },
  { name: 'Corn', symbol: 'Corn (ZC=F)' },
  { name: 'Soybeans', symbol: 'Soybeans (ZS=F)' },
  { name: 'Copper', symbol: 'Copper (HG=F)' },
  { name: 'Coffee', symbol: 'Coffee (KC=F)' },
  { name: 'Sugar', symbol: 'Sugar (SB=F)' },
  { name: 'Cotton', symbol: 'Cotton (CT=F)' },
  { name: 'Platinum', symbol: 'Platinum (PL=F)' },
];

export const CommoditySelector: React.FC<CommoditySelectorProps> = ({
  selectedCommodities,
  onAddCommodity,
  onRemoveCommodity,
  onUpdateQuantity,
  onContinue
}) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [inputValues, setInputValues] = useState<Record<string, string>>({});

  const filteredCommodities = COMMON_COMMODITIES.filter(c => 
    c.name.toLowerCase().includes(searchTerm.toLowerCase()) &&
    !selectedCommodities.some(sc => sc.commodity === c.symbol)
  );

  // Initialize input values when commodities are added
  useEffect(() => {
    const newValues: Record<string, string> = {};
    selectedCommodities.forEach(({ commodity, quantity }) => {
      if (!(commodity in inputValues)) {
        newValues[commodity] = quantity > 0 ? quantity.toString() : '';
      }
    });
    if (Object.keys(newValues).length > 0) {
      setInputValues(prev => ({ ...prev, ...newValues }));
    }
  }, [selectedCommodities.map(c => c.commodity).join(',')]);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Hero Section */}
      <div className="card bg-gradient-to-br from-primary-500/10 to-blue-500/10 border-primary-500/30">
        <div className="text-center py-8">
          <TrendingUp className="w-16 h-16 text-primary-400 mx-auto mb-4" />
          <h2 className="text-3xl font-bold mb-2">Portfolio Risk Management</h2>
          <p className="text-slate-300 text-lg">
            Select the commodities in your portfolio and enter their values in USD. 
            We'll find relevant Polymarket events to help you hedge your positions.
          </p>
        </div>
      </div>

      {/* Selected Commodities with Quantities */}
      {selectedCommodities.length > 0 && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <DollarSign className="w-5 h-5 text-primary-400" />
            <h3 className="text-xl font-bold">Selected Commodities ({selectedCommodities.length})</h3>
          </div>
          
          <div className="space-y-4 mb-6">
            {selectedCommodities.map(({ commodity, quantity }) => (
              <div
                key={commodity}
                className="flex items-center gap-4 p-4 bg-slate-800/50 rounded-lg border border-slate-700 hover:border-primary-500/50 transition-all"
              >
                <div className="flex-1">
                  <div className="font-medium text-white mb-1">{commodity}</div>
                  <div className="text-sm text-slate-400">Enter your current position value in USD</div>
                </div>
                
                <div className="flex items-center gap-3">
                  <div className="flex flex-col">
                    <label className="text-xs text-slate-400 mb-1 flex items-center gap-1">
                      Quantity (USD)
                      <span className="text-slate-500">$</span>
                    </label>
                    <div className="relative">
                      <span className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400 font-medium z-10">$</span>
                      <input
                        type="text"
                        inputMode="decimal"
                        value={inputValues[commodity] ?? (quantity > 0 ? quantity.toString() : '')}
                        onChange={(e) => {
                          const rawValue = e.target.value;
                          // Allow empty string, numbers, and decimal point
                          if (rawValue === '' || /^\d*\.?\d*$/.test(rawValue)) {
                            // Update local state immediately for responsive UI
                            setInputValues(prev => ({ ...prev, [commodity]: rawValue }));
                            
                            // Update parent state
                            if (rawValue === '' || rawValue === '.') {
                              onUpdateQuantity(commodity, 0);
                            } else {
                              const numValue = parseFloat(rawValue);
                              if (!isNaN(numValue) && numValue >= 0) {
                                onUpdateQuantity(commodity, numValue);
                              }
                            }
                          }
                        }}
                        onBlur={(e) => {
                          const inputValue = e.target.value.trim();
                          if (inputValue === '' || inputValue === '.') {
                            setInputValues(prev => ({ ...prev, [commodity]: '' }));
                            onUpdateQuantity(commodity, 0);
                          } else {
                            const numValue = parseFloat(inputValue);
                            if (!isNaN(numValue) && numValue >= 0) {
                              const formatted = numValue.toString();
                              setInputValues(prev => ({ ...prev, [commodity]: formatted }));
                              onUpdateQuantity(commodity, numValue);
                            }
                          }
                        }}
                        className="w-40 pl-7 pr-3 py-2 bg-slate-700/50 border border-slate-600 rounded-lg text-white text-right focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent placeholder:text-slate-500"
                        placeholder="0.00"
                        autoComplete="off"
                      />
                    </div>
                  </div>
                  
                  <button
                    onClick={() => onRemoveCommodity(commodity)}
                    className="p-2 hover:bg-danger-500/20 text-danger-400 rounded transition-colors"
                    title="Remove commodity"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
          
          {selectedCommodities.length > 0 && (
            <div className="mb-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
              <p className="text-sm text-blue-300">
                💡 <strong>Note:</strong> Enter the total dollar value of your current position for each commodity. 
                This will be used to calculate portfolio risk and find appropriate hedging opportunities.
              </p>
            </div>
          )}
          
          {selectedCommodities.length > 0 && (
            <button
              onClick={onContinue}
              className="btn-primary w-full py-4 text-lg"
            >
              Continue to Market Selection →
            </button>
          )}
        </div>
      )}

      {/* Search Common Commodities */}
      <div className="card">
        <h3 className="text-xl font-bold mb-4">Popular Commodities</h3>
        
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="input-field w-full pl-10"
            placeholder="Search commodities..."
          />
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
          {filteredCommodities.map(commodity => (
            <button
              key={commodity.symbol}
              onClick={() => onAddCommodity(commodity.symbol)}
              className="flex items-center justify-between p-3 bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700 hover:border-primary-500 rounded-lg transition-all group"
            >
              <span className="font-medium text-slate-300 group-hover:text-primary-300">
                {commodity.name}
              </span>
              <Plus className="w-4 h-4 text-slate-500 group-hover:text-primary-400" />
            </button>
          ))}
        </div>

        {filteredCommodities.length === 0 && searchTerm && (
          <div className="text-center py-8 text-slate-400">
            No commodities found matching "{searchTerm}"
          </div>
        )}
      </div>
    </div>
  );
};
