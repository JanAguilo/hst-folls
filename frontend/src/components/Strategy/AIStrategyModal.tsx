import React, { useState } from 'react';
import { X, Sparkles, TrendingUp, Target, DollarSign, Loader } from 'lucide-react';
import type { Greeks, CommodityWithQuantity } from '../../types';

interface AIStrategyModalProps {
  isOpen: boolean;
  onClose: () => void;
  initialGreeks: Greeks | null;
  selectedCommodities: CommodityWithQuantity[];
  commodityNames: string[];
  onOptimize: (targetGreeks: Greeks, maxBudget: number) => Promise<void>;
  isOptimizing: boolean;
}

export const AIStrategyModal: React.FC<AIStrategyModalProps> = ({
  isOpen,
  onClose,
  initialGreeks,
  selectedCommodities,
  commodityNames,
  onOptimize,
  isOptimizing
}) => {
  const [targetGreeks, setTargetGreeks] = useState<Greeks>({
    delta: initialGreeks?.delta || 0,
    gamma: initialGreeks?.gamma || 0,
    vega: initialGreeks?.vega || 0,
    theta: initialGreeks?.theta || 0,
    rho: initialGreeks?.rho || 0
  });
  
  const [maxBudget, setMaxBudget] = useState<number>(5000);

  const handleGreekChange = (greek: keyof Greeks, value: string) => {
    const numValue = parseFloat(value) || 0;
    setTargetGreeks(prev => ({ ...prev, [greek]: numValue }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onOptimize(targetGreeks, maxBudget);
  };

  const handleUseInitial = () => {
    if (initialGreeks) {
      setTargetGreeks({ ...initialGreeks });
    }
  };

  const handleSetNeutral = () => {
    setTargetGreeks({
      delta: 0,
      gamma: 0,
      vega: 0,
      theta: 0,
      rho: 0
    });
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gradient-to-br from-slate-900 to-slate-800 border border-primary-500/30 rounded-2xl shadow-2xl max-w-3xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="sticky top-0 bg-gradient-to-r from-primary-600 to-blue-600 p-6 flex items-center justify-between rounded-t-2xl">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-white/20 rounded-lg">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold text-white">Build Your Strategy with AI</h2>
              <p className="text-primary-100 text-sm">Optimize your portfolio to achieve target Greeks</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-white/20 rounded-lg transition-colors"
            disabled={isOptimizing}
          >
            <X className="w-6 h-6 text-white" />
          </button>
        </div>

        {/* Content */}
        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Current Portfolio Summary */}
          {initialGreeks && (
            <div className="card bg-slate-800/50 border-slate-700">
              <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-primary-400" />
                Current Portfolio Greeks
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {(['delta', 'gamma', 'vega', 'theta'] as const).map((greek) => (
                  <div key={greek} className="bg-slate-900/50 p-3 rounded-lg">
                    <div className="text-slate-400 text-sm capitalize">{greek}</div>
                    <div className="text-xl font-bold text-white">
                      {initialGreeks[greek].toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Target Greeks Input */}
          <div className="card bg-slate-800/50 border-slate-700">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold flex items-center gap-2">
                <Target className="w-5 h-5 text-primary-400" />
                Target Greeks
              </h3>
              <div className="flex gap-2">
                {initialGreeks && (
                  <button
                    type="button"
                    onClick={handleUseInitial}
                    className="text-sm px-3 py-1 bg-primary-600/20 hover:bg-primary-600/30 text-primary-400 rounded-lg transition-colors"
                  >
                    Use Current
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleSetNeutral}
                  className="text-sm px-3 py-1 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors"
                >
                  Set Neutral
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {(['delta', 'gamma', 'vega', 'theta'] as const).map((greek) => (
                <div key={greek}>
                  <label className="block text-sm font-medium text-slate-300 mb-2 capitalize">
                    {greek}
                  </label>
                  <input
                    type="number"
                    step="0.0001"
                    value={targetGreeks[greek]}
                    onChange={(e) => handleGreekChange(greek, e.target.value)}
                    className="w-full px-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white focus:border-primary-500 focus:ring-2 focus:ring-primary-500/50 outline-none transition-all"
                    placeholder="0.0000"
                  />
                  {initialGreeks && (
                    <div className="text-xs text-slate-500 mt-1">
                      Current: {initialGreeks[greek].toFixed(4)}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Budget Input */}
          <div className="card bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
              <DollarSign className="w-5 h-5 text-success-400" />
              Maximum Budget
            </h3>
            <div className="relative">
              <span className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400 text-lg">$</span>
              <input
                type="number"
                min="100"
                step="100"
                value={maxBudget}
                onChange={(e) => setMaxBudget(parseFloat(e.target.value) || 0)}
                className="w-full pl-8 pr-4 py-3 bg-slate-900 border border-slate-600 rounded-lg text-white text-lg font-semibold focus:border-success-500 focus:ring-2 focus:ring-success-500/50 outline-none transition-all"
                placeholder="5000"
              />
            </div>
            <p className="text-xs text-slate-400 mt-2">
              Maximum amount to invest in new positions
            </p>
          </div>

          {/* Commodities Info */}
          <div className="card bg-slate-800/50 border-slate-700">
            <h3 className="text-lg font-semibold mb-3">
              Optimization Scope
            </h3>
            <div className="space-y-2">
              <div className="flex items-center justify-between py-2">
                <span className="text-slate-400">Selected Commodities:</span>
                <span className="text-white font-semibold">{commodityNames.length}</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {commodityNames.map((commodity) => (
                  <span
                    key={commodity}
                    className="px-3 py-1 bg-primary-600/20 text-primary-400 text-sm rounded-full"
                  >
                    {commodity.split('(')[0].trim()}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              type="button"
              onClick={onClose}
              disabled={isOptimizing}
              className="flex-1 px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isOptimizing || maxBudget <= 0}
              className="flex-1 px-6 py-3 bg-gradient-to-r from-primary-600 to-blue-600 hover:from-primary-500 hover:to-blue-500 text-white font-semibold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isOptimizing ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  Optimizing...
                </>
              ) : (
                <>
                  <Sparkles className="w-5 h-5" />
                  Generate Strategy
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
