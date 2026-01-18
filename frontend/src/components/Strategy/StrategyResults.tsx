import React from 'react';
import { CheckCircle, TrendingUp, DollarSign, Target, ArrowRight, Info, Loader } from 'lucide-react';
import type { Greeks } from '../../types';

interface OptimalPosition {
  market_id: string;
  market_title: string;
  action: string;
  quantity: number;
  position_value: number;
  yes_price: number;
  no_price: number;
  greek_contributions: Greeks;
}

interface StrategyResult {
  success: boolean;
  optimal_positions: OptimalPosition[];
  achieved_greeks: Greeks;
  target_greeks: Greeks;
  deviations: Record<string, number>;
  total_investment: number;
  num_positions: number;
  optimization_time_ms: number;
  metrics: Record<string, any>;
  initial_greeks?: Greeks;
  greek_changes_from_initial?: Greeks;
}

interface StrategyResultsProps {
  result: StrategyResult | null;
  onApplyPositions: (positions: OptimalPosition[]) => void;
  onClose: () => void;
  isApplying?: boolean;
}

export const StrategyResults: React.FC<StrategyResultsProps> = ({
  result,
  onApplyPositions,
  onClose,
  isApplying = false
}) => {
  if (!result) return null;

  const efficiency = result.metrics?.portfolio_efficiency || 0;
  
  const getDeviationColor = (deviation: number, target: number) => {
    if (Math.abs(target) < 0.0001) return 'text-slate-400';
    const errorPct = Math.abs(deviation / target) * 100;
    if (errorPct < 10) return 'text-success-400';
    if (errorPct < 25) return 'text-warning-400';
    return 'text-danger-400';
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Success Banner */}
      {result.success && (
        <div className="card bg-gradient-to-r from-success-600/20 to-primary-600/20 border-success-500/30">
          <div className="flex items-start gap-3">
            <CheckCircle className="w-6 h-6 text-success-400 flex-shrink-0 mt-1" />
            <div>
              <h3 className="text-lg font-semibold text-white mb-1">
                Strategy Generated Successfully!
              </h3>
              <p className="text-success-200 text-sm">
                Found {result.num_positions} optimal position{result.num_positions !== 1 ? 's' : ''} 
                {' '}for ${result.total_investment.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                {' '}(Efficiency: {(efficiency * 100).toFixed(1)}%)
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Greeks Comparison */}
      <div className="card bg-slate-800/50">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-primary-400" />
          Greeks Achievement
        </h3>
        
        <div className="space-y-3">
          {(['delta', 'gamma', 'vega', 'theta'] as const).map((greek) => {
            const target = result.target_greeks[greek] || 0;
            const achieved = result.achieved_greeks[greek] || 0;
            const deviation = result.deviations[greek] || (achieved - target);
            const deviationColor = getDeviationColor(deviation, target);
            
            return (
              <div key={greek} className="bg-slate-900/50 p-4 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-slate-400 capitalize">{greek}</span>
                  <span className={`text-sm font-semibold ${deviationColor}`}>
                    {Math.abs(target) > 0.0001
                      ? `${Math.abs(deviation / target * 100).toFixed(1)}% error`
                      : 'N/A'}
                  </span>
                </div>
                
                <div className="flex items-center gap-3">
                  <div className="flex-1">
                    <div className="text-xs text-slate-500 mb-1">Target</div>
                    <div className="text-lg font-bold text-white">
                      {target.toFixed(4)}
                    </div>
                  </div>
                  
                  <ArrowRight className="w-5 h-5 text-slate-600 flex-shrink-0" />
                  
                  <div className="flex-1">
                    <div className="text-xs text-slate-500 mb-1">Achieved</div>
                    <div className={`text-lg font-bold ${deviationColor}`}>
                      {achieved.toFixed(4)}
                    </div>
                  </div>
                  
                  {result.initial_greeks && (
                    <>
                      <div className="h-8 w-px bg-slate-700"></div>
                      <div className="flex-1">
                        <div className="text-xs text-slate-500 mb-1">Initial</div>
                        <div className="text-sm font-semibold text-slate-400">
                          {result.initial_greeks[greek].toFixed(4)}
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Investment Summary */}
      <div className="card bg-slate-800/50">
        <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
          <DollarSign className="w-5 h-5 text-success-400" />
          Investment Summary
        </h3>
        
        <div className="grid grid-cols-3 gap-4">
          <div className="bg-slate-900/50 p-4 rounded-lg">
            <div className="text-slate-400 text-sm mb-1">Total Investment</div>
            <div className="text-2xl font-bold text-white">
              ${result.total_investment.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
            </div>
          </div>
          
          <div className="bg-slate-900/50 p-4 rounded-lg">
            <div className="text-slate-400 text-sm mb-1">Num Positions</div>
            <div className="text-2xl font-bold text-white">
              {result.num_positions}
            </div>
          </div>
          
          <div className="bg-slate-900/50 p-4 rounded-lg">
            <div className="text-slate-400 text-sm mb-1">Efficiency</div>
            <div className="text-2xl font-bold text-success-400">
              {(efficiency * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Optimal Positions */}
      {result.optimal_positions && result.optimal_positions.length > 0 && (
        <div className="card bg-slate-800/50">
          <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary-400" />
            Recommended Positions
          </h3>
          
          <div className="space-y-3">
            {result.optimal_positions.slice(0, 10).map((position, idx) => (
              <div key={position.market_id} className="bg-slate-900/50 p-4 rounded-lg hover:bg-slate-900/70 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-bold text-slate-500">#{idx + 1}</span>
                      <span className={`px-2 py-0.5 text-xs font-semibold rounded ${
                        position.action === 'BUY YES' 
                          ? 'bg-success-600/20 text-success-400' 
                          : 'bg-danger-600/20 text-danger-400'
                      }`}>
                        {position.action}
                      </span>
                    </div>
                    <h4 className="font-semibold text-white text-sm leading-tight">
                      {position.market_title}
                    </h4>
                  </div>
                  <div className="text-right ml-4">
                    <div className="text-sm text-slate-400">Investment</div>
                    <div className="text-lg font-bold text-white">
                      ${position.position_value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </div>
                  </div>
                </div>
                
                <div className="mt-3 pt-3 border-t border-slate-800">
                  <div className="bg-slate-800/50 p-3 rounded-lg mb-2">
                    <div className="text-xs font-semibold text-slate-300 mb-1">📋 Trading Instructions:</div>
                    <div className="text-sm text-white">
                      {position.action === 'BUY YES' ? (
                        <>
                          Buy <span className="font-bold text-success-400">{Math.abs(position.quantity).toFixed(2)} shares</span> of the <span className="font-bold">YES</span> outcome
                          {' '}at ${position.yes_price.toFixed(3)} per share
                          {' '}= <span className="font-bold text-success-400">${position.position_value.toFixed(2)}</span> total
                        </>
                      ) : (
                        <>
                          Buy <span className="font-bold text-danger-400">{Math.abs(position.quantity).toFixed(2)} shares</span> of the <span className="font-bold">NO</span> outcome
                          {' '}at ${position.no_price.toFixed(3)} per share
                          {' '}= <span className="font-bold text-danger-400">${position.position_value.toFixed(2)}</span> total
                        </>
                      )}
                    </div>
                  </div>
                  <div className="flex gap-4 text-xs text-slate-400">
                    <span>
                      Quantity: <span className="text-white font-semibold">{Math.abs(position.quantity).toFixed(2)}</span>
                    </span>
                    <span>
                      YES Price: <span className="text-white font-semibold">${position.yes_price.toFixed(3)}</span>
                    </span>
                    <span>
                      NO Price: <span className="text-white font-semibold">${position.no_price.toFixed(3)}</span>
                    </span>
                  </div>
                </div>
                
                {/* Greek Contributions */}
                <div className="mt-3 pt-3 border-t border-slate-800">
                  <div className="text-xs text-slate-400 mb-2">Greek Contributions:</div>
                  <div className="grid grid-cols-4 gap-2">
                    {(['delta', 'gamma', 'vega', 'theta'] as const).map((greek) => (
                      <div key={greek} className="text-center">
                        <div className="text-xs text-slate-500 capitalize">{greek}</div>
                        <div className="text-sm font-semibold text-white">
                          {position.greek_contributions[greek].toFixed(4)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {result.optimal_positions.length > 10 && (
            <div className="mt-4 text-center text-sm text-slate-400">
              ... and {result.optimal_positions.length - 10} more position{result.optimal_positions.length - 10 !== 1 ? 's' : ''}
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={onClose}
          disabled={isApplying}
          className="flex-1 px-6 py-3 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Close
        </button>
        <button
          onClick={() => onApplyPositions(result.optimal_positions)}
          disabled={isApplying}
          className="flex-1 px-6 py-3 bg-gradient-to-r from-primary-600 to-blue-600 hover:from-primary-500 hover:to-blue-500 text-white font-semibold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isApplying ? (
            <>
              <Loader className="w-5 h-5 animate-spin" />
              Applying Positions...
            </>
          ) : (
            'Apply These Positions'
          )}
        </button>
      </div>

      {/* Info */}
      <div className="card bg-blue-600/10 border-blue-500/30">
        <div className="flex items-start gap-2">
          <Info className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-blue-200">
            These are AI-generated recommendations based on your target Greeks and budget. 
            Review each position carefully before executing trades on Polymarket.
          </p>
        </div>
      </div>
    </div>
  );
};
