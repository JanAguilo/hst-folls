import React from 'react';
import { X, Sparkles } from 'lucide-react';
import type { SimulatedPosition } from '../../types';

interface SimulationPanelProps {
  simulatedPositions: SimulatedPosition[];
  onRemove: (marketId: string) => void;
  onCalculate: () => void;
  isCalculating: boolean;
}

export const SimulationPanel: React.FC<SimulationPanelProps> = ({
  simulatedPositions,
  onRemove,
  onCalculate,
  isCalculating
}) => {
  if (simulatedPositions.length === 0) {
    return (
      <div className="card text-center py-12">
        <Sparkles className="w-12 h-12 text-slate-600 mx-auto mb-4" />
        <div className="text-slate-400 text-lg">
          No simulated positions yet
        </div>
        <div className="text-slate-500 text-sm mt-2">
          Add markets from the search results to simulate their impact
        </div>
      </div>
    );
  }

  const totalSize = simulatedPositions.reduce((sum, pos) => sum + pos.size, 0);

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold">Simulation ({simulatedPositions.length})</h3>
        <div className="text-slate-400">
          Total: <span className="font-bold text-white">${totalSize.toLocaleString()}</span>
        </div>
      </div>

      <div className="space-y-3">
        {simulatedPositions.map((pos, index) => (
          <div
            key={`${pos.market.id}-${index}`}
            className="bg-slate-700/30 rounded-lg p-4 flex items-start justify-between animate-slide-up"
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <h5 className="font-semibold">{pos.market.question}</h5>
                <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                  pos.side === 'YES'
                    ? 'bg-success-500/20 text-success-500'
                    : 'bg-danger-500/20 text-danger-500'
                }`}>
                  {pos.side}
                </span>
              </div>

              <div className="flex items-center gap-4 text-sm">
                <div>
                  <span className="text-slate-400">Size:</span>{' '}
                  <span className="font-semibold">${pos.size.toLocaleString()}</span>
                </div>
                <div>
                  <span className="text-slate-400">Price:</span>{' '}
                  <span className="font-semibold">
                    {pos.side === 'YES' ? pos.market.yesPrice.toFixed(2) : pos.market.noPrice.toFixed(2)}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">Commodity:</span>{' '}
                  <span className="font-semibold capitalize">{pos.market.relatedCommodity}</span>
                </div>
              </div>
            </div>

            <button
              onClick={() => onRemove(pos.market.id)}
              className="text-slate-400 hover:text-danger-500 transition-colors p-2"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        ))}
      </div>

      <button
        onClick={onCalculate}
        disabled={isCalculating}
        className="btn-primary w-full flex items-center justify-center gap-2 mt-4"
      >
        {isCalculating ? (
          <>
            <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
            Calculating...
          </>
        ) : (
          <>
            <Sparkles className="w-5 h-5" />
            Calculate Greeks Impact
          </>
        )}
      </button>
    </div>
  );
};
