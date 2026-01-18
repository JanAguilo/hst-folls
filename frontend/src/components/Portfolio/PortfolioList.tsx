import React from 'react';
import { Trash2, TrendingUp, TrendingDown } from 'lucide-react';
import type { Position } from '../../types';

interface PortfolioListProps {
  positions: Position[];
  onRemovePosition: (id: string) => void;
}

export const PortfolioList: React.FC<PortfolioListProps> = ({ positions, onRemovePosition }) => {
  if (positions.length === 0) {
    return (
      <div className="card text-center py-12">
        <div className="text-slate-400 text-lg">
          No positions yet. Add your first position to get started.
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {positions.map((position, index) => (
        <div
          key={position.id}
          className="card-hover animate-slide-up"
          style={{ animationDelay: `${index * 50}ms` }}
        >
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <h4 className="font-semibold text-lg">{position.marketQuestion}</h4>
                <span className={`px-2 py-1 rounded text-sm font-medium ${
                  position.side === 'YES' 
                    ? 'bg-success-500/20 text-success-500' 
                    : 'bg-danger-500/20 text-danger-500'
                }`}>
                  {position.side}
                </span>
              </div>

              <div className="flex items-center gap-6 text-sm text-slate-300">
                <div>
                  <span className="text-slate-400">Commodity:</span>{' '}
                  <span className="font-medium capitalize">{position.commodity}</span>
                </div>
                <div>
                  <span className="text-slate-400">Size:</span>{' '}
                  <span className="font-medium">${position.size.toLocaleString()}</span>
                </div>
                <div>
                  <span className="text-slate-400">Entry:</span>{' '}
                  <span className="font-medium">{position.entryPrice.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-slate-400">Current:</span>{' '}
                  <span className="font-medium">{position.currentPrice.toFixed(2)}</span>
                </div>
              </div>

              <div className="mt-3 flex items-center gap-2">
                {position.pnl >= 0 ? (
                  <TrendingUp className="w-4 h-4 text-success-500" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-danger-500" />
                )}
                <span className={`font-semibold ${
                  position.pnl >= 0 ? 'text-success-500' : 'text-danger-500'
                }`}>
                  {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(2)} P&L
                </span>
              </div>
            </div>

            <button
              onClick={() => onRemovePosition(position.id)}
              className="text-slate-400 hover:text-danger-500 transition-colors p-2"
              title="Remove position"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};
