import React, { useState } from 'react';
import { TrendingUp, Calendar, DollarSign, Activity, CheckCircle } from 'lucide-react';
import type { PolymarketMarket, HypotheticalPosition } from '../../types';

interface MarketCardProps {
  market: PolymarketMarket;
  onSelect: (market: PolymarketMarket, side: 'YES' | 'NO', size: number) => void;
  isCorrelated?: boolean;
  existingPosition?: HypotheticalPosition;
}

export const MarketCard: React.FC<MarketCardProps> = ({ 
  market, 
  onSelect, 
  isCorrelated,
  existingPosition 
}) => {
  const [showSizeInput, setShowSizeInput] = useState(false);
  const [side, setSide] = useState<'YES' | 'NO'>('YES');
  const [size, setSize] = useState('1000');

  const handleAddPosition = () => {
    onSelect(market, side, parseFloat(size));
    setShowSizeInput(false);
    setSize('1000');
  };

  const endDate = new Date(market.endDate);
  const daysUntilEnd = Math.ceil((endDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24));

  return (
    <div className={`card-hover ${isCorrelated ? 'border-amber-500/30' : ''} ${existingPosition ? 'border-primary-500 border-2' : ''}`}>
      <div className="flex items-start justify-between mb-2">
        {isCorrelated && (
          <span className="px-2 py-1 bg-amber-500/20 text-amber-400 text-xs rounded font-medium">
            Correlated Market
          </span>
        )}
        {existingPosition && (
          <div className="flex items-center gap-1 px-2 py-1 bg-primary-500/20 text-primary-400 text-xs rounded font-medium ml-auto">
            <CheckCircle className="w-3 h-3" />
            Position Added
          </div>
        )}
      </div>

      <h5 className="font-semibold text-lg mb-3 line-clamp-2">
        {market.question}
      </h5>

      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-success-500" />
          <div>
            <div className="text-xs text-slate-400">YES</div>
            <div className="font-bold text-success-500">{(market.yesPrice * 100).toFixed(1)}¢</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-danger-500 rotate-180" />
          <div>
            <div className="text-xs text-slate-400">NO</div>
            <div className="font-bold text-danger-500">{(market.noPrice * 100).toFixed(1)}¢</div>
          </div>
        </div>
      </div>

      <div className="space-y-3 mb-4">
        {/* Market Stats */}
        <div className="flex items-center justify-between text-sm text-slate-400">
          <div className="flex items-center gap-1">
            <Calendar className="w-4 h-4" />
            <span>{market.expiryDays !== undefined ? `${market.expiryDays}d` : `${daysUntilEnd}d`} left</span>
          </div>
          <div className="flex items-center gap-1">
            <DollarSign className="w-4 h-4" />
            <span>${(market.volume / 1000).toFixed(1)}k vol</span>
          </div>
          <div className="flex items-center gap-1">
            <Activity className="w-4 h-4" />
            <span>${(market.liquidity / 1000).toFixed(1)}k liq</span>
          </div>
        </div>

        {/* Greeks Display (if available) */}
        {market.delta !== undefined && (
          <div className="bg-slate-900/50 rounded-lg p-3 border border-primary-500/20">
            <div className="text-xs text-slate-400 mb-2 font-semibold">Greeks (per share)</div>
            <div className="grid grid-cols-4 gap-2 text-xs">
              <div>
                <div className="text-slate-500">Δ</div>
                <div className="font-semibold text-white">{market.delta.toFixed(4)}</div>
              </div>
              <div>
                <div className="text-slate-500">Γ</div>
                <div className="font-semibold text-white">{market.gamma?.toFixed(6) || '0'}</div>
              </div>
              <div>
                <div className="text-slate-500">V</div>
                <div className="font-semibold text-white">{market.vega?.toFixed(4) || '0'}</div>
              </div>
              <div>
                <div className="text-slate-500">Θ</div>
                <div className="font-semibold text-white">{market.theta?.toFixed(6) || '0'}</div>
              </div>
            </div>
          </div>
        )}
      </div>

      {!showSizeInput ? (
        <button
          onClick={() => setShowSizeInput(true)}
          className="btn-primary w-full"
        >
          {existingPosition ? 'Add Another Position' : 'Add Hypothetical Position'}
        </button>
      ) : (
        <div className="space-y-3 animate-slide-up">
          <div className="flex gap-2">
            <button
              onClick={() => setSide('YES')}
              className={`flex-1 py-2 rounded-lg font-semibold transition-all ${
                side === 'YES'
                  ? 'bg-success-500 text-white'
                  : 'bg-slate-700 text-slate-400'
              }`}
            >
              YES
            </button>
            <button
              onClick={() => setSide('NO')}
              className={`flex-1 py-2 rounded-lg font-semibold transition-all ${
                side === 'NO'
                  ? 'bg-danger-500 text-white'
                  : 'bg-slate-700 text-slate-400'
              }`}
            >
              NO
            </button>
          </div>

          <input
            type="number"
            value={size}
            onChange={(e) => setSize(e.target.value)}
            className="input-field w-full"
            placeholder="Position size ($)"
          />

          <div className="flex gap-2">
            <button
              onClick={handleAddPosition}
              className="btn-primary flex-1"
            >
              Add Position
            </button>
            <button
              onClick={() => setShowSizeInput(false)}
              className="btn-secondary"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
