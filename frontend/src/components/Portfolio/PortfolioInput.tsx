import React, { useState } from 'react';
import { Plus, X } from 'lucide-react';
import type { Position } from '../../types';

interface PortfolioInputProps {
  onAddPosition: (position: Omit<Position, 'id' | 'pnl'>) => void;
}

export const PortfolioInput: React.FC<PortfolioInputProps> = ({ onAddPosition }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [formData, setFormData] = useState({
    marketQuestion: '',
    commodity: '',
    side: 'YES' as 'YES' | 'NO',
    size: '',
    entryPrice: '',
    currentPrice: '',
    marketId: ''
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const position: Omit<Position, 'id' | 'pnl'> = {
      marketId: formData.marketId || `market-${Date.now()}`,
      marketQuestion: formData.marketQuestion,
      commodity: formData.commodity,
      side: formData.side,
      size: parseFloat(formData.size),
      entryPrice: parseFloat(formData.entryPrice),
      currentPrice: parseFloat(formData.currentPrice)
    };

    onAddPosition(position);
    setFormData({
      marketQuestion: '',
      commodity: '',
      side: 'YES',
      size: '',
      entryPrice: '',
      currentPrice: '',
      marketId: ''
    });
    setIsOpen(false);
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="btn-primary flex items-center gap-2"
      >
        <Plus className="w-5 h-5" />
        Add Position
      </button>
    );
  }

  return (
    <div className="card animate-slide-up">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-xl font-bold">Add New Position</h3>
        <button
          onClick={() => setIsOpen(false)}
          className="text-slate-400 hover:text-white transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1">
            Market Question
          </label>
          <input
            type="text"
            value={formData.marketQuestion}
            onChange={e => setFormData({ ...formData, marketQuestion: e.target.value })}
            className="input-field w-full"
            placeholder="Will Gold hit $2,800 by January?"
            required
          />
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Commodity
            </label>
            <input
              type="text"
              value={formData.commodity}
              onChange={e => setFormData({ ...formData, commodity: e.target.value })}
              className="input-field w-full"
              placeholder="gold, silver, oil..."
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Side
            </label>
            <select
              value={formData.side}
              onChange={e => setFormData({ ...formData, side: e.target.value as 'YES' | 'NO' })}
              className="input-field w-full"
            >
              <option value="YES">YES</option>
              <option value="NO">NO</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Size ($)
            </label>
            <input
              type="number"
              step="0.01"
              value={formData.size}
              onChange={e => setFormData({ ...formData, size: e.target.value })}
              className="input-field w-full"
              placeholder="1000"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Entry Price
            </label>
            <input
              type="number"
              step="0.01"
              value={formData.entryPrice}
              onChange={e => setFormData({ ...formData, entryPrice: e.target.value })}
              className="input-field w-full"
              placeholder="0.60"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Current Price
            </label>
            <input
              type="number"
              step="0.01"
              value={formData.currentPrice}
              onChange={e => setFormData({ ...formData, currentPrice: e.target.value })}
              className="input-field w-full"
              placeholder="0.65"
              required
            />
          </div>
        </div>

        <div className="flex gap-3 pt-2">
          <button type="submit" className="btn-primary flex-1">
            Add Position
          </button>
          <button
            type="button"
            onClick={() => setIsOpen(false)}
            className="btn-secondary"
          >
            Cancel
          </button>
        </div>
      </form>
    </div>
  );
};
