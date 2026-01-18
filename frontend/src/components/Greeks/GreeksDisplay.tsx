import React from 'react';
import { TrendingUp, TrendingDown, Activity, Zap, Clock, DollarSign } from 'lucide-react';
import type { Greeks } from '../../types';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell
} from 'recharts';

interface GreeksDisplayProps {
  greeks: Greeks;
  title?: string;
}

export const GreeksDisplay: React.FC<GreeksDisplayProps> = ({ greeks, title = "Portfolio Greeks" }) => {
  const greekItems = [
    { name: 'Delta', value: greeks.delta, icon: TrendingUp, color: 'text-blue-400', description: '$ per 1% move' },
    { name: 'Gamma', value: greeks.gamma, icon: Activity, color: 'text-purple-400', description: 'Δ per 1% move' },
    { name: 'Vega', value: greeks.vega, icon: Zap, color: 'text-yellow-400', description: '$ per 1% vol' },
    { name: 'Theta', value: greeks.theta, icon: Clock, color: 'text-red-400', description: '$ per day' },
  ];

  // Prepare data for radar chart
  const radarData = greekItems.map(item => ({
    greek: item.name,
    value: Math.abs(item.value),
    fullMark: 100
  }));

  // Prepare data for bar chart
  const barData = greekItems.map(item => ({
    name: item.name,
    value: item.value
  }));

  return (
    <div className="card space-y-6">
      <h3 className="text-2xl font-bold">{title}</h3>

      {/* Greek Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {greekItems.map((item) => {
          const Icon = item.icon;
          const isPositive = item.value >= 0;

          return (
            <div key={item.name} className="greek-card">
              <div className="flex items-center justify-between mb-2">
                <Icon className={`w-5 h-5 ${item.color}`} />
                {isPositive ? (
                  <TrendingUp className="w-4 h-4 text-success-500" />
                ) : (
                  <TrendingDown className="w-4 h-4 text-danger-500" />
                )}
              </div>
              <div className="text-sm text-slate-400 mb-1">{item.name}</div>
              <div className={`text-2xl font-bold ${isPositive ? 'text-success-500' : 'text-danger-500'}`}>
                {isPositive ? '+' : ''}{item.value.toFixed(2)}
              </div>
              <div className="text-xs text-slate-500 mt-1">{item.description}</div>
            </div>
          );
        })}
      </div>

      {/* Visualizations */}
      <div className="grid grid-cols-2 gap-6 mt-6">
        {/* Radar Chart */}
        <div className="bg-slate-800/30 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-slate-300 mb-4">Greeks Radar</h4>
          <ResponsiveContainer width="100%" height={250}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#475569" />
              <PolarAngleAxis dataKey="greek" stroke="#94a3b8" />
              <PolarRadiusAxis stroke="#475569" />
              <Radar
                name="Greeks"
                dataKey="value"
                stroke="#0ea5e9"
                fill="#0ea5e9"
                fillOpacity={0.6}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* Bar Chart */}
        <div className="bg-slate-800/30 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-slate-300 mb-4">Greeks Breakdown</h4>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={barData}>
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #475569',
                  borderRadius: '8px'
                }}
              />
              <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                {barData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.value >= 0 ? '#10b981' : '#ef4444'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
