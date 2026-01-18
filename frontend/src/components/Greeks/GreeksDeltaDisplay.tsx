import React from 'react';
import { ArrowRight, Shield, AlertTriangle, Minus } from 'lucide-react';
import type { GreeksDelta } from '../../types';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface GreeksDeltaDisplayProps {
  greeksDelta: GreeksDelta;
}

export const GreeksDeltaDisplay: React.FC<GreeksDeltaDisplayProps> = ({ greeksDelta }) => {
  const { current, simulated, delta, riskChange } = greeksDelta;

  const riskConfig = {
    hedge: {
      icon: Shield,
      color: 'text-success-500',
      bg: 'bg-success-500/20',
      text: 'Hedging Risk',
      description: 'Simulated positions reduce portfolio risk'
    },
    concentrate: {
      icon: AlertTriangle,
      color: 'text-danger-500',
      bg: 'bg-danger-500/20',
      text: 'Concentrating Risk',
      description: 'Simulated positions increase portfolio risk'
    },
    neutral: {
      icon: Minus,
      color: 'text-slate-400',
      bg: 'bg-slate-500/20',
      text: 'Neutral Risk',
      description: 'Simulated positions have minimal impact'
    }
  };

  const config = riskConfig[riskChange];
  const Icon = config.icon;

  // Prepare chart data
  const chartData = [
    {
      greek: 'Delta',
      current: current.delta,
      simulated: simulated.delta,
      change: delta.delta
    },
    {
      greek: 'Gamma',
      current: current.gamma,
      simulated: simulated.gamma,
      change: delta.gamma
    },
    {
      greek: 'Vega',
      current: current.vega,
      simulated: simulated.vega,
      change: delta.vega
    },
    {
      greek: 'Theta',
      current: current.theta,
      simulated: simulated.theta,
      change: delta.theta
    }
  ];

  return (
    <div className="card space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-2xl font-bold">Risk Analysis</h3>
        <div className={`flex items-center gap-2 px-4 py-2 rounded-lg ${config.bg}`}>
          <Icon className={`w-5 h-5 ${config.color}`} />
          <span className={`font-semibold ${config.color}`}>{config.text}</span>
        </div>
      </div>

      <p className="text-slate-400">{config.description}</p>

      {/* Greeks Comparison Grid */}
      <div className="grid grid-cols-4 gap-4">
        {['Delta', 'Gamma', 'Vega', 'Theta'].map((name) => {
          const greekKey = name.toLowerCase() as keyof typeof current;
          const currentVal = current[greekKey];
          const simulatedVal = simulated[greekKey];
          const deltaVal = delta[greekKey];
          const isIncrease = deltaVal > 0;

          return (
            <div key={name} className="greek-card">
              <div className="text-sm text-slate-400 mb-3">{name}</div>
              
              <div className="flex items-center justify-between mb-2">
                <div>
                  <div className="text-xs text-slate-500">Current</div>
                  <div className="text-lg font-bold">{currentVal.toFixed(2)}</div>
                </div>
                <ArrowRight className="w-4 h-4 text-slate-600" />
                <div>
                  <div className="text-xs text-slate-500">Simulated</div>
                  <div className="text-lg font-bold">{simulatedVal.toFixed(2)}</div>
                </div>
              </div>

              <div className={`text-sm font-semibold ${isIncrease ? 'text-danger-500' : 'text-success-500'}`}>
                {isIncrease ? '+' : ''}{deltaVal.toFixed(2)} change
              </div>
            </div>
          );
        })}
      </div>

      {/* Line Chart Visualization */}
      <div className="bg-slate-800/30 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-slate-300 mb-4">Greeks Evolution</h4>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <XAxis dataKey="greek" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1e293b',
                border: '1px solid #475569',
                borderRadius: '8px'
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="current"
              stroke="#6366f1"
              strokeWidth={3}
              name="Current"
              dot={{ fill: '#6366f1', r: 6 }}
            />
            <Line
              type="monotone"
              dataKey="simulated"
              stroke="#0ea5e9"
              strokeWidth={3}
              name="Simulated"
              dot={{ fill: '#0ea5e9', r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Risk Metrics */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-sm text-slate-400 mb-1">Total Delta Change</div>
          <div className={`text-2xl font-bold ${delta.delta >= 0 ? 'text-danger-500' : 'text-success-500'}`}>
            {delta.delta >= 0 ? '+' : ''}{delta.delta.toFixed(2)}
          </div>
        </div>
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-sm text-slate-400 mb-1">Absolute Risk</div>
          <div className="text-2xl font-bold text-primary-400">
            {Math.abs(simulated.delta).toFixed(2)}
          </div>
        </div>
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-sm text-slate-400 mb-1">Risk Change %</div>
          <div className={`text-2xl font-bold ${
            Math.abs(simulated.delta) < Math.abs(current.delta) ? 'text-success-500' : 'text-danger-500'
          }`}>
            {((Math.abs(simulated.delta) / Math.abs(current.delta) - 1) * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </div>
  );
};
