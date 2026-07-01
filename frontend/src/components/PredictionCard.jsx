import React from 'react';
import { Sparkles, TrendingUp, AlertTriangle } from 'lucide-react';

const PredictionCard = ({ result }) => {
  if (!result) return null;

  const isYes = result.prediction === 'yes' || result.subscribed === 'yes';
  const prob = result.probability ?? (result.conversion_probability * 100) ?? 0;
  const priority = result.priority || (prob > 70 ? 'High' : prob > 40 ? 'Medium' : 'Low');

  const priorityColors = {
    High: 'bg-red-500/10 text-red-400 border-red-500/20',
    Medium: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    Low: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  };

  return (
    <div className="bg-[#0f172a]/90 border border-slate-800 rounded-2xl p-6 shadow-xl relative overflow-hidden">
      {/* Background Glow */}
      <div className={`absolute -right-16 -top-16 w-36 h-36 rounded-full blur-3xl opacity-20 ${isYes ? 'bg-emerald-500' : 'bg-red-500'}`} />

      <h4 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
        <Sparkles size={18} className="text-blue-400" />
        AI Prediction Analysis
      </h4>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 items-center">
        {/* Gauge Section */}
        <div className="flex flex-col items-center">
          <div className="relative w-32 h-32 flex items-center justify-center">
            {/* Simple Circular Track */}
            <svg className="w-full h-full transform -rotate-90">
              <circle
                cx="64"
                cy="64"
                r="52"
                stroke="#1e293b"
                strokeWidth="10"
                fill="transparent"
              />
              <circle
                cx="64"
                cy="64"
                r="52"
                stroke={isYes ? '#10b981' : '#f43f5e'}
                strokeWidth="10"
                fill="transparent"
                strokeDasharray={2 * Math.PI * 52}
                strokeDashoffset={2 * Math.PI * 52 * (1 - prob / 100)}
                strokeLinecap="round"
              />
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className="text-2xl font-black text-white">{prob.toFixed(1)}%</span>
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">Probability</span>
            </div>
          </div>
        </div>

        {/* Prediction Status & Priority */}
        <div className="space-y-4">
          <div>
            <p className="text-xs text-slate-400 mb-1">Target Subscription Model Result</p>
            <div className="flex items-center gap-2">
              <span className={`text-xl font-bold ${isYes ? 'text-emerald-400' : 'text-rose-400'}`}>
                {isYes ? 'Highly Likely' : 'Unlikely'} to Subscribe
              </span>
            </div>
          </div>

          <div className="flex gap-4">
            <div>
              <p className="text-xs text-slate-400 mb-1">Lead Priority</p>
              <span className={`inline-flex px-3 py-1 rounded-full text-xs font-bold border ${priorityColors[priority]}`}>
                {priority} Priority
              </span>
            </div>
            {result.confidence_band && (
              <div>
                <p className="text-xs text-slate-400 mb-1">Confidence Band</p>
                <span className="text-sm font-semibold text-white">
                  {result.confidence_band.low}% - {result.confidence_band.high}%
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Action Strategy */}
        <div className="bg-[#1e293b]/40 border border-slate-800 rounded-xl p-4 flex gap-3 h-full items-start">
          <TrendingUp className="text-blue-400 shrink-0 mt-0.5" size={18} />
          <div>
            <h5 className="text-xs font-bold text-white uppercase tracking-wider mb-1">Actionable Strategy</h5>
            <p className="text-xs text-slate-300 leading-relaxed">
              {result.strategy || 'Unprocessed. Run single prediction to obtain targeting insights.'}
            </p>
          </div>
        </div>
      </div>

      {/* SHAP top factors if present */}
      {result.top_positive_factors && result.top_positive_factors.length > 0 && (
        <div className="mt-6 pt-6 border-t border-slate-800 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h5 className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-2 flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
              Top Positive Drivers
            </h5>
            <ul className="space-y-1">
              {result.top_positive_factors.map((f, i) => (
                <li key={i} className="text-xs text-slate-300 flex justify-between">
                  <span className="font-semibold text-slate-400 capitalize">{f.feature.replace(/_/g, ' ')}</span>
                  <span className="text-emerald-500">+{f.impact.toFixed(4)}</span>
                </li>
              ))}
            </ul>
          </div>
          {result.top_negative_factors && result.top_negative_factors.length > 0 && (
            <div>
              <h5 className="text-xs font-bold text-rose-400 uppercase tracking-wider mb-2 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-rose-400" />
                Top Negative Drivers
              </h5>
              <ul className="space-y-1">
                {result.top_negative_factors.map((f, i) => (
                  <li key={i} className="text-xs text-slate-300 flex justify-between">
                    <span className="font-semibold text-slate-400 capitalize">{f.feature.replace(/_/g, ' ')}</span>
                    <span className="text-rose-500">{f.impact.toFixed(4)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PredictionCard;
