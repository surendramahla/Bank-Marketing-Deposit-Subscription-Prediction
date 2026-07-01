import React, { useState, useEffect } from 'react';
import API from '../api/api';
import FeatureImportanceChart from '../components/FeatureImportanceChart';
import { Shield, Sparkles, TrendingUp } from 'lucide-react';
import toast from 'react-hot-toast';

const Analytics = () => {
  const [metrics, setMetrics] = useState(null);
  const [fi, setFi] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadMetrics = async () => {
      try {
        const [perfRes, fiRes] = await Promise.all([
          API.get('/analytics/model-performance'),
          API.get('/analytics/feature-importance')
        ]);
        setMetrics(perfRes.data.data[0]);

        const chartData = Object.entries(fiRes.data.data.feature_importance || {})
          .map(([name, value]) => ({ name, value }))
          .sort((a, b) => b.value - a.value);
        setFi(chartData);
      } catch (err) {
        toast.error('Failed to load performance metrics');
      } finally {
        setLoading(false);
      }
    };
    loadMetrics();
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#070b19]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto bg-[#070b19] p-8 space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-black text-white tracking-tight">Model Analytics & Health</h2>
        <p className="text-sm text-slate-400 mt-1">Audit the baseline machine learning performance, metrics, and feature significance.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Metrics details */}
        <div className="bg-[#0f172a]/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Shield size={18} className="text-blue-400" />
            Accuracy & Precision Audits
          </h3>

          <div className="space-y-4">
            <div>
              <p className="text-xs text-slate-400 mb-1">Model Name</p>
              <p className="text-sm font-semibold text-white">{metrics?.model_name || 'Random Forest Pipeline'}</p>
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-900/60 p-3.5 border border-slate-850 rounded-xl">
                <p className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Accuracy</p>
                <p className="text-xl font-black text-white">{(metrics?.accuracy * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-slate-900/60 p-3.5 border border-slate-850 rounded-xl">
                <p className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">F1-Score</p>
                <p className="text-xl font-black text-white">{(metrics?.f1_score * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-slate-900/60 p-3.5 border border-slate-850 rounded-xl">
                <p className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">ROC-AUC</p>
                <p className="text-xl font-black text-white">{(metrics?.roc_auc * 100).toFixed(2)}%</p>
              </div>
              <div className="bg-slate-900/60 p-3.5 border border-slate-850 rounded-xl">
                <p className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Precision</p>
                <p className="text-xl font-black text-white">{(metrics?.precision * 100).toFixed(2)}%</p>
              </div>
            </div>
          </div>
        </div>

        {/* Global SHAP chart */}
        <div className="lg:col-span-2 bg-[#0f172a]/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Sparkles size={18} className="text-blue-400" />
            Global Feature Significance (Full Model SHAP)
          </h3>
          <FeatureImportanceChart data={fi} />
        </div>
      </div>
    </div>
  );
};

export default Analytics;
