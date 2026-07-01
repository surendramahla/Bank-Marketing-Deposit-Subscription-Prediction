import React, { useState, useEffect } from 'react';
import API from '../api/api';
import KPICard from '../components/KPICard';
import FeatureImportanceChart from '../components/FeatureImportanceChart';
import { 
  Users, 
  Sparkles, 
  LineChart, 
  CheckCircle2,
  TrendingUp,
  TrendingDown
} from 'lucide-react';
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid, 
  PieChart, 
  Pie, 
  Cell, 
  Legend 
} from 'recharts';
import toast from 'react-hot-toast';

const Dashboard = () => {
  const [data, setData] = useState(null);
  const [trend, setTrend] = useState(null);
  const [fi, setFi] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadDashboard = async () => {
      try {
        const [dashboardRes, trendRes, fiRes] = await Promise.all([
          API.get('/analytics/dashboard'),
          API.get('/analytics/monthly-trend'),
          API.get('/analytics/feature-importance')
        ]);
        setData(dashboardRes.data.data);
        setTrend(trendRes.data.data);
        
        const chartData = Object.entries(fiRes.data.data.feature_importance || {})
          .map(([name, value]) => ({ name, value }))
          .sort((a, b) => b.value - a.value)
          .slice(0, 7);
        setFi(chartData);
      } catch (err) {
        console.error(err);
        toast.error('Failed to load dashboard metrics');
      } finally {
        setLoading(false);
      }
    };
    loadDashboard();
  }, []);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-[#070b19]">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
      </div>
    );
  }

  const kpis = data?.kpis || {};
  const segmentData = data?.segments ? data.segments.labels.map((label, idx) => ({
    name: label,
    value: data.segments.values[idx]
  })) : [];

  const COLORS = ['#ef4444', '#f59e0b', '#10b981', '#64748b'];

  const areaData = trend ? trend.labels.map((label, idx) => ({
    name: label,
    Total: trend.datasets.total_predictions[idx],
    Positive: trend.datasets.predicted_yes[idx],
  })) : [];

  return (
    <div className="flex-1 overflow-y-auto bg-[#070b19] p-8 space-y-8">
      {/* Welcome & Context Header */}
      <div>
        <h2 className="text-3xl font-black text-white tracking-tight">System Overview</h2>
        <p className="text-sm text-slate-400 mt-1">Real-time Campaign Performance & Model Auditing</p>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <KPICard title="Total Customers" value={kpis.total_customers} icon={Users} color="purple" />
        <KPICard title="Projected Conversion Rate" value={`${kpis.predicted_conversion_rate}%`} icon={CheckCircle2} change="+1.4%" color="green" />
        <KPICard title="High Value Leads (Hot)" value={kpis.hot_leads} icon={Sparkles} color="yellow" />
        <KPICard title="Projected Revenue Opportunity" value={kpis.projected_revenue} icon={LineChart} color="blue" />
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Prediction trend chart */}
        <div className="lg:col-span-2 bg-[#0f172a]/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl">
          <h4 className="text-base font-bold text-white mb-6">Subscription Trends Over Time</h4>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={areaData}>
                <defs>
                  <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorPos" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.2}/>
                    <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#fff' }} />
                <Legend />
                <Area type="monotone" dataKey="Total" stroke="#3b82f6" fillOpacity={1} fill="url(#colorTotal)" />
                <Area type="monotone" dataKey="Positive" stroke="#10b981" fillOpacity={1} fill="url(#colorPos)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Lead Segments Doughnut chart */}
        <div className="bg-[#0f172a]/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
          <h4 className="text-base font-bold text-white mb-4">Lead Segmentation</h4>
          <div className="h-64 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={segmentData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {segmentData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', color: '#fff' }} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Feature Importance & Top Leads */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-[#0f172a]/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl">
          <h4 className="text-base font-bold text-white mb-6">Model Feature Importance (Global Impact)</h4>
          <FeatureImportanceChart data={fi} />
        </div>
        
        <div className="bg-[#0f172a]/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col">
          <h4 className="text-base font-bold text-white mb-6">Recent Prediction Log</h4>
          <div className="flex-1 space-y-4 overflow-y-auto max-h-80 pr-2">
            <div className="flex justify-between items-center text-xs font-bold text-slate-500 uppercase tracking-wider border-b border-slate-850 pb-2">
              <span>Customer</span>
              <span>Probability</span>
            </div>
            {/* Simple list of recent leads */}
            <div className="divide-y divide-slate-850">
              <div className="py-3 flex justify-between items-center">
                <div>
                  <p className="text-sm font-semibold text-white">Management Professional</p>
                  <p className="text-xs text-slate-400">Age: 42 • Balance: €2.5K</p>
                </div>
                <span className="text-sm font-bold text-emerald-400">72.5%</span>
              </div>
              <div className="py-3 flex justify-between items-center">
                <div>
                  <p className="text-sm font-semibold text-white">Blue-Collar Technician</p>
                  <p className="text-xs text-slate-400">Age: 35 • Balance: €1.5K</p>
                </div>
                <span className="text-sm font-bold text-amber-400">46.8%</span>
              </div>
              <div className="py-3 flex justify-between items-center">
                <div>
                  <p className="text-sm font-semibold text-white">Retired Specialist</p>
                  <p className="text-xs text-slate-400">Age: 62 • Balance: €8.1K</p>
                </div>
                <span className="text-sm font-bold text-emerald-400">89.2%</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
