import React from 'react';

const KPICard = ({ title, value, change, icon: Icon, color = 'blue' }) => {
  const colorMap = {
    blue: 'from-blue-600 to-indigo-600 text-blue-400 border-blue-900/30',
    green: 'from-emerald-600 to-teal-600 text-emerald-400 border-emerald-900/30',
    yellow: 'from-amber-600 to-orange-600 text-amber-400 border-amber-900/30',
    purple: 'from-indigo-600 to-purple-600 text-indigo-400 border-indigo-900/30',
  };

  return (
    <div className={`bg-[#0f172a]/80 backdrop-blur-md border border-slate-800 rounded-xl p-6 shadow-lg flex items-center justify-between`}>
      <div>
        <p className="text-sm font-semibold text-slate-400 mb-1">{title}</p>
        <h3 className="text-3xl font-extrabold text-white tracking-tight">{value}</h3>
        {change && (
          <p className="text-xs font-semibold text-emerald-400 mt-2 flex items-center gap-1">
            <span>{change}</span>
            <span className="text-slate-500 font-normal">vs last month</span>
          </p>
        )}
      </div>
      <div className={`p-4 rounded-xl bg-gradient-to-br ${colorMap[color]} bg-opacity-20`}>
        <Icon size={24} className="text-white" />
      </div>
    </div>
  );
};

export default KPICard;
