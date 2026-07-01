import React, { useState, useEffect } from 'react';
import API from '../api/api';
import { Search, Plus, Filter, RefreshCw, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';

const Customers = () => {
  const [customers, setCustomers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [segment, setSegment] = useState('');
  const [page, setPage] = useState(1);
  const [pagination, setPagination] = useState({});
  const [predictingId, setPredictingId] = useState(null);

  const loadCustomers = async () => {
    setLoading(true);
    try {
      const { data } = await API.get('/customers', {
        params: { page, limit: 10, search, segment }
      });
      setCustomers(data.data);
      setPagination(data.pagination);
    } catch (err) {
      toast.error('Failed to load customers');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCustomers();
  }, [page, segment]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setPage(1);
    loadCustomers();
  };

  const handlePredict = async (id) => {
    setPredictingId(id);
    try {
      const { data } = await API.post(`/customers/${id}/predict`);
      toast.success(`Prediction Complete: ${(data.data.prediction.probability).toFixed(1)}%`);
      loadCustomers(); // refresh
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setPredictingId(null);
    }
  };

  const segmentColors = {
    Hot: 'bg-red-500/10 text-red-400 border-red-500/20',
    Warm: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    Cold: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[#070b19] p-8 space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-black text-white tracking-tight">Customer Database</h2>
          <p className="text-sm text-slate-400 mt-1">Manage prospects and trigger AI scoring.</p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-lg shadow-lg shadow-blue-900/30 transition-colors">
          <Plus size={18} />
          Add Customer
        </button>
      </div>

      {/* Filters Bar */}
      <div className="bg-[#0f172a]/80 backdrop-blur-md border border-slate-800 rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <form onSubmit={handleSearchSubmit} className="flex-1 flex gap-2 max-w-md">
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
            <input
              type="text"
              placeholder="Search by job or education..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="w-full bg-[#1e293b]/50 border border-slate-800 rounded-lg pl-10 pr-4 py-2 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500 text-sm"
            />
          </div>
          <button type="submit" className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-white font-semibold rounded-lg text-sm transition-colors">
            Search
          </button>
        </form>

        <div className="flex gap-3">
          <div className="flex items-center gap-2">
            <Filter size={16} className="text-slate-500" />
            <select
              value={segment}
              onChange={(e) => { setSegment(e.target.value); setPage(1); }}
              className="bg-[#1e293b]/50 border border-slate-800 text-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
            >
              <option value="">All Segments</option>
              <option value="Hot">Hot (High Prob)</option>
              <option value="Warm">Warm (Medium Prob)</option>
              <option value="Cold">Cold (Low Prob)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Customer List */}
      <div className="bg-[#0f172a]/80 border border-slate-800 rounded-2xl shadow-xl overflow-hidden">
        {loading ? (
          <div className="p-12 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mx-auto" />
          </div>
        ) : customers.length === 0 ? (
          <div className="p-12 text-center text-slate-500">
            <AlertCircle size={36} className="mx-auto mb-4 text-slate-600" />
            <p>No customer records matched your query</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-850 text-xs font-bold text-slate-400 uppercase tracking-wider bg-slate-900/30">
                  <th className="p-4 pl-6">Profile</th>
                  <th className="p-4">Balance</th>
                  <th className="p-4">Loans</th>
                  <th className="p-4">Lead Score</th>
                  <th className="p-4 text-right pr-6">Trigger AI</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-850">
                {customers.map((c) => (
                  <tr key={c.id} className="hover:bg-slate-800/20 transition-colors">
                    <td className="p-4 pl-6">
                      <p className="font-semibold text-white capitalize">{c.job.replace(/_/g, ' ')}</p>
                      <p className="text-xs text-slate-400">Age: {c.age} • Marital: {c.marital} • Education: {c.education}</p>
                    </td>
                    <td className="p-4">
                      <span className="font-mono text-sm text-slate-200">€{c.balance.toLocaleString()}</span>
                    </td>
                    <td className="p-4 text-xs space-y-1">
                      <p>Housing: <span className={c.housing === 'yes' ? 'text-rose-400' : 'text-emerald-400'}>{c.housing}</span></p>
                      <p>Personal: <span className={c.loan === 'yes' ? 'text-rose-400' : 'text-emerald-400'}>{c.loan}</span></p>
                    </td>
                    <td className="p-4">
                      {c.last_predicted_at ? (
                        <div className="flex items-center gap-2">
                          <span className={`px-2.5 py-0.5 rounded-full text-xs font-bold border ${segmentColors[c.lead_segment]}`}>
                            {c.lead_segment}
                          </span>
                          <span className="text-sm font-bold text-white">{(c.conversion_probability * 100).toFixed(1)}%</span>
                        </div>
                      ) : (
                        <span className="text-xs text-slate-500">Not Scored</span>
                      )}
                    </td>
                    <td className="p-4 text-right pr-6">
                      <button
                        onClick={() => handlePredict(c.id)}
                        disabled={predictingId === c.id}
                        className="p-2 hover:bg-blue-600/20 text-blue-400 hover:text-white border border-blue-900/30 rounded-lg transition-all disabled:opacity-40"
                      >
                        {predictingId === c.id ? <RefreshCw className="animate-spin" size={16} /> : 'Predict'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination footer */}
        {pagination.totalPages > 1 && (
          <div className="p-4 border-t border-slate-850 flex items-center justify-between">
            <span className="text-xs text-slate-500">Page {page} of {pagination.totalPages}</span>
            <div className="flex gap-2">
              <button
                disabled={!pagination.hasPrev}
                onClick={() => setPage(page - 1)}
                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 font-semibold rounded text-xs transition-colors"
              >
                Previous
              </button>
              <button
                disabled={!pagination.hasNext}
                onClick={() => setPage(page + 1)}
                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 disabled:opacity-40 text-slate-300 font-semibold rounded text-xs transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Customers;
