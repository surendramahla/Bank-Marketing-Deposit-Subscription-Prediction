import React, { useState, useEffect } from 'react';
import API from '../api/api';
import { Megaphone, RefreshCw, AlertCircle, Plus } from 'lucide-react';
import toast from 'react-hot-toast';

const Campaigns = () => {
  const [campaigns, setCampaigns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCampaign, setSelectedCampaign] = useState(null);
  const [aiStrategy, setAiStrategy] = useState('');
  const [loadingStrategy, setLoadingStrategy] = useState(false);

  const loadCampaigns = async () => {
    setLoading(true);
    try {
      const { data } = await API.get('/campaigns');
      setCampaigns(data.data);
    } catch (err) {
      toast.error('Failed to load campaigns');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCampaigns();
  }, []);

  const loadAiStrategy = async (id) => {
    setLoadingStrategy(true);
    setAiStrategy('');
    try {
      const { data } = await API.get(`/campaigns/${id}/ai-strategy`);
      setAiStrategy(data.data.strategy);
    } catch (err) {
      toast.error('Failed to generate AI recommendations. Verify API key.');
    } finally {
      setLoadingStrategy(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[#070b19] p-8 space-y-8">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-black text-white tracking-tight">Marketing Campaigns</h2>
          <p className="text-sm text-slate-400 mt-1">Deploy campaign configurations and review AI targeted recommendations.</p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white font-semibold rounded-lg shadow-lg shadow-blue-900/30 transition-colors">
          <Plus size={18} />
          Create Campaign
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Campaign List */}
        <div className="lg:col-span-2 bg-[#0f172a]/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Megaphone size={18} className="text-blue-400" />
            Active & Draft Campaigns
          </h3>

          {loading ? (
            <div className="p-12 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500 mx-auto" />
            </div>
          ) : campaigns.length === 0 ? (
            <div className="p-12 text-center text-slate-500">
              <AlertCircle size={36} className="mx-auto mb-4 text-slate-600" />
              <p>No campaign history found</p>
            </div>
          ) : (
            <div className="space-y-4">
              {campaigns.map((c) => (
                <div
                  key={c.id}
                  onClick={() => { setSelectedCampaign(c); loadAiStrategy(c.id); }}
                  className={`p-4 rounded-xl border transition-all cursor-pointer ${selectedCampaign?.id === c.id ? 'bg-blue-600/10 border-blue-500/50' : 'bg-slate-900/40 border-slate-850 hover:border-slate-800'}`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h4 className="font-bold text-white">{c.name}</h4>
                    <span className="text-xs font-bold uppercase tracking-wider px-2 py-0.5 bg-slate-800 border border-slate-700/50 text-slate-400 rounded">
                      {c.status}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 leading-relaxed mb-4">{c.description || 'No description provided'}</p>
                  
                  <div className="grid grid-cols-3 gap-4 text-center border-t border-slate-850 pt-3">
                    <div>
                      <p className="text-[10px] font-bold text-slate-500 uppercase">Target Segment</p>
                      <p className="text-xs text-slate-300 font-semibold">{c.target_segment || 'All'}</p>
                    </div>
                    <div>
                      <p className="text-[10px] font-bold text-slate-500 uppercase">Contacted</p>
                      <p className="text-xs text-slate-300 font-semibold">{c.total_contacted}</p>
                    </div>
                    <div>
                      <p className="text-[10px] font-bold text-slate-500 uppercase">Conversion Rate</p>
                      <p className="text-xs text-emerald-400 font-bold">{c.conversion_rate}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* AI Recommendations */}
        <div className="bg-[#0f172a]/80 border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col justify-between">
          <div>
            <h3 className="text-base font-bold text-white mb-4">AI Campaign Advisor</h3>
            
            {selectedCampaign ? (
              <div className="space-y-4">
                <p className="text-xs font-bold text-slate-400 uppercase tracking-wider">Recommendations for {selectedCampaign.name}</p>
                {loadingStrategy ? (
                  <div className="py-12 text-center text-slate-500 text-xs flex items-center justify-center gap-2">
                    <RefreshCw className="animate-spin" size={14} />
                    Analyzing strategy...
                  </div>
                ) : (
                  <div className="text-xs text-slate-300 leading-relaxed bg-slate-900/60 border border-slate-850 rounded-xl p-4 whitespace-pre-line max-h-96 overflow-y-auto">
                    {aiStrategy || 'AI feedback could not be loaded.'}
                  </div>
                )}
              </div>
            ) : (
              <div className="py-12 text-center text-slate-500 text-xs border border-dashed border-slate-800 rounded-xl">
                <Megaphone className="mx-auto mb-3 text-slate-700" size={32} />
                Select a campaign to request targeting recommendations.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Campaigns;
