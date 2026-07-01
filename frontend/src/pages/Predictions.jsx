import React, { useState } from 'react';
import API from '../api/api';
import PredictionCard from '../components/PredictionCard';
import { Upload, HelpCircle, FileText, ArrowRight, Brain } from 'lucide-react';
import toast from 'react-hot-toast';

const Predictions = () => {
  const [activeTab, setActiveTab] = useState('single');
  const [formData, setFormData] = useState({
    age: 42, job: 'management', marital: 'married', education: 'tertiary',
    default: 'no', balance: 2500, housing: 'yes', loan: 'no',
    contact: 'cellular', day: 15, month: 'may', campaign: 2,
    pdays: -1, previous: 0, poutcome: 'unknown'
  });
  const [singleResult, setSingleResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Bulk state
  const [file, setFile] = useState(null);
  const [bulkResult, setBulkResult] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: ['age', 'balance', 'day', 'campaign', 'pdays', 'previous'].includes(name) 
        ? Number(value) 
        : value
    });
  };

  const handleSingleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setSingleResult(null);
    try {
      const { data } = await API.post('/predictions/single', formData);
      setSingleResult(data.data);
      toast.success('Prediction Successful');
    } catch (err) {
      toast.error(err.response?.data?.error || 'Validation error check your inputs');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleBulkSubmit = async (e) => {
    e.preventDefault();
    if (!file) return toast.error('Please select a CSV file');
    setLoading(true);
    setBulkResult(null);
    const dataForm = new FormData();
    dataForm.append('file', file);
    try {
      const { data } = await API.post('/predictions/bulk', dataForm, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setBulkResult(data.data);
      toast.success('Bulk scoring completed');
    } catch (err) {
      toast.error('Failed to process bulk CSV');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto bg-[#070b19] p-8 space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-black text-white tracking-tight">Predict Deposit Subscription</h2>
        <p className="text-sm text-slate-400 mt-1">Run real-time predictions for individuals or batch score via CSV upload.</p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-slate-800">
        <button
          onClick={() => setActiveTab('single')}
          className={`px-6 py-3 font-semibold text-sm transition-colors border-b-2 ${activeTab === 'single' ? 'border-blue-500 text-white' : 'border-transparent text-slate-400 hover:text-white'}`}
        >
          Single Prediction
        </button>
        <button
          onClick={() => setActiveTab('bulk')}
          className={`px-6 py-3 font-semibold text-sm transition-colors border-b-2 ${activeTab === 'bulk' ? 'border-blue-500 text-white' : 'border-transparent text-slate-400 hover:text-white'}`}
        >
          Bulk CSV Scoring
        </button>
      </div>

      {activeTab === 'single' ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* Form */}
          <form onSubmit={handleSingleSubmit} className="bg-[#0f172a]/80 backdrop-blur-md border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Brain size={18} className="text-blue-400" />
              Customer Demographics & History
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Age</label>
                <input type="number" name="age" value={formData.age} onChange={handleInputChange} className="w-full bg-[#1e293b]/50 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200" required />
              </div>
              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Job</label>
                <select name="job" value={formData.job} onChange={handleInputChange} className="w-full bg-[#1e293b]/50 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200">
                  <option value="management">Management</option>
                  <option value="technician">Technician</option>
                  <option value="entrepreneur">Entrepreneur</option>
                  <option value="blue-collar">Blue-Collar</option>
                  <option value="retired">Retired</option>
                  <option value="services">Services</option>
                  <option value="admin.">Admin</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Marital Status</label>
                <select name="marital" value={formData.marital} onChange={handleInputChange} className="w-full bg-[#1e293b]/50 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200">
                  <option value="married">Married</option>
                  <option value="single">Single</option>
                  <option value="divorced">Divorced</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Education</label>
                <select name="education" value={formData.education} onChange={handleInputChange} className="w-full bg-[#1e293b]/50 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200">
                  <option value="tertiary">Tertiary</option>
                  <option value="secondary">Secondary</option>
                  <option value="primary">Primary</option>
                  <option value="unknown">Unknown</option>
                </select>
              </div>
              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Yearly Balance (€)</label>
                <input type="number" name="balance" value={formData.balance} onChange={handleInputChange} className="w-full bg-[#1e293b]/50 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200" required />
              </div>
              <div>
                <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Housing Loan</label>
                <select name="housing" value={formData.housing} onChange={handleInputChange} className="w-full bg-[#1e293b]/50 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200">
                  <option value="no">No</option>
                  <option value="yes">Yes</option>
                </select>
              </div>
            </div>

            <button type="submit" disabled={loading} className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl shadow-lg shadow-blue-900/30 transition-all flex items-center justify-center">
              {loading ? 'Analyzing Profile...' : 'Execute AI Prediction'}
            </button>
          </form>

          {/* Result view */}
          <div className="space-y-6">
            {singleResult ? (
              <PredictionCard result={singleResult} />
            ) : (
              <div className="bg-[#0f172a]/50 border border-dashed border-slate-800 rounded-2xl p-12 text-center text-slate-500">
                <HelpCircle size={48} className="mx-auto mb-4 text-slate-700" />
                <p>Fill out the profile fields and click predict to get live AI analysis.</p>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          <form onSubmit={handleBulkSubmit} className="bg-[#0f172a]/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <h3 className="text-base font-bold text-white">Batch Lead CSV Scoring</h3>
            <div className="border-2 border-dashed border-slate-850 hover:border-blue-500/50 rounded-xl p-8 text-center transition-colors cursor-pointer relative">
              <input type="file" onChange={handleFileChange} accept=".csv" className="absolute inset-0 opacity-0 cursor-pointer" />
              <Upload size={32} className="mx-auto mb-3 text-slate-500" />
              <p className="text-sm font-semibold text-white">{file ? file.name : 'Select or drop customer CSV'}</p>
              <p className="text-xs text-slate-500 mt-1">Accepts only .csv format up to 32MB</p>
            </div>
            <button type="submit" disabled={loading || !file} className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl transition-all disabled:opacity-40">
              {loading ? 'Scoring Batch File...' : 'Start Batch Prediction'}
            </button>
          </form>

          <div>
            {bulkResult ? (
              <div className="bg-[#0f172a]/80 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
                <h4 className="font-bold text-white text-base">Bulk Scoring Summary</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-900/60 p-4 border border-slate-850 rounded-xl">
                    <p className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Total Records</p>
                    <p className="text-2xl font-black text-white">{bulkResult.total_records}</p>
                  </div>
                  <div className="bg-slate-900/60 p-4 border border-slate-850 rounded-xl">
                    <p className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">Projected Subscriptions</p>
                    <p className="text-2xl font-black text-emerald-400">{bulkResult.summary.predicted_yes}</p>
                  </div>
                </div>
                <div className="pt-4">
                  <a href="http://localhost:8000/predict/bulk/template" className="flex items-center justify-center gap-2 p-2.5 bg-slate-800 hover:bg-slate-750 text-slate-200 text-sm font-semibold rounded-lg border border-slate-700/50 transition-colors">
                    <FileText size={16} />
                    Download Scored CSV Results
                  </a>
                </div>
              </div>
            ) : (
              <div className="bg-[#0f172a]/50 border border-dashed border-slate-800 rounded-2xl p-12 text-center text-slate-500">
                <Upload size={48} className="mx-auto mb-4 text-slate-700" />
                <p>Upload a customer bank CSV file to trigger batch lead segmenting.</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default Predictions;
