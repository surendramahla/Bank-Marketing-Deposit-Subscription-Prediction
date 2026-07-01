import React, { useState, useEffect, useRef } from 'react';
import API from '../api/api';
import ChatMessage from '../components/ChatMessage';
import { Send, Sparkles, MessageSquare, AlertCircle, RefreshCw } from 'lucide-react';
import toast from 'react-hot-toast';

const AIAssistant = () => {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: 'Hello! I am your AI Banking Copilot. I can assist you with explaining predictions, recommending marketing strategies, and generating custom outreach call scripts and emails.\n\nSelect a customer from the database to enable tailored insights.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  
  // Selected customer states
  const [customers, setCustomers] = useState([]);
  const [selectedCust, setSelectedCust] = useState(null);

  const chatEndRef = useRef(null);

  useEffect(() => {
    const loadCustomers = async () => {
      try {
        const { data } = await API.get('/customers?limit=5');
        setCustomers(data.data);
      } catch (err) {
        console.warn('Failed to load sample customers for chat context');
      }
    };
    loadCustomers();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMsg = { sender: 'user', text: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const { data } = await API.post('/chat/ask', { question: input });
      setMessages((prev) => [...prev, { sender: 'ai', text: data.data.response, action: 'General Q&A' }]);
    } catch (err) {
      toast.error('AI assistant offline. Configure Gemini/OpenAI key.');
    } finally {
      setLoading(false);
    }
  };

  const handleQuickAction = async (action, label) => {
    if (!selectedCust) {
      return toast.error('Please select a customer context first');
    }
    setLoading(true);
    setMessages((prev) => [...prev, { sender: 'user', text: `Triggering: ${label} for Selected Customer` }]);

    try {
      // Map selected customer to backend schema
      const customerPayload = {
        age: selectedCust.age,
        job: selectedCust.job,
        marital: selectedCust.marital,
        education: selectedCust.education,
        default: selectedCust.default_credit || 'no',
        balance: selectedCust.balance,
        housing: selectedCust.housing,
        loan: selectedCust.loan,
        contact: selectedCust.contact || 'unknown',
        day: selectedCust.day || 15,
        month: selectedCust.month || 'may',
        campaign: selectedCust.campaign || 1,
        pdays: selectedCust.pdays ?? -1,
        previous: selectedCust.previous ?? 0,
        poutcome: selectedCust.poutcome || 'unknown'
      };

      const { data } = await API.post('/chat/quick-action', {
        action,
        customer: customerPayload,
        customer_id: selectedCust.id
      });

      setMessages((prev) => [...prev, { sender: 'ai', text: data.data.response, action: label }]);
    } catch (err) {
      toast.error('Quick action failed. Check service configuration.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 overflow-hidden bg-[#070b19] p-8 flex gap-6 h-full">
      {/* Main chat interface */}
      <div className="flex-1 bg-[#0f172a]/80 border border-slate-800 rounded-2xl flex flex-col h-[calc(100vh-64px-32px)] shadow-xl overflow-hidden">
        {/* Header */}
        <div className="p-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/40">
          <div className="flex items-center gap-2">
            <Sparkles className="text-blue-400" size={18} />
            <span className="font-bold text-white text-sm">AI Copilot Chat</span>
          </div>
          {selectedCust && (
            <span className="text-xs text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">
              Active Context: {selectedCust.job} ({selectedCust.age}y)
            </span>
          )}
        </div>

        {/* Messages */}
        <div className="flex-1 p-6 overflow-y-auto space-y-6 flex flex-col">
          {messages.map((m, idx) => (
            <ChatMessage key={idx} msg={m} />
          ))}
          {loading && (
            <div className="self-start flex gap-2 items-center text-xs text-slate-500 bg-slate-800/40 border border-slate-700/30 px-4 py-2.5 rounded-full">
              <RefreshCw className="animate-spin" size={14} />
              AI is writing...
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input area */}
        <form onSubmit={handleSend} className="p-4 border-t border-slate-800 bg-slate-900/40 flex gap-2">
          <input
            type="text"
            placeholder="Ask a general banking or FAQ question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1 bg-[#1e293b]/50 border border-slate-850 rounded-lg px-4 py-2.5 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500 text-sm"
          />
          <button type="submit" className="p-2.5 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors">
            <Send size={18} />
          </button>
        </form>
      </div>

      {/* Context side panel */}
      <div className="w-80 bg-[#0f172a]/40 border border-slate-800 rounded-2xl p-6 flex flex-col justify-between shadow-xl">
        <div className="space-y-6">
          <div>
            <h4 className="font-bold text-white text-sm flex items-center gap-2">
              <MessageSquare size={16} className="text-blue-400" />
              AI Copilot Actions
            </h4>
            <p className="text-xs text-slate-500 mt-1">Run specific prompt templates against the selected client.</p>
          </div>

          {/* Customer list selector */}
          <div className="space-y-2">
            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider">Select Client Context</label>
            <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
              {customers.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setSelectedCust(c)}
                  className={`w-full text-left p-2.5 rounded-lg border text-xs transition-all flex justify-between items-center ${selectedCust?.id === c.id ? 'bg-blue-600/10 border-blue-500/50 text-white' : 'bg-slate-900/40 border-slate-850 text-slate-400 hover:border-slate-800'}`}
                >
                  <span className="capitalize font-semibold">{c.job.replace(/_/g, ' ')}</span>
                  <span className="font-mono opacity-80">€{c.balance.toLocaleString()}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Quick actions buttons */}
          <div className="space-y-2 pt-2 border-t border-slate-850">
            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider block mb-2">Prompt Templates</label>
            <div className="grid grid-cols-1 gap-2">
              <button onClick={() => handleQuickAction('explain', 'Explain Prediction')} className="w-full text-left py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg transition-colors border border-slate-750">
                Why is this prediction 92%?
              </button>
              <button onClick={() => handleQuickAction('strategy', 'Next Best Strategy')} className="w-full text-left py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg transition-colors border border-slate-750">
                What should the bank employee do?
              </button>
              <button onClick={() => handleQuickAction('call_script', 'Call Script')} className="w-full text-left py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg transition-colors border border-slate-750">
                Generate a call script
              </button>
              <button onClick={() => handleQuickAction('email', 'Marketing Email')} className="w-full text-left py-2 px-3 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg transition-colors border border-slate-750">
                Generate a marketing email
              </button>
            </div>
          </div>
        </div>

        {!selectedCust && (
          <div className="flex gap-2 items-center text-xs text-amber-500 bg-amber-500/10 border border-amber-500/20 p-3 rounded-lg mt-4">
            <AlertCircle size={16} className="shrink-0" />
            <span>Please select a client to activate prompt templates.</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIAssistant;
