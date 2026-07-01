import React from 'react';
import { User, ShieldAlert, Sparkles, MessageSquare } from 'lucide-react';

const ChatMessage = ({ msg }) => {
  const isAI = msg.sender === 'ai';

  return (
    <div className={`flex gap-3 max-w-[85%] ${isAI ? 'self-start' : 'self-end flex-row-reverse'}`}>
      <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${isAI ? 'bg-blue-600 text-white' : 'bg-indigo-600 text-white'}`}>
        {isAI ? <Sparkles size={16} /> : <User size={16} />}
      </div>

      <div className={`rounded-2xl p-4 shadow-md leading-relaxed text-sm ${isAI ? 'bg-slate-800 text-slate-100 border border-slate-700/50' : 'bg-blue-600 text-white'}`}>
        <p className="whitespace-pre-line">{msg.text}</p>
        
        {msg.action && (
          <span className="inline-block mt-2 text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 bg-slate-900/60 rounded text-slate-400">
            {msg.action}
          </span>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;
