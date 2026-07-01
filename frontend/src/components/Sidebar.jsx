import React, { useState } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { 
  LayoutDashboard, 
  Users, 
  BrainCircuit, 
  MessageSquareCode, 
  LineChart, 
  Megaphone, 
  LogOut, 
  Menu,
  X
} from 'lucide-react';

const Sidebar = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const navItems = [
    { name: 'Dashboard', path: '/', icon: LayoutDashboard },
    { name: 'Customers', path: '/customers', icon: Users },
    { name: 'Predictions', path: '/predictions', icon: BrainCircuit },
    { name: 'AI Copilot', path: '/chat', icon: MessageSquareCode },
    { name: 'Analytics', path: '/analytics', icon: LineChart },
    { name: 'Campaigns', path: '/campaigns', icon: Megaphone },
  ];

  return (
    <aside className={`bg-[#0f172a] border-r border-slate-800 text-slate-300 flex flex-col transition-all duration-300 ${collapsed ? 'w-20' : 'w-64'}`}>
      {/* Header */}
      <div className="p-4 border-b border-slate-800 flex items-center justify-between">
        {!collapsed && <span className="font-bold text-lg bg-gradient-to-r from-blue-400 to-indigo-500 bg-clip-text text-transparent">BankAI Pro</span>}
        <button onClick={() => setCollapsed(!collapsed)} className="p-2 hover:bg-slate-800 rounded">
          {collapsed ? <Menu size={20} /> : <X size={20} />}
        </button>
      </div>

      {/* Nav Links */}
      <nav className="flex-1 p-4 space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.name}
            to={item.path}
            className={({ isActive }) => 
              `flex items-center gap-3 p-3 rounded-lg transition-colors ${
                isActive 
                  ? 'bg-blue-600 text-white font-semibold shadow-md shadow-blue-900/30' 
                  : 'hover:bg-slate-800 hover:text-white'
              }`
            }
          >
            <item.icon size={20} />
            {!collapsed && <span>{item.name}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Footer Profile */}
      <div className="p-4 border-t border-slate-800">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-indigo-600 flex items-center justify-center text-white font-bold">
            {user?.username?.substring(0, 2).toUpperCase() || 'US'}
          </div>
          {!collapsed && (
            <div className="overflow-hidden">
              <p className="text-sm font-semibold text-white truncate">{user?.username}</p>
              <p className="text-xs text-slate-400 truncate">{user?.role?.toUpperCase()}</p>
            </div>
          )}
        </div>
        <button 
          onClick={handleLogout}
          className="w-full flex items-center justify-center gap-2 p-2 bg-slate-850 hover:bg-red-900/30 border border-slate-800 rounded-lg text-red-400 text-sm transition-colors"
        >
          <LogOut size={16} />
          {!collapsed && <span>Sign Out</span>}
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;
