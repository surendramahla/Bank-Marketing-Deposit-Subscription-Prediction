import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import Sidebar from './components/Sidebar';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Customers from './pages/Customers';
import Predictions from './pages/Predictions';
import AIAssistant from './pages/AIAssistant';
import Analytics from './pages/Analytics';
import Campaigns from './pages/Campaigns';
import { Toaster } from 'react-hot-toast';

const ProtectedLayout = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen bg-[#070b19] flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500" />
      </div>
    );
  }

  if (!user) {
    return <Navigate to="/login" replace />;
  }

  return (
    <div className="flex h-screen overflow-hidden bg-[#070b19]">
      <Sidebar />
      <main className="flex-1 flex flex-col overflow-hidden">
        {children}
      </main>
    </div>
  );
};

const App = () => {
  return (
    <>
      <Toaster position="top-right" toastOptions={{ style: { backgroundColor: '#1e293b', color: '#fff' } }} />
      <Routes>
        <Route path="/login" element={<Login />} />
        
        {/* Protected Routes */}
        <Route path="/" element={<ProtectedLayout><Dashboard /></ProtectedLayout>} />
        <Route path="/customers" element={<ProtectedLayout><Customers /></ProtectedLayout>} />
        <Route path="/predictions" element={<ProtectedLayout><Predictions /></ProtectedLayout>} />
        <Route path="/chat" element={<ProtectedLayout><AIAssistant /></ProtectedLayout>} />
        <Route path="/analytics" element={<ProtectedLayout><Analytics /></ProtectedLayout>} />
        <Route path="/campaigns" element={<ProtectedLayout><Campaigns /></ProtectedLayout>} />
        
        {/* Fallback */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
};

export default App;
