import { useEffect, useMemo, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { FiBarChart2, FiMessageSquare } from 'react-icons/fi';
import Navbar from './components/Navbar.jsx';
import Sidebar from './components/Sidebar.jsx';
import MobileTabs from './components/MobileTabs.jsx';
import AnalyticsSidebar from './components/AnalyticsSidebar.jsx';
import ChatPage from './pages/ChatPage.jsx';
import Dashboard from './pages/Dashboard.jsx';
import ToastContainer from './components/ToastContainer.jsx';
import { useChat } from './hooks/useChat.js';
import { useStats } from './hooks/useStats.js';
import { useToasts } from './hooks/useToasts.js';

const pages = {
  chat: { label: 'Chat', icon: FiMessageSquare },
  dashboard: { label: 'Dashboard', icon: FiBarChart2 }
};

export default function App() {
  const [activePage, setActivePage] = useState('chat');
  const toasts = useToasts();
  const chat = useChat({ onToast: toasts.addToast });
  const stats = useStats({ onToast: toasts.addToast });

  const activeMeta = useMemo(() => pages[activePage], [activePage]);

  useEffect(() => {
    const onKeyDown = (event) => {
      const isModifier = event.ctrlKey || event.metaKey;
      if (!isModifier) return;
      if (event.key === '1') setActivePage('chat');
      if (event.key === '2') setActivePage('dashboard');
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  return (
    <div className="min-h-screen bg-ink-950 text-zinc-100">
      <Navbar activePage={activeMeta.label} />
      <div className="mx-auto flex min-h-[calc(100vh-72px)] w-full max-w-[1720px] gap-4 px-4 pb-4 lg:px-6">
        <Sidebar activePage={activePage} pages={pages} onPageChange={setActivePage} />
        <main className="min-w-0 flex-1">
          <div className="pt-4 lg:hidden">
            <MobileTabs activePage={activePage} pages={pages} onPageChange={setActivePage} />
          </div>
          <AnimatePresence mode="wait">
            <motion.div
              key={activePage}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.18, ease: 'easeOut' }}
              className="h-full"
            >
              {activePage === 'chat' ? (
                <ChatPage chat={chat} />
              ) : (
                <Dashboard stats={stats} onRefresh={stats.refreshStats} />
              )}
            </motion.div>
          </AnimatePresence>
        </main>
        <AnalyticsSidebar analytics={chat.analytics} isLoading={chat.isLoading} />
      </div>
      <ToastContainer toasts={toasts.toasts} onDismiss={toasts.dismissToast} />
    </div>
  );
}
