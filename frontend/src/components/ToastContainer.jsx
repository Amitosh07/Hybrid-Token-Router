import { AnimatePresence, motion } from 'framer-motion';
import { FiAlertCircle, FiCheckCircle, FiInfo, FiX } from 'react-icons/fi';

const icons = {
  success: FiCheckCircle,
  error: FiAlertCircle,
  info: FiInfo
};

const tones = {
  success: 'text-success',
  error: 'text-danger',
  info: 'text-accent'
};

export default function ToastContainer({ toasts, onDismiss }) {
  return (
    <div className="fixed bottom-4 right-4 z-50 flex w-[min(360px,calc(100vw-32px))] flex-col gap-3">
      <AnimatePresence>
        {toasts.map((toast) => {
          const Icon = icons[toast.type] || icons.info;
          return (
            <motion.div
              key={toast.id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 12 }}
              className="flex items-start gap-3 rounded-xl border border-ink-800 bg-ink-900 p-4 shadow-panel"
            >
              <Icon className={`mt-0.5 shrink-0 ${tones[toast.type] || tones.info}`} />
              <p className="flex-1 text-sm text-zinc-100">{toast.message}</p>
              <button type="button" onClick={() => onDismiss(toast.id)} className="text-ink-500 hover:text-zinc-100">
                <FiX />
              </button>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
