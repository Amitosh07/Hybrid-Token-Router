import { motion } from 'framer-motion';

export default function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div className="rounded-xl border border-ink-800 bg-ink-900 p-5">
        <div className="flex items-center gap-2">
          {[0, 1, 2].map((dot) => (
            <motion.span
              key={dot}
              className="h-2 w-2 rounded-full bg-ink-400"
              animate={{ opacity: [0.35, 1, 0.35], y: [0, -3, 0] }}
              transition={{ duration: 1, repeat: Infinity, delay: dot * 0.16 }}
            />
          ))}
          <span className="ml-2 text-sm text-ink-400">Routing prompt</span>
        </div>
      </div>
    </div>
  );
}
