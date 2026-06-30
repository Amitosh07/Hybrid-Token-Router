/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        ink: {
          950: '#09090B',
          900: '#18181B',
          800: '#27272A',
          700: '#3F3F46',
          300: '#D4D4D8',
          400: '#A1A1AA',
          500: '#71717A'
        },
        accent: '#3B82F6',
        success: '#22C55E',
        warning: '#F59E0B',
        danger: '#EF4444'
      },
      boxShadow: {
        panel: '0 18px 60px rgba(0, 0, 0, 0.28)'
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'Segoe UI', 'Arial', 'sans-serif'],
        mono: ['JetBrains Mono', 'SFMono-Regular', 'Consolas', 'monospace']
      }
    }
  },
  plugins: []
};
