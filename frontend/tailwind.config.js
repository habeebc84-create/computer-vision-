/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#050914', panel: '#0a1128', card: '#0e1730',
        accent: '#0ea5e9', cyan2: '#22d3ee', violet2: '#8b5cf6',
        muted: '#8b98b8'
      },
      fontFamily: { sans: ['Inter', 'system-ui', 'Segoe UI', 'sans-serif'] }
    }
  },
  plugins: []
}
