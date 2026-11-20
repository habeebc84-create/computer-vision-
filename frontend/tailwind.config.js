/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        ink: '#0c1019', panel: '#111622', card: '#151b28',
        accent: '#a5b4fc', cyan2: '#22d3ee', violet2: '#8b5cf6',
        muted: '#8b98b8'
      },
      fontFamily: { sans: ['Inter', 'system-ui', 'Segoe UI', 'sans-serif'] }
    }
  },
  plugins: []
}
