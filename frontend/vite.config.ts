import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    allowedHosts: ['.e2b.app'],
    proxy: Object.fromEntries(['/api', '/outputs', '/dataset'].map(path => [path, 'http://localhost:8000'])),
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          // The optional text-prompt detector uses its own WebGPU runtime.
          if (id.includes('onnxruntime-web') && id.includes('webgpu')) return 'prompt-runtime'
        },
      },
    },
  },
})
