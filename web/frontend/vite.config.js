import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          if (id.includes('/react/') || id.includes('/react-dom/') || id.includes('/scheduler/')) return 'vendor-react'
          if (
            id.includes('/recharts/')
            || id.includes('/d3-')
            || id.includes('/victory-vendor/')
            || id.includes('/decimal.js-light/')
          ) return 'vendor-recharts'
          return 'vendor'
        },
      },
    },
  },
  server: {
    port: 5173,
    host: '127.0.0.1'
  }
})
