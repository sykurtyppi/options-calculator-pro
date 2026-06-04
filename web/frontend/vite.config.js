/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Vitest config for rendered-component tests. Scoped to *.dom.test.{js,jsx}
  // ONLY, so it never collects the node:test pure-function suites (which run
  // under `npm test` via `node --test 'src/**/*.test.js'`). The two runners are
  // deliberately separate: node:test for fast pure-logic/source guards,
  // Vitest+jsdom for DOM rendering. The `.test.js` and `.dom.test.jsx` globs do
  // not overlap, so every test file is owned by exactly one runner.
  test: {
    environment: 'jsdom',
    include: ['src/**/*.dom.test.{js,jsx}'],
    setupFiles: ['./src/test/setup.js'],
    globals: true,
    css: false,
  },
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
