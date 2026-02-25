import { defineConfig } from 'vite'

export default defineConfig(async () => {
  const plugins = []
  try {
    const { default: react } = await import('@vitejs/plugin-react')
    plugins.push(react())
  } catch (err) {
    // Keep dev/build usable in offline environments where optional plugin install is unavailable.
  }

  return {
    plugins,
    server: {
      port: 5173,
      host: true
    }
  }
})
