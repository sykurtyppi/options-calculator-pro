import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

class RootErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, errorInfo) {
    console.error('Root render failure:', error, errorInfo)
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 24, color: '#f8fafc', fontFamily: 'monospace' }}>
          <h2>Frontend render error</h2>
          <p>{String(this.state.error?.message || this.state.error)}</p>
          <p>Open browser console for stack trace.</p>
        </div>
      )
    }
    return this.props.children
  }
}

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </React.StrictMode>
)
