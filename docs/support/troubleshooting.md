# Troubleshooting Guide

Comprehensive guide to resolving common issues with Options Calculator Pro.

## Quick Diagnostic Checklist

Before diving into specific issues, run through this quick checklist:

- [ ] Is your Python version 3.11 or higher?
- [ ] Are you connected to the internet?
- [ ] Are your API keys configured correctly?
- [ ] Is the application up to date?
- [ ] Have you restarted the application recently?
- [ ] Do you have sufficient disk space (>1GB free)?
- [ ] Is your system meeting minimum requirements?

## Installation Issues

### Python Version Problems

**Problem**: Error about Python version compatibility

**Solutions:**
```bash
# Check Python version
python --version

# Should show 3.11.0 or higher
# If not, install Python 3.11+ from python.org

# Create new virtual environment with correct Python
python3.11 -m venv options_calc_env
source options_calc_env/bin/activate  # Linux/Mac
# or
options_calc_env\Scripts\activate     # Windows

# Install in clean environment
pip install options-calculator-pro