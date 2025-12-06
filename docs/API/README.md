# API Documentation

Options Calculator Pro provides both internal APIs for plugin development and external APIs for integration.

## Overview

The application exposes several APIs:

- **Core Services API**: Access to market data and analysis services
- **Plugin API**: For developing custom plugins
- **Export API**: For extracting data and results
- **Configuration API**: For programmatic configuration

## Quick Start

### Basic Usage

```python
from options_calculator import OptionsCalculator

# Initialize calculator
calc = OptionsCalculator()

# Analyze a symbol
result = calc.analyze('AAPL', contracts=1, debit=2.50)

# Access results
print(f"Recommendation: {result.recommendation}")
print(f"Confidence: {result.confidence}")
print(f"Expected Profit: {result.expected_profit}")