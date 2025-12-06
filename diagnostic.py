#!/usr/bin/env python3
"""
Diagnostic Script - Check Options Calculator Pro Status
"""

import os
import sys
from pathlib import Path

def check_file_status(filepath, min_lines=10):
    """Check if file exists and has content"""
    try:
        if not Path(filepath).exists():
            return "❌ MISSING"
        
        with open(filepath, 'r') as f:
            lines = len(f.readlines())
        
        if lines < min_lines:
            return f"⚠️  EXISTS ({lines} lines - may need content)"
        else:
            return f"✅ OK ({lines} lines)"
            
    except Exception as e:
        return f"❌ ERROR: {e}"

def check_dependencies():
    """Check if required dependencies are installed"""
    required = [
        'PySide6', 'yfinance', 'pandas', 'numpy', 
        'scipy', 'matplotlib', 'scikit-learn'
    ]
    
    results = {}
    for module in required:
        try:
            __import__(module)
            results[module] = "✅ INSTALLED"
        except ImportError:
            results[module] = "❌ MISSING"
    
    return results

def main():
    print("="*60)
    print("OPTIONS CALCULATOR PRO - DIAGNOSTIC")
    print("="*60)
    
    # Check critical files
    critical_files = {
        'main.py': 'Main application entry point',
        'core/app.py': 'Core application class',
        'utils/greeks_calculator.py': 'Greeks calculation engine',
        'services/market_data.py': 'Market data service',
        'services/options_service.py': 'Options analysis service',
        'services/volatility_service.py': 'Volatility analysis service',
        'services/ML_service.py': 'Machine learning service',
        'utils/config_manager.py': 'Configuration management',
        'utils/logger.py': 'Logging system'
    }
    
    print("\n📁 CRITICAL FILES STATUS:")
    print("-" * 60)
    for filepath, description in critical_files.items():
        status = check_file_status(filepath)
        print(f"{filepath:<35} {status}")
        print(f"   {description}")
        print()
    
    # Check dependencies
    print("📦 DEPENDENCIES STATUS:")
    print("-" * 60)
    deps = check_dependencies()
    for module, status in deps.items():
        print(f"{module:<20} {status}")
    
    # Check virtual environment
    print(f"\n🐍 VIRTUAL ENVIRONMENT:")
    print("-" * 60)
    if Path('.venv').exists():
        print("✅ Virtual environment exists (.venv)")
    else:
        print("❌ Virtual environment not found")
        print("   Run: python -m venv .venv")
    
    # Summary and recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("-" * 60)
    
    missing_deps = [k for k, v in deps.items() if "MISSING" in v]
    if missing_deps:
        print(f"1. Install missing dependencies:")
        print(f"   pip install {' '.join(missing_deps)}")
    
    empty_files = []
    for filepath in critical_files.keys():
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                if len(f.read().strip()) < 50:  # Less than 50 characters
                    empty_files.append(filepath)
    
    if empty_files:
        print(f"\n2. Files that need implementation:")
        for filepath in empty_files:
            print(f"   - {filepath}")
    
    print(f"\n3. Next steps:")
    print(f"   - Run setup script: ./setup.sh")
    print(f"   - Copy provided code into empty files")
    print(f"   - Test with: python main.py")
    
    print("="*60)

if __name__ == "__main__":
    main()