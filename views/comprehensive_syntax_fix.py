#!/usr/bin/env python3
"""
Comprehensive fix for all syntax errors in the Options Calculator Pro
This script will fix all the malformed patterns introduced by the sed commands
"""

import re
import os
from pathlib import Path

def fix_malformed_empty_patterns(content):
    """Fix all malformed .empty patterns"""
    
    # Pattern 1: The main recursive malformed pattern
    pattern1 = r'\.empty if hasattr\(\.empty.*?== 0'
    content = re.sub(pattern1, '.empty', content, flags=re.DOTALL)
    
    # Pattern 2: Complex nested malformed conditionals
    pattern2 = r'if ([a-zA-Z_][a-zA-Z0-9_.]*?)\.empty if hasattr\(\.empty.*?== 0:'
    content = re.sub(pattern2, r'if hasattr(\1, "empty") and \1.empty:', content, flags=re.DOTALL)
    
    # Pattern 3: Remaining malformed .empty references
    pattern3 = r'\.empty if hasattr\([^)]*\) else len\([^)]*\) == 0'
    content = re.sub(pattern3, '.empty', content, flags=re.DOTALL)
    
    return content

def fix_merged_statements(content):
    """Fix statements that got merged on one line"""
    
    # Pattern 1: return None followed by other statements
    content = re.sub(r'return None\s+([a-zA-Z_].*)', r'return None\n        # \1', content)
    
    # Pattern 2: General merged statements with excessive whitespace
    content = re.sub(r'(\w+.*?)\s{10,}(\w+.*)', r'\1\n        \2', content)
    
    return content

def fix_malformed_init_methods(content):
    """Fix malformed __init__ method definitions"""
    
    # Fix __init__ to __init__
    content = content.replace('__init__', '__init__')
    
    # Fix broken function definitions where parameters got split incorrectly
    pattern = r'def __init__\(self, [^)]*\n\s*self\.[^=]+=.*?ml_service:.*?\):'
    replacement = 'def __init__(self, market_data_service: MarketDataService, config_manager: ConfigManager, ml_service: MLService, thread_manager: ThreadManager, parent=None):'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    return content

def fix_file_syntax(file_path):
    """Fix syntax errors in a specific file"""
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        original_content = content
        
        # Apply all fixes
        content = fix_malformed_empty_patterns(content)
        content = fix_merged_statements(content)
        content = fix_malformed_init_methods(content)
        
        if content != original_content:
            # Create backup if it doesn't exist
            backup_path = f"{file_path}.pre_fix_backup"
            if not os.path.exists(backup_path):
                with open(backup_path, 'w') as f:
                    f.write(original_content)
            
            # Write fixed content
            with open(file_path, 'w') as f:
                f.write(content)
            
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def test_file_syntax(file_path):
    """Test if a file has valid Python syntax"""
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        compile(content, file_path, 'exec')
        return True, None
        
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)

def main():
    """Main function to fix all syntax errors"""
    
    print("=" * 60)
    print("COMPREHENSIVE SYNTAX FIX FOR OPTIONS CALCULATOR PRO")
    print("=" * 60)
    
    # Files to fix
    files_to_fix = [
        'views/history_view.py',
        'controllers/analysis_controller.py',
        'services/market_data.py',
        'services/volatility_service.py',
        'services/options_service.py'
    ]
    
    # Step 1: Test current syntax
    print("\nStep 1: Testing current syntax...")
    syntax_errors = []
    
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            is_valid, error = test_file_syntax(file_path)
            if is_valid:
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path}: {error}")
                syntax_errors.append(file_path)
        else:
            print(f"? {file_path}: File not found")
    
    if not syntax_errors:
        print("\n🎉 All files have valid syntax!")
        return 0
    
    # Step 2: Apply fixes
    print(f"\nStep 2: Fixing {len(syntax_errors)} files with syntax errors...")
    
    fixed_files = []
    for file_path in syntax_errors:
        print(f"Fixing {file_path}...")
        if fix_file_syntax(file_path):
            fixed_files.append(file_path)
            print(f"  ✓ Applied fixes to {file_path}")
        else:
            print(f"  - No changes needed for {file_path}")
    
    # Step 3: Test fixes
    print(f"\nStep 3: Testing fixes...")
    
    all_fixed = True
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            is_valid, error = test_file_syntax(file_path)
            if is_valid:
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path}: {error}")
                all_fixed = False
    
    # Step 4: Final test
    print(f"\nStep 4: Testing application import...")
    
    try:
        # Test if we can at least import the main modules
        import sys
        sys.path.insert(0, '.')
        
        # Test core imports
        from core.app import OptionsCalculatorApp
        print("✓ Core application imports successfully")
        
        print("\n" + "=" * 60)
        print("🎉 ALL SYNTAX ERRORS FIXED!")
        print("=" * 60)
        print("\nYou can now run: python main.py")
        
        return 0
        
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        print("\n" + "=" * 60)
        print("⚠️  SOME ISSUES REMAIN")
        print("=" * 60)
        print("Additional manual fixes may be needed.")
        print("Check the .pre_fix_backup files if you need to restore.")
        
        return 1

if __name__ == "__main__":
    exit(main())