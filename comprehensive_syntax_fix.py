#!/usr/bin/env python3
"""
Comprehensive fix for all syntax errors in the Options Calculator Pro
"""

import re
import os

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
    
    # Fix broken function definitions
    if 'def __init__(self, market_data_service: MarketDataService' in content and 'ml_service:' in content:
        # Find and fix broken init method
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'def __init__' in line and not line.strip().endswith('):'):
                # Fix broken function definition
                base_indent = len(line) - len(line.lstrip())
                lines[i] = ' ' * base_indent + 'def __init__(self, market_data_service: MarketDataService, config_manager: ConfigManager, ml_service: MLService, thread_manager: ThreadManager, parent=None):'
                # Remove continuation line if it exists
                if i + 1 < len(lines) and 'ml_service:' in lines[i + 1]:
                    lines[i + 1] = ''
                break
        content = '\n'.join(lines)
    
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
    print("\nTesting current syntax...")
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
        print("\nAll files have valid syntax!")
        return 0
    
    # Step 2: Apply fixes
    print(f"\nFixing {len(syntax_errors)} files with syntax errors...")
    
    for file_path in syntax_errors:
        print(f"Fixing {file_path}...")
        if fix_file_syntax(file_path):
            print(f"  ✓ Applied fixes")
        else:
            print(f"  - No changes needed")
    
    # Step 3: Test fixes
    print(f"\nTesting fixes...")
    
    all_fixed = True
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            is_valid, error = test_file_syntax(file_path)
            if is_valid:
                print(f"✓ {file_path}")
            else:
                print(f"✗ {file_path}: {error}")
                all_fixed = False
    
    if all_fixed:
        print("\n" + "=" * 60)
        print("ALL SYNTAX ERRORS FIXED!")
        print("=" * 60)
        print("\nYou can now run: python main.py")
        return 0
    else:
        print("\nSome issues remain - check output above")
        return 1

if __name__ == "__main__":
    exit(main())
