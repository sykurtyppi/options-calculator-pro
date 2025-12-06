#!/usr/bin/env python3
"""
Quick fix for the malformed .empty syntax error in history_view.py
"""

import re
import os
from pathlib import Path

def fix_malformed_empty_checks():
    """Fix the malformed .empty conditional expressions"""
    
    history_view_path = Path("views/history_view.py")
    
    if not history_view_path.exists():
        print(f"Error: {history_view_path} not found")
        return False
    
    print(f"Fixing malformed syntax in {history_view_path}")
    
    # Create backup
    backup_path = history_view_path.with_suffix('.py.backup')
    if not backup_path.exists():
        print(f"Creating backup at {backup_path}")
        import shutil
        shutil.copy(history_view_path, backup_path)
    
    try:
        # Read the file
        with open(history_view_path, 'r') as f:
            content = f.read()
        
        # Fix the specific malformed pattern
        # This pattern matches the recursive/malformed conditionals created by the sed command
        malformed_patterns = [
            # Pattern 1: Complex nested conditionals
            r'\.empty if hasattr\(\.empty.*?len\(\.empty.*?\) == 0',
            # Pattern 2: Simple malformed pattern  
            r'\.empty if hasattr\(\.empty, "empty"\) else len\(\.empty\) == 0',
            # Pattern 3: Any remaining malformed .empty patterns
            r'if hasattr\(\.empty.*?\)',
        ]
        
        original_content = content
        
        for pattern in malformed_patterns:
            content = re.sub(pattern, 'df.empty', content, flags=re.DOTALL)
        
        # Also fix specific line that's causing the syntax error
        # Replace the problematic line structure
        problematic_line_pattern = r'if self\.current_data\.empty if hasattr.*?== 0:'
        content = re.sub(problematic_line_pattern, 'if hasattr(self.current_data, "empty") and self.current_data.empty:', content)
        
        # Fix any remaining syntax errors with .empty
        content = re.sub(r'if\s+([a-zA-Z_][a-zA-Z0-9_.]*?)\.empty if hasattr.*?:', 
                        r'if hasattr(\1, "empty") and \1.empty:', content)
        
        # Write the fixed content back
        if content != original_content:
            with open(history_view_path, 'w') as f:
                f.write(content)
            print("✓ Fixed malformed syntax")
            return True
        else:
            print("No changes needed")
            return True
            
    except Exception as e:
        print(f"Error fixing file: {e}")
        # Restore from backup if it exists
        if backup_path.exists():
            print("Restoring from backup...")
            import shutil
            shutil.copy(backup_path, history_view_path)
        return False

def implement_proper_empty_checks():
    """Implement proper DataFrame empty checks throughout the codebase"""
    
    print("\nImplementing proper .empty checks...")
    
    # Define the safe empty check function
    safe_check_template = """
def safe_dataframe_check(df):
    \"\"\"Safely check if DataFrame is empty\"\"\"
    if hasattr(df, 'empty'):
        return df.empty
    elif hasattr(df, '__len__'):
        return len(df) == 0
    else:
        return True  # Assume empty if can't determine
"""
    
    # Files to check and fix
    python_files = []
    for directory in ['views', 'controllers', 'services', 'models']:
        if os.path.exists(directory):
            for file_path in Path(directory).glob('*.py'):
                python_files.append(file_path)
    
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Check if file uses .empty
            if '.empty' in content:
                print(f"Processing {file_path}")
                
                # Replace simple .empty checks with safe checks
                # Pattern: variable.empty -> safe check
                def replace_empty_check(match):
                    full_expr = match.group(0)
                    var_name = full_expr.replace('.empty', '')
                    return f"(hasattr({var_name}, 'empty') and {var_name}.empty)"
                
                # Apply the replacement
                pattern = r'[a-zA-Z_][a-zA-Z0-9_.]*\.empty(?!\s*if)'  # Don't match if followed by 'if'
                new_content = re.sub(pattern, replace_empty_check, content)
                
                if new_content != content:
                    # Create backup
                    backup_path = file_path.with_suffix('.py.backup')
                    if not backup_path.exists():
                        import shutil
                        shutil.copy(file_path, backup_path)
                    
                    # Write updated content
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    print(f"✓ Updated {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

def main():
    print("=" * 60)
    print("FIXING OPTIONS CALCULATOR PRO - EMPTY CHECK SYNTAX ERROR")
    print("=" * 60)
    
    # Step 1: Fix the immediate syntax error
    print("Step 1: Fixing immediate syntax error...")
    if not fix_malformed_empty_checks():
        print("Failed to fix syntax error")
        return 1
    
    # Step 2: Implement proper empty checks
    print("\nStep 2: Implementing proper empty checks...")
    implement_proper_empty_checks()
    
    print("\n" + "=" * 60)
    print("FIX COMPLETE!")
    print("=" * 60)
    print("\nYou can now test the application with:")
    print("python main.py")
    print("\nIf you still encounter issues, check the .backup files for reference")
    
    return 0

if __name__ == "__main__":
    exit(main())