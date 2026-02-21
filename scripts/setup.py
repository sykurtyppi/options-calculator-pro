#!/usr/bin/env python3
"""
Setup script for Options Calculator Pro
Handles installation, configuration, and environment setup
"""

import os
import sys
import subprocess
import platform
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

class OptionsCalculatorSetup:
    """Main setup class for Options Calculator Pro"""
    
    def __init__(self):
        self.system = platform.system()
        self.python_version = sys.version_info
        self.project_root = Path(__file__).parent.parent
        self.venv_path = self.project_root / ".venv"
        self.config_dir = self.get_config_directory()
        
    def get_config_directory(self) -> Path:
        """Get platform-specific configuration directory"""
        if self.system == "Windows":
            config_dir = Path(os.environ.get("APPDATA", "")) / "Options Calculator Pro"
        elif self.system == "Darwin":  # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "Options Calculator Pro"
        else:  # Linux and others
            config_dir = Path.home() / ".options_calculator"
        
        return config_dir
    
    def check_requirements(self) -> bool:
        """Check system requirements"""
        print("🔍 Checking system requirements...")
        
        issues = []
        
        # Check Python version
        if self.python_version < (3, 11):
            issues.append(f"Python 3.11+ required, found {self.python_version.major}.{self.python_version.minor}")
        else:
            print(f"✅ Python {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}")
        
        # Check available memory
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            if memory_gb < 8:
                issues.append(f"8GB+ RAM recommended, found {memory_gb:.1f}GB")
            else:
                print(f"✅ Memory: {memory_gb:.1f}GB")
        except ImportError:
            print("⚠️  Cannot check memory (psutil not installed)")
        
        # Check disk space
        try:
            disk_space = shutil.disk_usage(self.project_root).free / (1024**3)
            if disk_space < 2:
                issues.append(f"2GB+ disk space required, found {disk_space:.1f}GB")
            else:
                print(f"✅ Disk space: {disk_space:.1f}GB available")
        except Exception as e:
            print(f"⚠️  Cannot check disk space: {e}")
        
        # Check internet connectivity
        try:
            import urllib.request
            urllib.request.urlopen('https://pypi.org', timeout=10)
            print("✅ Internet connectivity")
        except Exception:
            issues.append("Internet connection required for installation")
        
        if issues:
            print("\n❌ System requirement issues:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        
        print("✅ All system requirements met")
        return True
    
    def create_virtual_environment(self) -> bool:
        """Create Python virtual environment"""
        print("\n🐍 Setting up virtual environment...")
        
        if self.venv_path.exists():
            print("📁 Virtual environment already exists")
            response = input("Recreate it? (y/N): ").strip().lower()
            if response == 'y':
                print("🗑️  Removing existing virtual environment...")
                shutil.rmtree(self.venv_path)
            else:
                return True
        
        try:
            print("📦 Creating virtual environment...")
            subprocess.run([
                sys.executable, "-m", "venv", str(self.venv_path)
            ], check=True, capture_output=True, text=True)
            
            print("✅ Virtual environment created")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    def get_pip_command(self) -> List[str]:
        """Get pip command for current platform"""
        if self.system == "Windows":
            return [str(self.venv_path / "Scripts" / "python.exe"), "-m", "pip"]
        else:
            return [str(self.venv_path / "bin" / "python"), "-m", "pip"]
    
    def install_dependencies(self) -> bool:
        """Install Python dependencies"""
        print("\n📦 Installing dependencies...")
        
        pip_cmd = self.get_pip_command()
        
        # Upgrade pip first
        try:
            print("⬆️  Upgrading pip...")
            subprocess.run(
                pip_cmd + ["install", "--upgrade", "pip"],
                check=True, capture_output=True, text=True
            )
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to upgrade pip: {e}")
        
        # Install requirements
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            try:
                print("📋 Installing from requirements.txt...")
                subprocess.run(
                    pip_cmd + ["install", "-r", str(requirements_file)],
                    check=True, capture_output=True, text=True
                )
                print("✅ Dependencies installed")
                return True
                
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install dependencies: {e}")
                print(f"Error output: {e.stderr}")
                return False
        else:
            print("⚠️  requirements.txt not found, installing core dependencies...")
            core_deps = [
                "PySide6>=6.4.0",
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "scipy>=1.7.0",
                "matplotlib>=3.5.0",
                "scikit-learn>=1.0.0",
                "yfinance>=0.1.70",
                "requests>=2.25.0"
            ]
            
            for dep in core_deps:
                try:
                    print(f"📦 Installing {dep}...")
                    subprocess.run(
                        pip_cmd + ["install", dep],
                        check=True, capture_output=True, text=True
                    )
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {dep}: {e}")
                    return False
            
            print("✅ Core dependencies installed")
            return True
    
    def create_configuration(self) -> bool:
        """Create initial configuration"""
        print("\n⚙️  Setting up configuration...")
        
        # Create config directory
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        default_config = {
            "version": "1.0.0",
            "first_run": True,
            "theme": "Dark Professional",
            "api_keys": {
                "alpha_vantage": "",
                "finnhub": ""
            },
            "portfolio_value": 100000,
            "max_position_risk": 0.02,
            "cache_ttl_minutes": 30,
            "log_level": "INFO",
            "data_directory": str(self.config_dir / "data"),
            "cache_directory": str(self.config_dir / "cache"),
            "logs_directory": str(self.config_dir / "logs")
        }
        
        config_file = self.config_dir / "config.json"
        
        if config_file.exists():
            print("📁 Configuration file already exists")
            response = input("Overwrite with defaults? (y/N): ").strip().lower()
            if response != 'y':
                return True
        
        try:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            
            # Create subdirectories
            for directory in ["data", "cache", "logs", "exports", "backups"]:
                (self.config_dir / directory).mkdir(exist_ok=True)
            
            print("✅ Configuration created")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create configuration: {e}")
            return False
    
    def setup_desktop_integration(self) -> bool:
        """Setup desktop integration (shortcuts, file associations)"""
        print("\n🖥️  Setting up desktop integration...")
        
        try:
            if self.system == "Windows":
                return self.setup_windows_integration()
            elif self.system == "Darwin":
                return self.setup_macos_integration()
            else:
                return self.setup_linux_integration()
        except Exception as e:
            print(f"⚠️  Desktop integration failed: {e}")
            return True  # Non-critical failure
    
    def setup_windows_integration(self) -> bool:
        """Setup Windows-specific integration"""
        try:
            import winshell
            from win32com.client import Dispatch
            
            desktop = winshell.desktop()
            
            # Create desktop shortcut
            shortcut_path = os.path.join(desktop, "Options Calculator Pro.lnk")
            target = sys.executable
            arguments = f'"{self.project_root / "main.py"}"'
            
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = target
            shortcut.Arguments = arguments
            shortcut.WorkingDirectory = str(self.project_root)
            shortcut.IconLocation = str(self.project_root / "resources" / "icon.ico")
            shortcut.save()
            
            print("✅ Desktop shortcut created")
            return True
            
        except ImportError:
            print("⚠️  Windows integration requires pywin32 and winshell")
            return True
        except Exception as e:
            print(f"⚠️  Windows integration failed: {e}")
            return True
    
    def setup_macos_integration(self) -> bool:
        """Setup macOS-specific integration"""
        try:
            # Create .app bundle structure
            app_path = Path.home() / "Applications" / "Options Calculator Pro.app"
            contents_path = app_path / "Contents"
            macos_path = contents_path / "MacOS"
            resources_path = contents_path / "Resources"
            
            # Create directories
            macos_path.mkdir(parents=True, exist_ok=True)
            resources_path.mkdir(parents=True, exist_ok=True)
            
            # Create Info.plist
            info_plist = contents_path / "Info.plist"
            plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>options_calculator</string>
    <key>CFBundleIdentifier</key>
    <string>com.optionscalculatorpro.app</string>
    <key>CFBundleName</key>
    <string>Options Calculator Pro</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
</dict>
</plist>"""
            
            with open(info_plist, 'w') as f:
                f.write(plist_content)
            
            # Create launcher script
            launcher_script = macos_path / "options_calculator"
            launcher_content = f"""#!/bin/bash
cd "{self.project_root}"
"{self.venv_path / 'bin' / 'python'}" main.py
"""
            
            with open(launcher_script, 'w') as f:
                f.write(launcher_content)
            
            os.chmod(launcher_script, 0o755)
            
            print("✅ macOS app bundle created")
            return True
            
        except Exception as e:
            print(f"⚠️  macOS integration failed: {e}")
            return True
    
    def setup_linux_integration(self) -> bool:
        """Setup Linux-specific integration"""
        try:
            # Create .desktop file
            desktop_file_path = Path.home() / ".local" / "share" / "applications" / "options-calculator-pro.desktop"
            desktop_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Options Calculator Pro
Comment=Professional options trading analysis
Exec={self.venv_path / 'bin' / 'python'} {self.project_root / 'main.py'}
Icon={self.project_root / 'resources' / 'icon.png'}
Terminal=false
StartupNotify=true
Categories=Office;Finance;
"""
            
            with open(desktop_file_path, 'w') as f:
                f.write(desktop_content)
            
            os.chmod(desktop_file_path, 0o644)
            
            print("✅ Linux desktop integration created")
            return True
            
        except Exception as e:
            print(f"⚠️  Linux integration failed: {e}")
            return True
    
    def run_initial_setup(self) -> bool:
        """Run initial application setup"""
        print("\n🚀 Running initial application setup...")
        
        python_cmd = self.get_pip_command()[0]
        
        try:
            # Test import
            test_script = f"""
import sys
sys.path.insert(0, '{self.project_root}')
try:
    import options_calculator
    print('✅ Import successful')
except Exception as e:
    print(f'❌ Import failed: {{e}}')
    sys.exit(1)
"""
            
            result = subprocess.run(
                [python_cmd, "-c", test_script],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"❌ Application test failed: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Initial setup failed: {e}")
            return False
    
    def setup_api_keys(self) -> bool:
        """Interactive API key setup"""
        print("\n🔑 API Keys Setup")
        print("API keys are optional but recommended for better data quality.")
        print("You can skip this step and configure them later in the application.")
        
        response = input("\nWould you like to configure API keys now? (y/N): ").strip().lower()
        if response != 'y':
            return True
        
        config_file = self.config_dir / "config.json"
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except:
            print("❌ Could not load configuration file")
            return False
        
        # Alpha Vantage setup
        print("\n📊 Alpha Vantage (Primary data source)")
        print("Get free API key at: https://www.alphavantage.co/support/#api-key")
        av_key = input("Enter Alpha Vantage API key (or press Enter to skip): ").strip()
        
        if av_key:
            config["api_keys"]["alpha_vantage"] = av_key
            print("✅ Alpha Vantage API key saved")
        
        # Finnhub setup
        print("\n📈 Finnhub (Backup data source)")
        print("Get free API key at: https://finnhub.io/register")
        finnhub_key = input("Enter Finnhub API key (or press Enter to skip): ").strip()
        
        if finnhub_key:
            config["api_keys"]["finnhub"] = finnhub_key
            print("✅ Finnhub API key saved")
        
        # Save updated configuration
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            print("✅ Configuration updated")
            return True
        except Exception as e:
            print(f"❌ Failed to save configuration: {e}")
            return False
    
    def create_launcher_scripts(self) -> bool:
        """Create convenient launcher scripts"""
        print("\n📜 Creating launcher scripts...")
        
        try:
            # Unix launcher script
            if self.system != "Windows":
                launcher_script = self.project_root / "run_options_calculator.sh"
                script_content = f"""#!/bin/bash
# Options Calculator Pro Launcher
cd "{self.project_root}"
source "{self.venv_path}/bin/activate"
python main.py "$@"
"""
                with open(launcher_script, 'w') as f:
                    f.write(script_content)
                os.chmod(launcher_script, 0o755)
                print("✅ Unix launcher script created")
            
            # Windows batch script
            if self.system == "Windows":
                launcher_script = self.project_root / "run_options_calculator.bat"
                script_content = f"""@echo off
rem Options Calculator Pro Launcher
cd /d "{self.project_root}"
"{self.venv_path}\\Scripts\\python.exe" main.py %*
"""
                with open(launcher_script, 'w') as f:
                    f.write(script_content)
                print("✅ Windows launcher script created")
            
            return True
            
        except Exception as e:
            print(f"⚠️  Failed to create launcher scripts: {e}")
            return True  # Non-critical
    
    def run_complete_setup(self) -> bool:
        """Run complete setup process"""
        print("🎯 Options Calculator Pro Setup")
        print("=" * 40)
        
        steps = [
            ("Check Requirements", self.check_requirements),
            ("Create Virtual Environment", self.create_virtual_environment),
            ("Install Dependencies", self.install_dependencies),
            ("Create Configuration", self.create_configuration),
            ("Setup Desktop Integration", self.setup_desktop_integration),
            ("Create Launcher Scripts", self.create_launcher_scripts),
            ("Run Initial Setup", self.run_initial_setup),
            ("Setup API Keys", self.setup_api_keys),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            if not step_func():
                print(f"❌ Setup failed at: {step_name}")
                return False
        
        print("\n" + "=" * 60)
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the application:")
        
        if self.system == "Windows":
            print(f"   Double-click: {self.project_root / 'run_options_calculator.bat'}")
        else:
            print(f"   Execute: {self.project_root / 'run_options_calculator.sh'}")
        
        print("2. Configure API keys (if not done during setup):")
        print("   Settings → API Settings")
        print("3. Read the documentation:")
        print("   docs/user-guide/quick-start.md")
        
        return True

def main():
    """Main setup entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        setup = OptionsCalculatorSetup()
        
        if command == "check":
            setup.check_requirements()
        elif command == "venv":
            setup.create_virtual_environment()
        elif command == "deps":
            setup.install_dependencies()
        elif command == "config":
            setup.create_configuration()
        elif command == "desktop":
            setup.setup_desktop_integration()
        elif command == "api":
            setup.setup_api_keys()
        elif command == "test":
            setup.run_initial_setup()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: check, venv, deps, config, desktop, api, test")
    else:
        # Run complete setup
        setup = OptionsCalculatorSetup()
        setup.run_complete_setup()

if __name__ == "__main__":
    main()
