#!/usr/bin/env python3
"""
Build script for Options Calculator Pro
Creates distributable packages for different platforms
"""

import os
import sys
import subprocess
import shutil
import platform
import json
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class BuildManager:
    """Manages the build process for Options Calculator Pro"""
    
    def __init__(self):
        self.system = platform.system()
        self.arch = platform.machine()
        self.project_root = Path(__file__).parent.parent
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"
        self.version = self.get_version()
        
        # Build configuration
        self.build_config = {
            "app_name": "Options Calculator Pro",
            "bundle_id": "com.optionscalculatorpro.app",
            "version": self.version,
            "build_date": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    
    def get_version(self) -> str:
        """Get application version"""
        try:
            init_file = self.project_root / "src" / "options_calculator" / "__init__.py"
            if init_file.exists():
                with open(init_file, 'r') as f:
                    for line in f:
                        if line.startswith('__version__'):
                            return line.split('=')[1].strip().strip('"\'')
            return "1.0.0"
        except Exception:
            return "1.0.0"
    
    def clean_build_directories(self):
        """Clean build and dist directories"""
        print("🧹 Cleaning build directories...")
        
        for directory in [self.build_dir, self.dist_dir]:
            if directory.exists():
                shutil.rmtree(directory)
                print(f"   Removed {directory}")
            directory.mkdir(parents=True, exist_ok=True)
            print(f"   Created {directory}")
    
    def install_build_dependencies(self):
        """Install build-specific dependencies"""
        print("📦 Installing build dependencies...")
        
        build_deps = [
            "pyinstaller>=5.0.0",
            "wheel>=0.37.0",
            "setuptools>=60.0.0",
            "twine>=4.0.0"
        ]
        
        if self.system == "Windows":
            build_deps.extend([
                "pywin32>=304",
                "winshell>=0.6"
            ])
        elif self.system == "Darwin":
            build_deps.extend([
                "py2app>=0.28",
                "dmgbuild>=1.6.0"
            ])
        
        for dep in build_deps:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)
                print(f"   ✅ {dep}")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Failed to install {dep}: {e}")
                return False
        
        return True
    
    def create_pyinstaller_spec(self) -> Path:
        """Create PyInstaller spec file"""
        print("📄 Creating PyInstaller spec file...")
        
        spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['{self.project_root}'],
    binaries=[],
    datas=[
        ('resources', 'resources'),
        ('docs', 'docs'),
        ('configs', 'configs'),
    ],
    hiddenimports=[
        'PySide6.QtCore',
        'PySide6.QtWidgets',
        'PySide6.QtGui',
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'sklearn',
        'yfinance',
        'requests',
        'packaging.version',
        'packaging.specifiers',
        'packaging.requirements',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'test',
        'tests',
        'unittest',
        'pdb',
        'doctest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='OptionsCalculatorPro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icon.{"ico" if self.system == "Windows" else "icns"}',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='OptionsCalculatorPro',
)

{'app = BUNDLE(' if self.system == 'Darwin' else ''}
{'    coll,' if self.system == 'Darwin' else ''}
{'    name="Options Calculator Pro.app",' if self.system == 'Darwin' else ''}
{'    icon="resources/icon.icns",' if self.system == 'Darwin' else ''}
{'    bundle_identifier="com.optionscalculatorpro.app",' if self.system == 'Darwin' else ''}
{'    info_plist={' if self.system == 'Darwin' else ''}
{'        "CFBundleShortVersionString": "' + self.version + '",' if self.system == 'Darwin' else ''}
{'        "CFBundleVersion": "' + self.version + '",' if self.system == 'Darwin' else ''}
{'        "NSHighResolutionCapable": True,' if self.system == 'Darwin' else ''}
{'    },' if self.system == 'Darwin' else ''}
{')' if self.system == 'Darwin' else ''}
'''
        
        spec_file = self.project_root / "options_calculator.spec"
        with open(spec_file, 'w') as f:
            f.write(spec_content)
        
        print(f"   ✅ Created {spec_file}")
        return spec_file
    
    def build_with_pyinstaller(self) -> bool:
        """Build executable with PyInstaller"""
        print("🔨 Building executable with PyInstaller...")
        
        spec_file = self.create_pyinstaller_spec()
        
        try:
            # Run PyInstaller
            cmd = [
                sys.executable, "-m", "PyInstaller",
                "--clean",
                "--noconfirm",
                str(spec_file)
            ]
            
            print(f"   Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   ✅ PyInstaller build successful")
                return True
            else:
                print(f"   ❌ PyInstaller build failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ PyInstaller build error: {e}")
            return False
    
    def create_windows_installer(self) -> bool:
        """Create Windows installer with NSIS"""
        if self.system != "Windows":
            return True
        
        print("📦 Creating Windows installer...")
        
        # Check if NSIS is available
        try:
            subprocess.run(["makensis", "/VERSION"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("   ⚠️  NSIS not found, skipping installer creation")
            print("   Install NSIS from: https://nsis.sourceforge.io/")
            return True
        
        # Create NSIS script
        nsis_script = self.build_dir / "installer.nsi"
        
        nsis_content = f'''
!define APPNAME "Options Calculator Pro"
!define COMPANYNAME "Options Calculator Pro"
!define DESCRIPTION "Professional Options Trading Analysis"
!define VERSIONMAJOR 1
!define VERSIONMINOR 0
!define VERSIONBUILD 0
!define HELPURL "https://github.com/username/options-calculator-pro"
!define UPDATEURL "https://github.com/username/options-calculator-pro/releases"
!define ABOUTURL "https://optionscalculatorpro.com"
!define INSTALLSIZE 150000  # Size in KB

RequestExecutionLevel admin
InstallDir "$PROGRAMFILES\\${{COMPANYNAME}}\\${{APPNAME}}"
Name "${{APPNAME}}"
Icon "..\\resources\\icon.ico"
outFile "..\\dist\\OptionsCalculatorPro-{self.version}-Windows-Installer.exe"

page directory
page instfiles

!macro VerifyUserIsAdmin
UserInfo::GetAccountType
pop $0
${{If}} $0 != "admin"
    messageBox mb_iconstop "Administrator rights required!"
    setErrorLevel 740
    quit
${{EndIf}}
!macroend

function .onInit
    setShellVarContext all
    !insertmacro VerifyUserIsAdmin
functionEnd

section "install"
    setOutPath $INSTDIR
    file /r "..\\dist\\OptionsCalculatorPro\\*"
    
    writeUninstaller "$INSTDIR\\uninstall.exe"
    
    createDirectory "$SMPROGRAMS\\${{COMPANYNAME}}"
    createShortCut "$SMPROGRAMS\\${{COMPANYNAME}}\\${{APPNAME}}.lnk" "$INSTDIR\\OptionsCalculatorPro.exe" "" "$INSTDIR\\OptionsCalculatorPro.exe"
    createShortCut "$DESKTOP\\${{APPNAME}}.lnk" "$INSTDIR\\OptionsCalculatorPro.exe" "" "$INSTDIR\\OptionsCalculatorPro.exe"
    
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "DisplayName" "${{APPNAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "UninstallString" "$\\"$INSTDIR\\uninstall.exe$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "QuietUninstallString" "$\\"$INSTDIR\\uninstall.exe$\\" /S"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "InstallLocation" "$\\"$INSTDIR$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "DisplayIcon" "$\\"$INSTDIR\\OptionsCalculatorPro.exe$\\""
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "Publisher" "${{COMPANYNAME}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "HelpLink" "${{HELPURL}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "URLUpdateInfo" "${{UPDATEURL}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "URLInfoAbout" "${{ABOUTURL}}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "DisplayVersion" "{self.version}"
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "VersionMajor" ${{VERSIONMAJOR}}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "VersionMinor" ${{VERSIONMINOR}}
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "NoModify" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "NoRepair" 1
    WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}" "EstimatedSize" ${{INSTALLSIZE}}
sectionEnd

section "uninstall"
    delete "$INSTDIR\\uninstall.exe"
    rmDir /r "$INSTDIR"
   
    delete "$SMPROGRAMS\\${{COMPANYNAME}}\\${{APPNAME}}.lnk"
    rmDir "$SMPROGRAMS\\${{COMPANYNAME}}"
    delete "$DESKTOP\\${{APPNAME}}.lnk"
   
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${{COMPANYNAME}} ${{APPNAME}}"
sectionEnd
'''
       
        with open(nsis_script, 'w') as f:
            f.write(nsis_content)
       
        try:
            subprocess.run([
                "makensis", str(nsis_script)
            ], check=True, capture_output=True, text=True)
           
            print("   ✅ Windows installer created")
            return True
           
        except subprocess.CalledProcessError as e:
            print(f"   ❌ NSIS installer creation failed: {e}")
            return False
   
    def create_macos_dmg(self) -> bool:
        """Create macOS DMG installer"""
        if self.system != "Darwin":
            return True
       
        print("📦 Creating macOS DMG...")
       
        try:
            import dmgbuild
           
            app_path = self.dist_dir / "Options Calculator Pro.app"
            dmg_path = self.dist_dir / f"OptionsCalculatorPro-{self.version}-macOS.dmg"
           
            # DMG build settings
            settings = {
                'filename': str(dmg_path),
                'volume_name': 'Options Calculator Pro',
                'format': 'UDZO',
                'compression_level': 9,
                'size': '200M',
                'files': [str(app_path)],
                'symlinks': {'Applications': '/Applications'},
                'icon_locations': {
                    'Options Calculator Pro.app': (100, 100),
                    'Applications': (400, 100),
                },
                'background': str(self.project_root / 'resources' / 'dmg_background.png'),
                'show_status_bar': False,
                'show_tab_view': False,
                'show_toolbar': False,
                'show_pathbar': False,
                'show_sidebar': False,
                'sidebar_width': 180,
                'window_rect': ((100, 100), (600, 400)),
                'default_view': 'icon-view',
                'show_icon_preview': False,
                'include_icon_view_settings': 'auto',
                'include_list_view_settings': 'auto',
                'arrange_by': None,
                'grid_offset': (0, 0),
                'grid_spacing': 100,
                'scroll_position': (0, 0),
                'label_pos': 'bottom',
                'text_size': 16,
                'icon_size': 128,
            }
           
            dmgbuild.build_dmg(str(dmg_path), 'Options Calculator Pro', settings)
            print("   ✅ macOS DMG created")
            return True
           
        except ImportError:
            print("   ⚠️  dmgbuild not available, creating simple DMG")
            return self.create_simple_macos_dmg()
        except Exception as e:
            print(f"   ❌ DMG creation failed: {e}")
            return False
   
    def create_simple_macos_dmg(self) -> bool:
        """Create simple macOS DMG without dmgbuild"""
        try:
            app_path = self.dist_dir / "Options Calculator Pro.app"
            dmg_path = self.dist_dir / f"OptionsCalculatorPro-{self.version}-macOS.dmg"
           
            # Create temporary directory
            temp_dir = self.build_dir / "dmg_temp"
            temp_dir.mkdir(exist_ok=True)
           
            # Copy app to temp directory
            shutil.copytree(app_path, temp_dir / "Options Calculator Pro.app")
           
            # Create Applications symlink
            os.symlink("/Applications", temp_dir / "Applications")
           
            # Create DMG
            subprocess.run([
                "hdiutil", "create",
                "-volname", "Options Calculator Pro",
                "-srcfolder", str(temp_dir),
                "-ov", "-format", "UDZO",
                str(dmg_path)
            ], check=True)
           
            # Cleanup
            shutil.rmtree(temp_dir)
           
            print("   ✅ Simple macOS DMG created")
            return True
           
        except Exception as e:
            print(f"   ❌ Simple DMG creation failed: {e}")
            return False
   
    def create_linux_package(self) -> bool:
        """Create Linux package (AppImage or deb)"""
        if self.system != "Linux":
            return True
       
        print("📦 Creating Linux package...")
       
        # Create tar.gz archive
        try:
            archive_name = f"OptionsCalculatorPro-{self.version}-Linux-{self.arch}.tar.gz"
            archive_path = self.dist_dir / archive_name
           
            # Create archive
            with zipfile.ZipFile(archive_path.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as zipf:
                app_dir = self.dist_dir / "OptionsCalculatorPro"
                for file_path in app_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(app_dir.parent)
                        zipf.write(file_path, arcname)
            
            # Also create tar.gz
            subprocess.run([
                "tar", "-czf", str(archive_path),
                "-C", str(self.dist_dir),
                "OptionsCalculatorPro"
            ], check=True)
           
            print("   ✅ Linux package created")
            return True
           
        except Exception as e:
            print(f"   ❌ Linux package creation failed: {e}")
            return False
   
    def create_source_distribution(self) -> bool:
        """Create Python source distribution"""
        print("📦 Creating source distribution...")
       
        try:
            # Clean previous builds
            for pattern in ["build", "*.egg-info", "__pycache__"]:
                for path in self.project_root.glob(pattern):
                    if path.is_dir():
                        shutil.rmtree(path)
           
            # Create source distribution
            subprocess.run([
                sys.executable, "setup.py", "sdist"
            ], cwd=self.project_root, check=True, capture_output=True)
           
            # Create wheel
            subprocess.run([
                sys.executable, "setup.py", "bdist_wheel"
            ], cwd=self.project_root, check=True, capture_output=True)
           
            print("   ✅ Source distribution created")
            return True
           
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Source distribution failed: {e}")
            return False
   
    def sign_packages(self) -> bool:
        """Code sign packages (platform-specific)"""
        print("🔐 Signing packages...")
       
        if self.system == "Darwin":
            return self.sign_macos_packages()
        elif self.system == "Windows":
            return self.sign_windows_packages()
        else:
            print("   ⚠️  Code signing not implemented for Linux")
            return True
   
    def sign_macos_packages(self) -> bool:
        """Sign macOS packages"""
        try:
            # Check for signing certificate
            result = subprocess.run([
                "security", "find-identity", "-v", "-p", "codesigning"
            ], capture_output=True, text=True)
            
            if "Developer ID Application" not in result.stdout:
                print("   ⚠️  No macOS signing certificate found")
                return True
           
            app_path = self.dist_dir / "Options Calculator Pro.app"
           
            # Sign the app
            subprocess.run([
                "codesign", "--force", "--verify", "--verbose",
                "--sign", "Developer ID Application",
                str(app_path)
            ], check=True)
            
            print("   ✅ macOS app signed")
            return True
           
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️  macOS signing failed: {e}")
            return True  # Non-critical failure
   
    def sign_windows_packages(self) -> bool:
        """Sign Windows packages"""
        try:
            # This would require a code signing certificate
            # Implementation depends on specific certificate setup
            print("   ⚠️  Windows code signing not configured")
            return True
           
        except Exception as e:
            print(f"   ⚠️  Windows signing failed: {e}")
            return True  # Non-critical failure
   
    def create_checksums(self) -> bool:
        """Create checksums for all packages"""
        print("🔐 Creating checksums...")
       
        checksums_file = self.dist_dir / "checksums.txt"
       
        try:
            with open(checksums_file, 'w') as f:
                f.write(f"# Options Calculator Pro {self.version} - Package Checksums\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
               
                for file_path in self.dist_dir.glob("*"):
                    if file_path.is_file() and file_path.name != "checksums.txt":
                        # Calculate SHA256
                        import hashlib
                        sha256_hash = hashlib.sha256()
                       
                        with open(file_path, "rb") as fp:
                            for chunk in iter(lambda: fp.read(4096), b""):
                                sha256_hash.update(chunk)
                       
                        checksum = sha256_hash.hexdigest()
                        f.write(f"{checksum}  {file_path.name}\n")
           
            print("   ✅ Checksums created")
            return True
           
        except Exception as e:
            print(f"   ❌ Checksum creation failed: {e}")
            return False
   
    def generate_build_report(self) -> bool:
        """Generate build report"""
        print("📊 Generating build report...")
       
        report = {
            "build_info": self.build_config,
            "system_info": {
                "platform": platform.platform(),
                "system": self.system,
                "architecture": self.arch,
                "python_version": sys.version,
            },
            "build_artifacts": [],
            "build_success": True,
            "build_timestamp": datetime.now().isoformat()
        }
       
        # List build artifacts
        for file_path in self.dist_dir.glob("*"):
            if file_path.is_file():
                report["build_artifacts"].append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "path": str(file_path.relative_to(self.project_root))
                })
       
        # Save report
        report_file = self.dist_dir / "build_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4)
       
        print("   ✅ Build report generated")
        return True
   
    def run_build_process(self, target: str = "all") -> bool:
        """Run the complete build process"""
        print("🔨 Starting build process...")
        print(f"Target: {target}")
        print(f"Platform: {self.system} ({self.arch})")
        print(f"Version: {self.version}")
        print("=" * 50)
       
        steps = [
            ("Clean Directories", self.clean_build_directories),
            ("Install Dependencies", self.install_build_dependencies),
            ("Build Executable", self.build_with_pyinstaller),
        ]
       
        # Platform-specific steps
        if target in ["all", "installer"]:
            if self.system == "Windows":
                steps.append(("Create Windows Installer", self.create_windows_installer))
            elif self.system == "Darwin":
                steps.append(("Create macOS DMG", self.create_macos_dmg))
            elif self.system == "Linux":
                steps.append(("Create Linux Package", self.create_linux_package))
       
        if target in ["all", "source"]:
            steps.append(("Create Source Distribution", self.create_source_distribution))
       
        if target in ["all", "sign"]:
            steps.append(("Sign Packages", self.sign_packages))
       
        steps.extend([
            ("Create Checksums", self.create_checksums),
            ("Generate Report", self.generate_build_report),
        ])
       
        # Execute steps
        for step_name, step_func in steps:
            print(f"\n{'='*10} {step_name} {'='*10}")
            if not step_func():
                print(f"❌ Build failed at: {step_name}")
                return False
       
        print("\n" + "=" * 50)
        print("🎉 Build completed successfully!")
        print(f"\nBuild artifacts in: {self.dist_dir}")
       
        # List created files
        artifacts = list(self.dist_dir.glob("*"))
        if artifacts:
            print("\nCreated files:")
            for artifact in artifacts:
                size = artifact.stat().st_size / (1024*1024)  # MB
                print(f"  📦 {artifact.name} ({size:.1f}MB)")
       
        return True

def main():
    """Main build entry point"""
    import argparse
   
    parser = argparse.ArgumentParser(description="Build Options Calculator Pro")
    parser.add_argument(
        "target", 
        choices=["all", "executable", "installer", "source", "sign", "clean"],
        default="all",
        nargs="?",
        help="Build target"
    )
    parser.add_argument(
        "--version",
        help="Override version number"
    )
   
    args = parser.parse_args()
   
    builder = BuildManager()
   
    if args.version:
        builder.version = args.version
        builder.build_config["version"] = args.version
   
    if args.target == "clean":
        builder.clean_build_directories()
        print("✅ Build directories cleaned")
        return
   
    success = builder.run_build_process(args.target)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
   main()