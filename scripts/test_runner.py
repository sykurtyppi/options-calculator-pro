#!/usr/bin/env python3
"""
Comprehensive test runner for Options Calculator Pro
Runs different test suites with reporting and coverage
"""

import os
import sys
import subprocess
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

class TestRunner:
    """Comprehensive test runner and reporter"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test_reports"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test configuration
        self.test_config = {
            "unit": {
                "path": "tests/unit",
                "markers": "",
                "timeout": 300,  # 5 minutes
                "parallel": True
            },
            "integration": {
                "path": "tests/integration", 
                "markers": "integration",
                "timeout": 600,  # 10 minutes
                "parallel": False
            },
            "gui": {
                "path": "tests/gui",
                "markers": "gui",
                "timeout": 900,  # 15 minutes
                "parallel": False,
                "requires_display": True
            },
            "performance": {
                "path": "tests/performance",
                "markers": "slow and performance",
                "timeout": 1800,  # 30 minutes
                "parallel": False
            },
            "api": {
                "path": "tests/api",
                "markers": "api",
                "timeout": 300,
                "parallel": True,
                "requires_network": True
            }
        }
    
    def check_environment(self) -> Dict[str, bool]:
        """Check test environment requirements"""
        print("🔍 Checking test environment...")
        
        checks = {
            "python_version": sys.version_info >= (3, 11),
            "pytest_available": self.check_package("pytest"),
            "coverage_available": self.check_package("coverage"),
            "display_available": self.check_display(),
            "network_available": self.check_network(),
            "test_directory": self.test_dir.exists()
        }
        
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {check}")
        
        return checks
    
    def check_package(self, package: str) -> bool:
        """Check if Python package is available"""
        try:
            __import__(package)
            return True
        except ImportError:
            return False
    
    def check_display(self) -> bool:
        """Check if display is available for GUI tests"""
        if os.name == 'nt':  # Windows
            return True
        else:
            return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
    
    def check_network(self) -> bool:
        """Check network connectivity"""
        try:
            import urllib.request
            urllib.request.urlopen('https://httpbin.org/get', timeout=5)
            return True
        except:
            return False
    
    def install_test_dependencies(self) -> bool:
        """Install test dependencies"""
        print("📦 Installing test dependencies...")
        
        test_deps = [
            "pytest>=7.0.0",
            "pytest-qt>=4.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xvfb>=2.0.0",  # For headless GUI tests
            "pytest-timeout>=2.1.0",
            "pytest-mock>=3.7.0",
            "pytest-asyncio>=0.20.0",
            "coverage[toml]>=6.3.0",
            "pytest-html>=3.1.0",
            "pytest-json-report>=1.5.0"
        ]
        
        try:
            for dep in test_deps:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)
                print(f"   ✅ {dep}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install dependencies: {e}")
            return False
    
    def run_test_suite(self, suite_name: str, **kwargs) -> Dict[str, Any]:
        """Run a specific test suite"""
        print(f"\n🧪 Running {suite_name} tests...")
        
        config = self.test_config.get(suite_name, {})
        if not config:
            print(f"❌ Unknown test suite: {suite_name}")
            return {"success": False, "error": "Unknown suite"}
        
        # Check requirements
        if config.get("requires_display", False) and not self.check_display():
            print("⚠️  Display not available, skipping GUI tests")
            return {"success": False, "error": "No display"}
        
        if config.get("requires_network", False) and not self.check_network():
            print("⚠️  Network not available, skipping API tests")
            return {"success": False, "error": "No network"}
        
        # Build pytest command
        cmd = [sys.executable, "-m", "pytest"]
        
        # Test path
        test_path = self.project_root / config["path"]
        if test_path.exists():
            cmd.append(str(test_path))
        else:
            print(f"⚠️  Test directory not found: {test_path}")
            return {"success": False, "error": "Test directory not found"}
        
        # Markers
        if config.get("markers"):
            cmd.extend(["-m", config["markers"]])
        
        # Parallel execution
        if config.get("parallel", False) and self.check_package("pytest-xdist"):
            cmd.extend(["-n", "auto"])
        
        # Coverage
        if kwargs.get("coverage", True):
            cmd.extend([
                "--cov=src",
                f"--cov-report=html:{self.reports_dir}/coverage_{suite_name}",
                f"--cov-report=xml:{self.reports_dir}/coverage_{suite_name}.xml",
                "--cov-report=term-missing"
            ])
        
        # Output formats
        report_base = self.reports_dir / f"report_{suite_name}"
        cmd.extend([
            f"--html={report_base}.html",
            "--self-contained-html",
            f"--json-report={report_base}.json",
            f"--junit-xml={report_base}.xml"
        ])
        
        # Timeout
        if config.get("timeout"):
            cmd.extend(["--timeout", str(config["timeout"])])
        
        # Verbose output
        if kwargs.get("verbose", False):
            cmd.append("-v")
        
        # Additional pytest args
        if kwargs.get("pytest_args"):
            cmd.extend(kwargs["pytest_args"])
        
        # Run tests
        start_time = time.time()
        
        try:
            print(f"   Command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=config.get("timeout", 3600)
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Parse results
            return self.parse_test_results(suite_name, result, duration)
            
        except subprocess.TimeoutExpired:
            print(f"   ❌ Tests timed out after {config['timeout']} seconds")
            return {
                "success": False,
                "error": "Timeout",
                "duration": config["timeout"]
            }
        except Exception as e:
            print(f"   ❌ Test execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": 0
            }
    
    def parse_test_results(self, suite_name: str, result: subprocess.CompletedProcess, duration: float) -> Dict[str, Any]:
        """Parse test results from pytest output"""
        success = result.returncode == 0
        
        # Try to parse JSON report
        json_report_path = self.reports_dir / f"report_{suite_name}.json"
        test_stats = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        
        try:
            if json_report_path.exists():
                with open(json_report_path, 'r') as f:
                    json_data = json.load(f)
                
                summary = json_data.get("summary", {})
                test_stats = {
                    "passed": summary.get("passed", 0),
                    "failed": summary.get("failed", 0),
                    "skipped": summary.get("skipped", 0),
                    "errors": summary.get("error", 0),
                    "total": summary.get("total", 0)
                }
        except Exception as e:
            print(f"   ⚠️  Failed to parse JSON report: {e}")
        
        # Parse coverage if available
        coverage_data = self.parse_coverage_report(suite_name)
        
        result_data = {
            "success": success,
            "duration": duration,
            "stats": test_stats,
            "coverage": coverage_data,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
        
        # Print summary
        if success:
            print(f"   ✅ {suite_name} tests passed")
        else:
            print(f"   ❌ {suite_name} tests failed")
        
        print(f"   📊 Results: {test_stats['passed']} passed, {test_stats['failed']} failed, {test_stats['skipped']} skipped")
        print(f"   ⏱️  Duration: {duration:.1f}s")
        
        if coverage_data:
            print(f"   📈 Coverage: {coverage_data.get('total_coverage', 0):.1f}%")
        
        return result_data
    
    def parse_coverage_report(self, suite_name: str) -> Optional[Dict[str, Any]]:
        """Parse coverage report"""
        coverage_xml = self.reports_dir / f"coverage_{suite_name}.xml"
        
        if not coverage_xml.exists():
            return None
        
        try:
            tree = ET.parse(coverage_xml)
            root = tree.getroot()
            
            # Get overall coverage
            coverage_elem = root.find(".//coverage")
            if coverage_elem is not None:
                line_rate = float(coverage_elem.get("line-rate", 0))
                branch_rate = float(coverage_elem.get("branch-rate", 0))
                
                return {
                    "total_coverage": line_rate * 100,
                    "branch_coverage": branch_rate * 100,
                    "lines_covered": int(coverage_elem.get("lines-covered", 0)),
                    "lines_valid": int(coverage_elem.get("lines-valid", 0))
                }
        
        except Exception as e:
            print(f"   ⚠️  Failed to parse coverage: {e}")
        
        return None
    
    def run_all_tests(self, **kwargs) -> Dict[str, Any]:
        """Run all test suites"""
        print("🚀 Running all test suites...")
        
        overall_results = {
            "start_time": datetime.now().isoformat(),
            "suites": {},
            "overall_success": True,
            "total_duration": 0
        }
        
        suite_order = ["unit", "integration", "gui", "performance", "api"]
        
        for suite_name in suite_order:
            if kwargs.get("skip_suite") and suite_name in kwargs["skip_suite"]:
                print(f"\n⏭️  Skipping {suite_name} tests")
                continue
            
            result = self.run_test_suite(suite_name, **kwargs)
            overall_results["suites"][suite_name] = result
            overall_results["total_duration"] += result.get("duration", 0)
            
            if not result.get("success", False):
                overall_results["overall_success"] = False
                
                if kwargs.get("fail_fast", False):
                    print(f"\n💥 Stopping due to {suite_name} test failures (--fail-fast)")
                    break
        
        overall_results["end_time"] = datetime.now().isoformat()
        
        # Generate summary report
        self.generate_summary_report(overall_results)
        
        return overall_results
    
    def generate_summary_report(self, results: Dict[str, Any]):
        """Generate summary test report"""
        print("\n" + "="*60)
        print("📋 TEST SUMMARY REPORT")
        print("="*60)
        
        # Overall status
        status = "✅ PASSED" if results["overall_success"] else "❌ FAILED"
        print(f"Overall Status: {status}")
        print(f"Total Duration: {results['total_duration']:.1f}s")
        
        # Suite breakdown
        print(f"\n📊 Suite Results:")
        for suite_name, suite_result in results["suites"].items():
            status_icon = "✅" if suite_result.get("success") else "❌"
            duration = suite_result.get("duration", 0)
            stats = suite_result.get("stats", {})
            
            print(f"   {status_icon} {suite_name:12s} | "
                  f"{duration:6.1f}s | "
                  f"P:{stats.get('passed', 0):3d} "
                  f"F:{stats.get('failed', 0):3d} "
                  f"S:{stats.get('skipped', 0):3d}")
        
        # Coverage summary
        print(f"\n📈 Coverage Summary:")
        for suite_name, suite_result in results["suites"].items():
            coverage = suite_result.get("coverage")
            if coverage:
                total_cov = coverage.get("total_coverage", 0)
                print(f"   {suite_name:12s}: {total_cov:5.1f}%")
        
        # Save detailed report
        report_file = self.reports_dir / "test_summary.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_file}")
        print(f"📁 Reports directory: {self.reports_dir}")
    
    def run_specific_tests(self, test_pattern: str, **kwargs) -> Dict[str, Any]:
        """Run specific tests matching a pattern"""
        print(f"🎯 Running tests matching: {test_pattern}")
        
        cmd = [
            sys.executable, "-m", "pytest",
            "-k", test_pattern,
            str(self.test_dir)
        ]
        
        if kwargs.get("verbose"):
            cmd.append("-v")
        
        if kwargs.get("coverage"):
            report_base = self.reports_dir / "report_specific"
            cmd.extend([
                "--cov=src",
                f"--cov-report=html:{report_base}_coverage",
                f"--cov-report=xml:{report_base}_coverage.xml",
                f"--html={report_base}.html",
                "--self-contained-html"
            ])
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, cwd=self.project_root, capture_output=True, text=True)
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "pattern": test_pattern
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "pattern": test_pattern
            }
   
    def run_coverage_report(self) -> bool:
        """Generate comprehensive coverage report"""
        print("📈 Generating coverage report...")
       
        try:
            # Combine coverage data from all test runs
            coverage_files = list(self.reports_dir.glob("coverage_*.xml"))
           
            if not coverage_files:
                print("   ⚠️  No coverage data found")
                return False
           
            # Generate combined HTML report
            cmd = [
                sys.executable, "-m", "coverage", "html",
                "-d", str(self.reports_dir / "coverage_combined"),
                "--title", "Options Calculator Pro - Combined Coverage"
            ]
           
            subprocess.run(cmd, cwd=self.project_root, check=True, capture_output=True)
           
            # Generate text report
            result = subprocess.run([
                sys.executable, "-m", "coverage", "report"
            ], cwd=self.project_root, capture_output=True, text=True)
           
            if result.returncode == 0:
                print("   ✅ Coverage report generated")
                print("\n📊 Coverage Summary:")
                print(result.stdout)
                return True
            else:
                print(f"   ❌ Coverage report failed: {result.stderr}")
                return False
       
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Coverage generation failed: {e}")
            return False
   
    def run_linting(self) -> Dict[str, Any]:
        """Run code quality checks"""
        print("🔍 Running code quality checks...")
       
        linting_results = {}
        
        # Black formatting check
        try:
            result = subprocess.run([
                sys.executable, "-m", "black", "--check", "--diff", "src/", "tests/"
            ], capture_output=True, text=True)
           
            linting_results["black"] = {
                "success": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
           
            status = "✅" if result.returncode == 0 else "❌"
            print(f"   {status} Black formatting")
           
        except FileNotFoundError:
            print("   ⚠️  Black not installed")
            linting_results["black"] = {"success": None, "error": "Not installed"}
       
        # Flake8 linting
        try:
            result = subprocess.run([
                sys.executable, "-m", "flake8", "src/", "tests/",
                "--output-file", str(self.reports_dir / "flake8_report.txt")
            ], capture_output=True, text=True)
           
            linting_results["flake8"] = {
                "success": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
           
            status = "✅" if result.returncode == 0 else "❌"
            print(f"   {status} Flake8 linting")
           
        except FileNotFoundError:
            print("   ⚠️  Flake8 not installed")
            linting_results["flake8"] = {"success": None, "error": "Not installed"}
       
        # MyPy type checking
        try:
            result = subprocess.run([
                sys.executable, "-m", "mypy", "src/",
                "--html-report", str(self.reports_dir / "mypy_report")
            ], capture_output=True, text=True)
           
            linting_results["mypy"] = {
                "success": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
           
            status = "✅" if result.returncode == 0 else "❌"
            print(f"   {status} MyPy type checking")
           
        except FileNotFoundError:
            print("   ⚠️  MyPy not installed")
            linting_results["mypy"] = {"success": None, "error": "Not installed"}
       
        # Overall success
        overall_success = all(
            result.get("success", False) 
            for result in linting_results.values() 
            if result.get("success") is not None
        )
       
        linting_results["overall_success"] = overall_success
       
        return linting_results
   
    def run_security_checks(self) -> Dict[str, Any]:
        """Run security vulnerability checks"""
        print("🔒 Running security checks...")
       
        security_results = {}
        
        # Bandit security linting
        try:
            result = subprocess.run([
                sys.executable, "-m", "bandit", "-r", "src/",
                "-f", "json", "-o", str(self.reports_dir / "bandit_report.json")
            ], capture_output=True, text=True)
           
            security_results["bandit"] = {
                "success": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
           
            status = "✅" if result.returncode == 0 else "❌"
            print(f"   {status} Bandit security scan")
           
        except FileNotFoundError:
            print("   ⚠️  Bandit not installed")
            security_results["bandit"] = {"success": None, "error": "Not installed"}
       
        # Safety dependency check
        try:
            result = subprocess.run([
                sys.executable, "-m", "safety", "check",
                "--json", "--output", str(self.reports_dir / "safety_report.json")
            ], capture_output=True, text=True)
           
            security_results["safety"] = {
                "success": result.returncode == 0,
                "output": result.stdout + result.stderr
            }
           
            status = "✅" if result.returncode == 0 else "❌"
            print(f"   {status} Safety dependency check")
           
        except FileNotFoundError:
            print("   ⚠️  Safety not installed")
            security_results["safety"] = {"success": None, "error": "Not installed"}
       
        return security_results
   
    def run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        print("⚡ Running performance benchmarks...")
       
        benchmark_cmd = [
            sys.executable, "-m", "pytest",
            "tests/performance/",
            "--benchmark-only",
            f"--benchmark-json={self.reports_dir}/benchmarks.json"
        ]
       
        try:
            result = subprocess.run(
                benchmark_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes
            )
           
            benchmark_results = {
                "success": result.returncode == 0,
                "duration": 0,  # Would be parsed from JSON
                "output": result.stdout + result.stderr
            }
           
            # Parse benchmark results if available
            benchmark_json = self.reports_dir / "benchmarks.json"
            if benchmark_json.exists():
                try:
                    with open(benchmark_json, 'r') as f:
                        bench_data = json.load(f)
                   
                    benchmark_results["benchmarks"] = bench_data.get("benchmarks", [])
                    benchmark_results["machine_info"] = bench_data.get("machine_info", {})
                   
                except Exception as e:
                    print(f"   ⚠️  Failed to parse benchmark results: {e}")
           
            status = "✅" if result.returncode == 0 else "❌"
            print(f"   {status} Performance benchmarks")
           
            return benchmark_results
           
        except subprocess.TimeoutExpired:
            print("   ❌ Benchmarks timed out")
            return {"success": False, "error": "Timeout"}
        except FileNotFoundError:
            print("   ⚠️  pytest-benchmark not installed")
            return {"success": None, "error": "Not installed"}
   
    def clean_reports(self):
        """Clean old test reports"""
        print("🧹 Cleaning old test reports...")
       
        if self.reports_dir.exists():
            for file_path in self.reports_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    print(f"   🗑️  Removed {file_path.name}")
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
                    print(f"   🗑️  Removed {file_path.name}/")
       
        print("   ✅ Reports cleaned")
   
    def generate_ci_report(self, results: Dict[str, Any]) -> bool:
        """Generate CI-friendly report"""
        print("📊 Generating CI report...")
       
        ci_report = {
            "success": results.get("overall_success", False),
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "total_duration": results.get("total_duration", 0),
                "coverage_percentage": 0
            },
            "suites": []
        }
       
        # Process suite results
        for suite_name, suite_result in results.get("suites", {}).items():
            stats = suite_result.get("stats", {})
            coverage = suite_result.get("coverage", {})
           
            suite_info = {
                "name": suite_name,
                "success": suite_result.get("success", False),
                "duration": suite_result.get("duration", 0),
                "tests": {
                    "total": stats.get("total", 0),
                    "passed": stats.get("passed", 0),
                    "failed": stats.get("failed", 0),
                    "skipped": stats.get("skipped", 0)
                },
                "coverage": coverage.get("total_coverage", 0)
            }
           
            ci_report["suites"].append(suite_info)
           
            # Update totals
            ci_report["summary"]["total_tests"] += stats.get("total", 0)
            ci_report["summary"]["passed_tests"] += stats.get("passed", 0)
            ci_report["summary"]["failed_tests"] += stats.get("failed", 0)
            ci_report["summary"]["skipped_tests"] += stats.get("skipped", 0)
       
        # Calculate average coverage
        coverages = [s["coverage"] for s in ci_report["suites"] if s["coverage"] > 0]
        if coverages:
            ci_report["summary"]["coverage_percentage"] = sum(coverages) / len(coverages)
       
        # Save CI report
        ci_report_file = self.reports_dir / "ci_report.json"
        with open(ci_report_file, 'w') as f:
            json.dump(ci_report, f, indent=2)
       
        # Generate simple status file for CI
        status_file = self.reports_dir / "test_status.txt"
        with open(status_file, 'w') as f:
            status = "PASS" if ci_report["success"] else "FAIL"
            f.write(f"{status}\n")
            f.write(f"Tests: {ci_report['summary']['passed_tests']}/{ci_report['summary']['total_tests']}\n")
            f.write(f"Coverage: {ci_report['summary']['coverage_percentage']:.1f}%\n")
            f.write(f"Duration: {ci_report['summary']['total_duration']:.1f}s\n")
       
        print(f"   ✅ CI report saved: {ci_report_file}")
        return True

def main():
    """Main test runner entry point"""
    import argparse
   
    parser = argparse.ArgumentParser(description="Run Options Calculator Pro tests")
    parser.add_argument(
        "suites",
        nargs="*",
        choices=["unit", "integration", "gui", "performance", "api", "all"],
        default=["unit"],
        help="Test suites to run"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        default=True,
        help="Generate coverage report"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Skip coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--pattern", "-k",
        help="Run tests matching pattern"
    )
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run linting checks"
    )
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run security checks"
    )
    parser.add_argument(
        "--benchmarks",
        action="store_true",
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clean old reports before running"
   )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies"
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Generate CI-friendly reports"
    )
    parser.add_argument(
        "--pytest-args",
        nargs=argparse.REMAINDER,
        help="Additional pytest arguments"
    )
   
    args = parser.parse_args()
   
    runner = TestRunner()
   
    # Check environment
    env_checks = runner.check_environment()
    if not env_checks["python_version"]:
        print("❌ Python 3.11+ required")
        sys.exit(1)
   
    # Install dependencies if requested
    if args.install_deps:
        if not runner.install_test_dependencies():
            sys.exit(1)
   
    # Clean reports if requested
    if args.clean:
        runner.clean_reports()
   
    # Set coverage flag
    coverage = args.coverage and not args.no_coverage
   
    kwargs = {
        "coverage": coverage,
        "verbose": args.verbose,
        "fail_fast": args.fail_fast,
        "pytest_args": args.pytest_args or []
    }
   
    success = True
   
    # Run specific pattern
    if args.pattern:
        result = runner.run_specific_tests(args.pattern, **kwargs)
        success = result.get("success", False)
   
    # Run test suites
    elif args.suites:
        if "all" in args.suites:
            results = runner.run_all_tests(**kwargs)
            success = results.get("overall_success", False)
           
            if args.ci:
                runner.generate_ci_report(results)
        else:
            results = {"suites": {}, "overall_success": True, "total_duration": 0}
           
            for suite in args.suites:
                result = runner.run_test_suite(suite, **kwargs)
                results["suites"][suite] = result
                results["total_duration"] += result.get("duration", 0)
               
                if not result.get("success", False):
                    results["overall_success"] = False
                    success = False
                   
                    if args.fail_fast:
                        break
           
            if args.ci:
                runner.generate_ci_report(results)
   
    # Run additional checks
    if args.lint:
        lint_results = runner.run_linting()
        if not lint_results.get("overall_success", False):
            success = False
   
    if args.security:
        security_results = runner.run_security_checks()
        # Security checks are informational, don't fail build
   
    if args.benchmarks:
        benchmark_results = runner.run_performance_benchmarks()
        # Benchmarks are informational, don't fail build
   
    # Generate coverage report
    if coverage and success:
        runner.run_coverage_report()
   
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
   main()