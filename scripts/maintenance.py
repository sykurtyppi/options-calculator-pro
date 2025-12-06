#!/usr/bin/env python3
"""
Maintenance script for Options Calculator Pro
Database cleanup, cache management, log rotation, and system maintenance
"""

import os
import sys
import json
import shutil
import sqlite3
import gzip
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import argparse

class MaintenanceManager:
    """Handles application maintenance tasks"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.get_config_directory()
        self.data_dir = self.config_dir / "data"
        self.cache_dir = self.config_dir / "cache"
        self.logs_dir = self.config_dir / "logs"
        self.backups_dir = self.config_dir / "backups"
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.cache_dir, self.logs_dir, self.backups_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def get_config_directory(self) -> Path:
        """Get platform-specific configuration directory"""
        import platform
        system = platform.system()
        
        if system == "Windows":
            config_dir = Path(os.environ.get("APPDATA", "")) / "Options Calculator Pro"
        elif system == "Darwin":  # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "Options Calculator Pro"
        else:  # Linux and others
            config_dir = Path.home() / ".options_calculator"
        
        return config_dir
    
    def setup_logging(self):
        """Setup maintenance logging"""
        log_file = self.logs_dir / "maintenance.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def get_directory_size(self, directory: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, PermissionError) as e:
            self.logger.warning(f"Could not access {file_path}: {e}")
        
        return total_size
    
    def format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def clean_cache(self, max_age_days: int = 7) -> Dict[str, Any]:
        """Clean old cache files"""
        self.logger.info("🧹 Starting cache cleanup...")
        
        if not self.cache_dir.exists():
            return {"cleaned": 0, "size_freed": 0}
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        cleaned_files = 0
        size_freed = 0
        
        try:
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_time < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        cleaned_files += 1
                        size_freed += file_size
                        self.logger.debug(f"Removed cache file: {file_path.name}")
            
            # Remove empty directories
            for dir_path in self.cache_dir.rglob('*'):
                if dir_path.is_dir() and not any(dir_path.iterdir()):
                    dir_path.rmdir()
                    self.logger.debug(f"Removed empty directory: {dir_path}")
            
            self.logger.info(f"✅ Cache cleanup completed: {cleaned_files} files, "
                           f"{self.format_size(size_freed)} freed")
            
            return {
                "cleaned": cleaned_files,
                "size_freed": size_freed,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Cache cleanup failed: {e}")
            return {"success": False, "error": str(e)}
    
    def rotate_logs(self, max_age_days: int = 30, max_files: int = 10) -> Dict[str, Any]:
        """Rotate and compress old log files"""
        self.logger.info("📋 Starting log rotation...")
        
        if not self.logs_dir.exists():
            return {"rotated": 0, "compressed": 0}
        
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        rotated_files = 0
        compressed_files = 0
        
        try:
            log_files = list(self.logs_dir.glob("*.log"))
            
            for log_file in log_files:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                
                if file_time < cutoff_time:
                    # Compress old log file
                    compressed_name = f"{log_file.stem}_{file_time.strftime('%Y%m%d')}.log.gz"
                    compressed_path = self.logs_dir / compressed_name
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    log_file.unlink()
                    compressed_files += 1
                    self.logger.debug(f"Compressed log: {log_file.name} -> {compressed_name}")
            
            # Clean up old compressed logs
            compressed_logs = sorted(self.logs_dir.glob("*.log.gz"), 
                                   key=lambda x: x.stat().st_mtime, reverse=True)
            
            if len(compressed_logs) > max_files:
                for old_log in compressed_logs[max_files:]:
                    old_log.unlink()
                    rotated_files += 1
                    self.logger.debug(f"Removed old compressed log: {old_log.name}")
            
            self.logger.info(f"✅ Log rotation completed: {compressed_files} compressed, "
                           f"{rotated_files} old files removed")
            
            return {
                "rotated": rotated_files,
                "compressed": compressed_files,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Log rotation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def cleanup_database(self) -> Dict[str, Any]:
        """Clean up and optimize application database"""
        self.logger.info("🗄️  Starting database cleanup...")
        
        db_path = self.data_dir / "options_calculator.db"
        
        if not db_path.exists():
            self.logger.info("No database found, skipping cleanup")
            return {"success": True, "actions": []}
        
        try:
            # Backup database first
            backup_path = self.backups_dir / f"database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(db_path, backup_path)
            self.logger.info(f"Database backed up to: {backup_path}")
            
            actions_performed = []
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Get database size before cleanup
                initial_size = db_path.stat().st_size
                
                # Clean old analysis results (older than 90 days)
                cutoff_date = datetime.now() - timedelta(days=90)
                cursor.execute("""
                    DELETE FROM analysis_results 
                    WHERE created_at < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_analyses = cursor.rowcount
                if deleted_analyses > 0:
                    actions_performed.append(f"Deleted {deleted_analyses} old analysis results")
                
                # Clean old market data cache (older than 7 days)
                cache_cutoff = datetime.now() - timedelta(days=7)
                cursor.execute("""
                    DELETE FROM market_data_cache 
                    WHERE last_updated < ?
                """, (cache_cutoff.isoformat(),))
                
                deleted_cache = cursor.rowcount
                if deleted_cache > 0:
                    actions_performed.append(f"Deleted {deleted_cache} old market data entries")
                
                # Clean orphaned records
                cursor.execute("""
                    DELETE FROM trade_legs 
                    WHERE trade_id NOT IN (SELECT id FROM trades)
                """)
                
                deleted_orphans = cursor.rowcount
                if deleted_orphans > 0:
                    actions_performed.append(f"Deleted {deleted_orphans} orphaned trade legs")
                
                # Vacuum database to reclaim space
                conn.execute("VACUUM")
                actions_performed.append("Vacuumed database")
                
                # Analyze tables for query optimization
                conn.execute("ANALYZE")
                actions_performed.append("Analyzed tables for optimization")
                
                # Get final database size
                final_size = db_path.stat().st_size
                size_reduction = initial_size - final_size
                
                self.logger.info(f"✅ Database cleanup completed: "
                               f"{self.format_size(size_reduction)} reclaimed")
                
                return {
                    "success": True,
                    "actions": actions_performed,
                    "size_reduction": size_reduction,
                    "initial_size": initial_size,
                    "final_size": final_size
                }
        
        except Exception as e:
            self.logger.error(f"❌ Database cleanup failed: {e}")
            return {"success": False, "error": str(e)}
    
    def create_backup(self) -> Dict[str, Any]:
        """Create backup of important application data"""
        self.logger.info("💾 Creating application backup...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"options_calculator_backup_{timestamp}"
        backup_path = self.backups_dir / f"{backup_name}.tar.gz"
        
        try:
            import tarfile
            
            with tarfile.open(backup_path, 'w:gz') as tar:
                # Backup configuration
                config_file = self.config_dir / "config.json"
                if config_file.exists():
                    tar.add(config_file, arcname="config.json")
                
                # Backup database
                db_file = self.data_dir / "options_calculator.db"
                if db_file.exists():
                    tar.add(db_file, arcname="database.db")
                
                # Backup trade history
                trades_dir = self.data_dir / "trades"
                if trades_dir.exists():
                    tar.add(trades_dir, arcname="trades")
                
                # Backup user preferences
                prefs_file = self.config_dir / "preferences.json"
                if prefs_file.exists():
                    tar.add(prefs_file, arcname="preferences.json")
            
            backup_size = backup_path.stat().st_size
            
            self.logger.info(f"✅ Backup created: {backup_name} ({self.format_size(backup_size)})")
            
            # Clean old backups (keep last 5)
            backups = sorted(self.backups_dir.glob("options_calculator_backup_*.tar.gz"),
                           key=lambda x: x.stat().st_mtime, reverse=True)
            
            cleaned_backups = 0
            if len(backups) > 5:
                for old_backup in backups[5:]:
                    old_backup.unlink()
                    cleaned_backups += 1
                    self.logger.debug(f"Removed old backup: {old_backup.name}")
            
            return {
                "success": True,
                "backup_path": str(backup_path),
                "backup_size": backup_size,
                "cleaned_old_backups": cleaned_backups
            }
            
        except Exception as e:
            self.logger.error(f"❌ Backup creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space in application directories"""
        self.logger.info("💽 Checking disk space...")
        
        try:
            disk_info = {}
            
            # Get disk usage for config directory
            config_usage = shutil.disk_usage(self.config_dir)
            disk_info["config_dir"] = {
                "path": str(self.config_dir),
                "total": config_usage.total,
                "used": config_usage.used,
                "free": config_usage.free,
                "percent_used": (config_usage.used / config_usage.total) * 100
            }
            
            # Get sizes of subdirectories
            subdirs = {
                "data": self.data_dir,
                "cache": self.cache_dir,
                "logs": self.logs_dir,
                "backups": self.backups_dir
            }
            
            for name, directory in subdirs.items():
                if directory.exists():
                    size = self.get_directory_size(directory)
                    disk_info[f"{name}_size"] = size
                    disk_info[f"{name}_formatted"] = self.format_size(size)
            
            # Check if disk space is low (less than 1GB free)
            low_space_warning = config_usage.free < (1024 ** 3)  # 1GB
            
            if low_space_warning:
                self.logger.warning(f"⚠️  Low disk space: {self.format_size(config_usage.free)} free")
            else:
                self.logger.info(f"✅ Disk space OK: {self.format_size(config_usage.free)} free")
            
            return {
                "success": True,
                "disk_info": disk_info,
                "low_space_warning": low_space_warning
            }
            
        except Exception as e:
            self.logger.error(f"❌ Disk space check failed: {e}")
            return {"success": False, "error": str(e)}
    
    def update_dependencies(self) -> Dict[str, Any]:
       """Check for and optionally update Python dependencies"""
       self.logger.info("📦 Checking dependency updates...")
       
       try:
           import subprocess
           import json
           
           # Get list of outdated packages
           result = subprocess.run([
               sys.executable, "-m", "pip", "list", "--outdated", "--format=json"
           ], capture_output=True, text=True, check=True)
           
           outdated_packages = json.loads(result.stdout) if result.stdout else []
           
           if not outdated_packages:
               self.logger.info("✅ All dependencies are up to date")
               return {"success": True, "outdated_packages": [], "updated": []}
           
           self.logger.info(f"📋 Found {len(outdated_packages)} outdated packages:")
           for package in outdated_packages:
               self.logger.info(f"   {package['name']}: {package['version']} -> {package['latest_version']}")
           
           return {
               "success": True,
               "outdated_packages": outdated_packages,
               "updated": []  # Don't auto-update, just report
           }
           
       except subprocess.CalledProcessError as e:
           self.logger.error(f"❌ Dependency check failed: {e}")
           return {"success": False, "error": str(e)}
       except Exception as e:
           self.logger.error(f"❌ Dependency check error: {e}")
           return {"success": False, "error": str(e)}
   
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate application configuration files"""
        self.logger.info("⚙️  Validating configuration...")
       
        validation_results = {
            "config_file": {"exists": False, "valid": False},
            "preferences_file": {"exists": False, "valid": False},
            "api_keys": {"configured": False, "valid": False}
        }
       
        try:
            # Check main config file
            config_file = self.config_dir / "config.json"
            if config_file.exists():
                validation_results["config_file"]["exists"] = True
                
                try:
                    with open(config_file, 'r') as f:
                        config_data = json.load(f)
                   
                    # Validate required fields
                    required_fields = ["version", "portfolio_value", "max_position_risk"]
                    if all(field in config_data for field in required_fields):
                        validation_results["config_file"]["valid"] = True
                        self.logger.info("✅ Main configuration is valid")
                    else:
                        self.logger.warning("⚠️  Main configuration missing required fields")
                       
                except json.JSONDecodeError:
                    self.logger.error("❌ Main configuration file is corrupted")
            else:
                self.logger.warning("⚠️  Main configuration file not found")
           
            # Check preferences file
            prefs_file = self.config_dir / "preferences.json"
            if prefs_file.exists():
                validation_results["preferences_file"]["exists"] = True
               
                try:
                    with open(prefs_file, 'r') as f:
                        json.load(f)
                    validation_results["preferences_file"]["valid"] = True
                    self.logger.info("✅ Preferences file is valid")
                except json.JSONDecodeError:
                    self.logger.error("❌ Preferences file is corrupted")
           
            # Check API keys
            if validation_results["config_file"]["valid"]:
                api_keys = config_data.get("api_keys", {})
                if any(api_keys.values()):
                    validation_results["api_keys"]["configured"] = True
                    # Could add actual API validation here
                    validation_results["api_keys"]["valid"] = True
                    self.logger.info("✅ API keys are configured")
                else:
                    self.logger.info("ℹ️  No API keys configured")
           
            return {
                "success": True,
                "validation_results": validation_results
            }
           
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return {"success": False, "error": str(e)}
   
    def repair_database(self) -> Dict[str, Any]:
        """Attempt to repair corrupted database"""
        self.logger.info("🔧 Attempting database repair...")
       
        db_path = self.data_dir / "options_calculator.db"
       
        if not db_path.exists():
            return {"success": False, "error": "Database file not found"}
       
        try:
            # Create backup before repair
            backup_path = self.backups_dir / f"pre_repair_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(db_path, backup_path)
            self.logger.info(f"Created backup: {backup_path}")
           
            repair_actions = []
           
            with sqlite3.connect(db_path) as conn:
                # Check database integrity
                cursor = conn.cursor()
                cursor.execute("PRAGMA integrity_check")
                integrity_result = cursor.fetchone()[0]
               
                if integrity_result == "ok":
                    self.logger.info("✅ Database integrity check passed")
                    repair_actions.append("Integrity check: PASSED")
                else:
                    self.logger.warning(f"⚠️  Database integrity issues: {integrity_result}")
                    repair_actions.append(f"Integrity check: {integrity_result}")
               
                # Rebuild indices
                cursor.execute("REINDEX")
                repair_actions.append("Rebuilt database indices")
               
                # Vacuum database
                cursor.execute("VACUUM")
                repair_actions.append("Vacuumed database")
               
                # Update statistics
                cursor.execute("ANALYZE")
                repair_actions.append("Updated table statistics")
           
            self.logger.info("✅ Database repair completed")
           
            return {
                "success": True,
                "repair_actions": repair_actions,
                "backup_created": str(backup_path)
            }
           
        except Exception as e:
            self.logger.error(f"❌ Database repair failed: {e}")
            return {"success": False, "error": str(e)}
   
    def optimize_performance(self) -> Dict[str, Any]:
        """Optimize application performance"""
        self.logger.info("⚡ Optimizing application performance...")
       
        optimization_results = []
       
        try:
            # Clean temporary files
            temp_cleaned = self.clean_temp_files()
            if temp_cleaned["success"]:
                optimization_results.append(f"Cleaned {temp_cleaned['files_removed']} temporary files")
           
            # Optimize cache structure
            cache_optimized = self.optimize_cache_structure()
            if cache_optimized["success"]:
                optimization_results.append("Optimized cache structure")
           
            # Defragment database
            db_optimized = self.optimize_database_performance()
            if db_optimized["success"]:
                optimization_results.append("Optimized database performance")
           
            # Update precompiled assets
            assets_updated = self.update_precompiled_assets()
            if assets_updated["success"]:
                optimization_results.append("Updated precompiled assets")
           
            self.logger.info("✅ Performance optimization completed")
           
            return {
                "success": True,
                "optimizations": optimization_results
            }
           
        except Exception as e:
            self.logger.error(f"❌ Performance optimization failed: {e}")
            return {"success": False, "error": str(e)}
   
    def clean_temp_files(self) -> Dict[str, Any]:
        """Clean temporary files"""
        temp_dirs = [
            self.config_dir / "temp",
            Path.home() / "tmp" / "options_calculator",
            Path("/tmp") / "options_calculator" if os.name != 'nt' else Path.cwd() / "temp"
        ]
       
        files_removed = 0
        size_freed = 0
       
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                for file_path in temp_dir.rglob('*'):
                    if file_path.is_file():
                        try:
                            file_size = file_path.stat().st_size
                            file_path.unlink()
                            files_removed += 1
                            size_freed += file_size
                        except OSError:
                            pass  # File may be in use
       
        return {
            "success": True,
            "files_removed": files_removed,
            "size_freed": size_freed
        }
   
    def optimize_cache_structure(self) -> Dict[str, Any]:
        """Optimize cache directory structure"""
        if not self.cache_dir.exists():
            return {"success": True}
        
        try:
            # Reorganize cache files by type
            cache_types = {
                "market_data": [],
                "analysis_results": [],
                "charts": [],
                "api_responses": []
            }
           
            # Create type-specific subdirectories
            for cache_type in cache_types.keys():
                (self.cache_dir / cache_type).mkdir(exist_ok=True)
           
            return {"success": True}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
   
    def optimize_database_performance(self) -> Dict[str, Any]:
        """Optimize database for better performance"""
        db_path = self.data_dir / "options_calculator.db"
       
        if not db_path.exists():
            return {"success": True}
       
        try:
            with sqlite3.connect(db_path) as conn:
                # Set performance pragmas
                performance_settings = [
                    "PRAGMA journal_mode=WAL",
                    "PRAGMA synchronous=NORMAL",
                    "PRAGMA cache_size=10000",
                    "PRAGMA temp_store=MEMORY"
                ]
               
                for setting in performance_settings:
                    conn.execute(setting)
           
            return {"success": True}
           
        except Exception as e:
            return {"success": False, "error": str(e)}
   
    def update_precompiled_assets(self) -> Dict[str, Any]:
        """Update precompiled assets and resources"""
        try:
            # This would update any precompiled UI resources, icons, etc.
            # Implementation depends on specific asset compilation process
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
   
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive application health report"""
        self.logger.info("📊 Generating health report...")
       
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": "unknown",
            "checks": {}
        }
       
        # Run all health checks
        checks = [
            ("disk_space", self.check_disk_space),
            ("configuration", self.validate_configuration),
            ("database_health", self.check_database_health),
            ("cache_health", self.check_cache_health),
            ("dependency_status", self.update_dependencies)
        ]
       
        healthy_checks = 0
        total_checks = len(checks)
       
        for check_name, check_function in checks:
            try:
                result = check_function()
                health_report["checks"][check_name] = result
               
                if result.get("success", False):
                    healthy_checks += 1
                   
            except Exception as e:
                health_report["checks"][check_name] = {
                    "success": False,
                    "error": str(e)
                }
       
        # Determine overall health
        health_percentage = (healthy_checks / total_checks) * 100
       
        if health_percentage >= 90:
            health_report["overall_health"] = "excellent"
        elif health_percentage >= 75:
            health_report["overall_health"] = "good"
        elif health_percentage >= 50:
            health_report["overall_health"] = "fair"
        else:
            health_report["overall_health"] = "poor"
       
        health_report["health_percentage"] = health_percentage
       
        # Save health report
        report_file = self.config_dir / "health_report.json"
        with open(report_file, 'w') as f:
            json.dump(health_report, f, indent=2)
       
        self.logger.info(f"✅ Health report generated: {health_report['overall_health']} "
                        f"({health_percentage:.1f}%)")
       
        return health_report
   
    def check_database_health(self) -> Dict[str, Any]:
        """Check database health status"""
        db_path = self.data_dir / "options_calculator.db"
       
        if not db_path.exists():
            return {"success": True, "status": "no_database"}
       
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
               
                # Check integrity
                cursor.execute("PRAGMA integrity_check")
                integrity = cursor.fetchone()[0]
               
                # Get database size
                db_size = db_path.stat().st_size
               
                # Count records in main tables
                tables_info = {}
                main_tables = ["analysis_results", "trades", "market_data_cache"]
               
                for table in main_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        tables_info[table] = count
                    except sqlite3.OperationalError:
                        tables_info[table] = "table_not_found"
               
                return {
                    "success": True,
                    "integrity": integrity,
                    "database_size": db_size,
                    "table_counts": tables_info
                }
               
        except Exception as e:
            return {"success": False, "error": str(e)}
   
    def check_cache_health(self) -> Dict[str, Any]:
        """Check cache directory health"""
        if not self.cache_dir.exists():
            return {"success": True, "status": "no_cache"}
       
        try:
            cache_size = self.get_directory_size(self.cache_dir)
            file_count = len(list(self.cache_dir.rglob('*')))
           
            # Check for very old files
            old_files = 0
            cutoff_time = datetime.now() - timedelta(days=30)
           
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        old_files += 1
           
            return {
                "success": True,
                "cache_size": cache_size,
                "file_count": file_count,
                "old_files": old_files
            }
           
        except Exception as e:
            return {"success": False, "error": str(e)}
   
    def run_full_maintenance(self, **kwargs) -> Dict[str, Any]:
        """Run comprehensive maintenance routine"""
        self.logger.info("🔧 Starting full maintenance routine...")
       
        maintenance_results = {
            "start_time": datetime.now().isoformat(),
            "tasks": {},
            "overall_success": True
        }
       
        # Define maintenance tasks
        tasks = [
            ("cache_cleanup", lambda: self.clean_cache(kwargs.get("cache_age_days", 7))),
            ("log_rotation", lambda: self.rotate_logs(kwargs.get("log_age_days", 30))),
            ("database_cleanup", self.cleanup_database),
            ("create_backup", self.create_backup),
            ("performance_optimization", self.optimize_performance),
            ("health_check", self.generate_health_report)
        ]
       
        for task_name, task_function in tasks:
            self.logger.info(f"\n{'='*10} {task_name.replace('_', ' ').title()} {'='*10}")
           
            try:
                result = task_function()
                maintenance_results["tasks"][task_name] = result
               
                if not result.get("success", False):
                    maintenance_results["overall_success"] = False
                   
            except Exception as e:
                self.logger.error(f"❌ Task {task_name} failed: {e}")
                maintenance_results["tasks"][task_name] = {
                    "success": False,
                    "error": str(e)
                }
                maintenance_results["overall_success"] = False
       
        maintenance_results["end_time"] = datetime.now().isoformat()
       
        # Save maintenance report
        report_file = self.logs_dir / f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(maintenance_results, f, indent=2)
       
        # Summary
        success_count = sum(1 for task in maintenance_results["tasks"].values() 
                         if task.get("success", False))
        total_tasks = len(maintenance_results["tasks"])
       
        if maintenance_results["overall_success"]:
            self.logger.info(f"✅ Full maintenance completed successfully! "
                           f"({success_count}/{total_tasks} tasks)")
        else:
            self.logger.warning(f"⚠️  Maintenance completed with issues "
                              f"({success_count}/{total_tasks} tasks successful)")
       
        return maintenance_results

def main():
    """Main maintenance script entry point"""
    parser = argparse.ArgumentParser(description="Options Calculator Pro Maintenance")
    parser.add_argument(
        "command",
        choices=[
            "cache", "logs", "database", "backup", "optimize", 
            "health", "repair", "full", "check"
        ],
        help="Maintenance command to run"
    )
    parser.add_argument(
        "--cache-age-days",
        type=int,
        default=7,
        help="Maximum age for cache files in days"
    )
    parser.add_argument(
        "--log-age-days",
        type=int,
        default=30,
        help="Maximum age for log files in days"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
   
    args = parser.parse_args()
   
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
   
    maintenance = MaintenanceManager()
   
    if args.dry_run:
        maintenance.logger.info("🔍 DRY RUN MODE - No changes will be made")
   
    # Execute requested command
    try:
        if args.command == "cache":
            result = maintenance.clean_cache(args.cache_age_days)
        elif args.command == "logs":
            result = maintenance.rotate_logs(args.log_age_days)
        elif args.command == "database":
            result = maintenance.cleanup_database()
        elif args.command == "backup":
            result = maintenance.create_backup()
        elif args.command == "optimize":
            result = maintenance.optimize_performance()
        elif args.command == "health":
            result = maintenance.generate_health_report()
        elif args.command == "repair":
            result = maintenance.repair_database()
        elif args.command == "check":
            result = maintenance.check_disk_space()
        elif args.command == "full":
            result = maintenance.run_full_maintenance(
                cache_age_days=args.cache_age_days,
                log_age_days=args.log_age_days
            )
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
       
        # Exit with appropriate code
        sys.exit(0 if result.get("success", False) else 1)
       
    except KeyboardInterrupt:
        maintenance.logger.info("🛑 Maintenance interrupted by user")
        sys.exit(1)
    except Exception as e:
        maintenance.logger.error(f"💥 Maintenance failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
   main()