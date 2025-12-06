#!/usr/bin/env python3
"""
Deployment script for Options Calculator Pro
Handles uploading releases and managing deployments
"""

import os
import sys
import subprocess
import json
import requests
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

class DeploymentManager:
    """Manages deployment of Options Calculator Pro"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.dist_dir = self.project_root / "dist"
        self.config_file = self.project_root / "deploy_config.json"
        self.version = self.get_version()
        
        # Load deployment configuration
        self.config = self.load_config()
        
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
    
    def load_config(self) -> Dict[str, Any]:
        """Load deployment configuration"""
        default_config = {
            "github": {
                "owner": "username",
                "repo": "options-calculator-pro",
                "token": ""
            },
            "pypi": {
                "username": "__token__",
                "password": ""
            },
            "s3": {
                "bucket": "optionscalculatorpro-releases",
                "region": "us-east-1",
                "access_key": "",
                "secret_key": ""
            },
            "docker": {
                "registry": "docker.io",
                "username": "",
                "password": "",
                "image_name": "optionscalculatorpro/app"
            },
            "notification": {
                "slack_webhook": "",
                "discord_webhook": "",
                "email": {
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                }
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                default_config.update(config)
                return default_config
            except Exception as e:
                print(f"⚠️  Failed to load config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save deployment configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"✅ Configuration saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Failed to save config: {e}")
    
    def validate_artifacts(self) -> List[Path]:
        """Validate build artifacts exist"""
        print("🔍 Validating build artifacts...")
        
        if not self.dist_dir.exists():
            print(f"❌ Distribution directory not found: {self.dist_dir}")
            return []
        
        artifacts = list(self.dist_dir.glob("*"))
        valid_artifacts = []
        
        for artifact in artifacts:
            if artifact.is_file() and artifact.suffix in ['.exe', '.dmg', '.zip', '.tar.gz', '.whl']:
                print(f"   ✅ {artifact.name}")
                valid_artifacts.append(artifact)
            elif artifact.name in ['checksums.txt', 'build_report.json']:
                print(f"   📄 {artifact.name}")
                valid_artifacts.append(artifact)
        
        if not valid_artifacts:
            print("❌ No valid build artifacts found")
            return []
        
        print(f"✅ Found {len(valid_artifacts)} valid artifacts")
        return valid_artifacts
    
    def create_github_release(self, artifacts: List[Path]) -> bool:
        """Create GitHub release"""
        print("🐙 Creating GitHub release...")
        
        github_config = self.config.get("github", {})
        token = github_config.get("token")
        owner = github_config.get("owner")
        repo = github_config.get("repo")
        
        if not all([token, owner, repo]):
            print("❌ GitHub configuration incomplete")
            return False
        
        # Create release
        release_data = {
            "tag_name": f"v{self.version}",
            "target_commitish": "main",
            "name": f"Options Calculator Pro v{self.version}",
            "body": self.generate_release_notes(),
            "draft": False,
            "prerelease": False
        }
        
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            # Create release
            url = f"https://api.github.com/repos/{owner}/{repo}/releases"
            response = requests.post(url, json=release_data, headers=headers)
            
            if response.status_code == 201:
                release = response.json()
                print(f"   ✅ Release created: {release['html_url']}")
                
                # Upload artifacts
                upload_url = release["upload_url"].replace("{?name,label}", "")
                
                for artifact in artifacts:
                    if not self.upload_github_asset(upload_url, artifact, headers):
                        return False
                
                return True
            else:
                print(f"   ❌ Failed to create release: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ GitHub release error: {e}")
            return False
    
    def upload_github_asset(self, upload_url: str, artifact: Path, headers: Dict[str, str]) -> bool:
        """Upload asset to GitHub release"""
        print(f"   📤 Uploading {artifact.name}...")
        
        try:
            upload_headers = headers.copy()
            upload_headers["Content-Type"] = "application/octet-stream"
            
            params = {"name": artifact.name}
            
            with open(artifact, 'rb') as f:
                response = requests.post(
                    upload_url, 
                    data=f, 
                    headers=upload_headers, 
                    params=params
                )
            
            if response.status_code == 201:
                print(f"      ✅ {artifact.name} uploaded")
                return True
            else:
                print(f"      ❌ Upload failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"      ❌ Upload error: {e}")
            return False
    
    def deploy_to_pypi(self) -> bool:
        """Deploy Python package to PyPI"""
        print("🐍 Deploying to PyPI...")
        
        pypi_config = self.config.get("pypi", {})
        username = pypi_config.get("username")
        password = pypi_config.get("password")
        
        if not all([username, password]):
            print("❌ PyPI configuration incomplete")
            return False
        
        # Find wheel and source distribution
        wheel_files = list(self.dist_dir.glob("*.whl"))
        source_files = list(self.dist_dir.glob("*.tar.gz"))
        
        if not wheel_files and not source_files:
            print("❌ No Python packages found")
            return False
        
        try:
            # Upload with twine
            files_to_upload = wheel_files + source_files
            cmd = [
                sys.executable, "-m", "twine", "upload",
                "--username", username,
                "--password", password,
                *[str(f) for f in files_to_upload]
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("   ✅ Successfully uploaded to PyPI")
                return True
            else:
                print(f"   ❌ PyPI upload failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ PyPI deployment error: {e}")
            return False
    
    def deploy_to_s3(self, artifacts: List[Path]) -> bool:
        """Deploy artifacts to S3"""
        print("☁️  Deploying to S3...")
        
        s3_config = self.config.get("s3", {})
        bucket = s3_config.get("bucket")
        access_key = s3_config.get("access_key")
        secret_key = s3_config.get("secret_key")
        region = s3_config.get("region", "us-east-1")
        
        if not all([bucket, access_key, secret_key]):
            print("❌ S3 configuration incomplete")
            return False
        
        try:
            import boto3
            
            session = boto3.Session(
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region
            )
            s3 = session.client('s3')
            
            # Upload artifacts
            for artifact in artifacts:
                key = f"releases/v{self.version}/{artifact.name}"
                
                print(f"   📤 Uploading {artifact.name}...")
                
                s3.upload_file(
                    str(artifact),
                    bucket,
                    key,
                    ExtraArgs={'ACL': 'public-read'}
                )
                
                print(f"      ✅ {artifact.name} uploaded")
            
            print("   ✅ All files uploaded to S3")
            return True
            
        except ImportError:
            print("   ❌ boto3 not installed (pip install boto3)")
            return False
        except Exception as e:
            print(f"   ❌ S3 deployment error: {e}")
            return False
    
    def build_docker_image(self) -> bool:
        """Build and push Docker image"""
        print("🐳 Building Docker image...")
        
        docker_config = self.config.get("docker", {})
        registry = docker_config.get("registry", "docker.io")
        username = docker_config.get("username")
        password = docker_config.get("password")
        image_name = docker_config.get("image_name", "optionscalculatorpro/app")
        
        if not all([username, password]):
            print("❌ Docker configuration incomplete")
            return False
        
        try:
            # Check if Dockerfile exists
            dockerfile = self.project_root / "Dockerfile"
            if not dockerfile.exists():
                print("❌ Dockerfile not found")
                return False
            
            # Build image
            image_tag = f"{registry}/{image_name}:{self.version}"
            latest_tag = f"{registry}/{image_name}:latest"
            
            print(f"   🔨 Building {image_tag}...")
            
            build_cmd = [
                "docker", "build",
                "-t", image_tag,
                "-t", latest_tag,
                "."
            ]
            
            result = subprocess.run(build_cmd, cwd=self.project_root, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"   ❌ Docker build failed: {result.stderr}")
                return False
            
            print("   ✅ Docker image built")
            
            # Login to registry
            login_cmd = ["docker", "login", registry, "-u", username, "-p", password]
            result = subprocess.run(login_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"   ❌ Docker login failed: {result.stderr}")
                return False
            
            # Push images
            for tag in [image_tag, latest_tag]:
                print(f"   📤 Pushing {tag}...")
                
                push_cmd = ["docker", "push", tag]
                result = subprocess.run(push_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"   ❌ Docker push failed: {result.stderr}")
                    return False
                
                print(f"      ✅ {tag} pushed")
            
            print("   ✅ Docker deployment complete")
            return True
            
        except Exception as e:
            print(f"   ❌ Docker deployment error: {e}")
            return False
    
    def generate_release_notes(self) -> str:
        """Generate release notes"""
        changelog_file = self.project_root / "CHANGELOG.md"
        
        if changelog_file.exists():
            try:
                with open(changelog_file, 'r') as f:
                    content = f.read()
                
                # Extract notes for current version
                lines = content.split('\n')
                in_current_version = False
                notes = []
                
                for line in lines:
                    if line.startswith(f'## [{self.version}]') or line.startswith(f'## {self.version}'):
                        in_current_version = True
                        continue
                    elif line.startswith('## ') and in_current_version:
                        break
                    elif in_current_version:
                        notes.append(line)
                
                if notes:
                    return '\n'.join(notes).strip()
                    
            except Exception as e:
               print(f"⚠️  Failed to read changelog: {e}")
       
        # Default release notes
        return f"""## Options Calculator Pro v{self.version}

### What's New
- Performance improvements and bug fixes
- Enhanced analysis algorithms
- Updated dependencies

### Download
Choose the appropriate package for your platform:
- **Windows**: `.exe` installer
- **macOS**: `.dmg` disk image  
- **Linux**: `.tar.gz` archive
- **Python**: `.whl` wheel package

### Installation
See the [Installation Guide](docs/user-guide/installation.md) for detailed instructions.

### Support
- [Documentation](docs/README.md)
- [GitHub Issues](https://github.com/{self.config['github']['owner']}/{self.config['github']['repo']}/issues)
- [Discord Community](https://discord.gg/options-calc-pro)

Built on {datetime.now().strftime('%Y-%m-%d')}"""
   
    def send_notifications(self, success: bool, deployment_targets: List[str]) -> bool:
        """Send deployment notifications"""
        print("📢 Sending notifications...")
       
        notification_config = self.config.get("notification", {})
       
        message = self.create_notification_message(success, deployment_targets)
        
        success_count = 0
       
        # Slack notification
        slack_webhook = notification_config.get("slack_webhook")
        if slack_webhook:
            if self.send_slack_notification(slack_webhook, message, success):
                success_count += 1
       
        # Discord notification
        discord_webhook = notification_config.get("discord_webhook")
        if discord_webhook:
            if self.send_discord_notification(discord_webhook, message, success):
                success_count += 1
       
        # Email notification
        email_config = notification_config.get("email", {})
        if email_config.get("smtp_server") and email_config.get("recipients"):
            if self.send_email_notification(email_config, message, success):
                success_count += 1
       
        if success_count > 0:
            print(f"   ✅ {success_count} notifications sent")
        else:
            print("   ⚠️  No notifications configured or sent")
       
        return True
   
    def create_notification_message(self, success: bool, targets: List[str]) -> Dict[str, str]:
        """Create notification message"""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        emoji = "🎉" if success else "💥"
       
        return {
            "title": f"{emoji} Deployment {status}",
            "text": f"Options Calculator Pro v{self.version} deployment {status.lower()}",
            "details": f"""
**Version:** {self.version}
**Targets:** {', '.join(targets)}
**Status:** {status}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Platform:** {os.name}

{f"🚀 Release available at: https://github.com/{self.config['github']['owner']}/{self.config['github']['repo']}/releases/tag/v{self.version}" if success else ""}
""".strip()
       }
   
    def send_slack_notification(self, webhook: str, message: Dict[str, str], success: bool) -> bool:
        """Send Slack notification"""
        try:
            color = "good" if success else "danger"
           
            payload = {
                "text": message["title"],
                "attachments": [
                    {
                        "color": color,
                        "title": message["text"],
                        "text": message["details"],
                        "footer": "Options Calculator Pro Deploy Bot",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
           
            response = requests.post(webhook, json=payload)
            
            if response.status_code == 200:
                print("   ✅ Slack notification sent")
                return True
            else:
                print(f"   ❌ Slack notification failed: {response.status_code}")
                return False
               
        except Exception as e:
            print(f"   ❌ Slack notification error: {e}")
            return False
   
    def send_discord_notification(self, webhook: str, message: Dict[str, str], success: bool) -> bool:
        """Send Discord notification"""
        try:
            color = 0x00ff00 if success else 0xff0000  # Green or red
            
            payload = {
                "embeds": [
                    {
                        "title": message["title"],
                        "description": message["details"],
                        "color": color,
                        "footer": {
                            "text": "Options Calculator Pro Deploy Bot"
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                ]
            }
           
            response = requests.post(webhook, json=payload)
           
            if response.status_code == 204:
                print("   ✅ Discord notification sent")
                return True
            else:
                print(f"   ❌ Discord notification failed: {response.status_code}")
                return False
               
        except Exception as e:
            print(f"   ❌ Discord notification error: {e}")
            return False
   
    def send_email_notification(self, email_config: Dict[str, Any], message: Dict[str, str], success: bool) -> bool:
        """Send email notification"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
           
            smtp_server = email_config["smtp_server"]
            smtp_port = email_config.get("smtp_port", 587)
            username = email_config["username"]
            password = email_config["password"]
            recipients = email_config["recipients"]
           
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[Deploy] {message['title']}"
           
            msg.attach(MIMEText(message["details"], 'plain'))
           
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
           
            print("   ✅ Email notification sent")
            return True
           
        except ImportError:
            print("   ❌ Email requires smtplib (standard library)")
            return False
        except Exception as e:
            print(f"   ❌ Email notification error: {e}")
            return False
   
    def update_version_file(self, new_version: str) -> bool:
        """Update version in source files"""
        print(f"📝 Updating version to {new_version}...")
       
        files_to_update = [
            self.project_root / "src" / "options_calculator" / "__init__.py",
            self.project_root / "setup.py",
            self.project_root / "pyproject.toml"
        ]
       
        for file_path in files_to_update:
            if not file_path.exists():
                continue
               
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
               
                # Update version patterns
                import re
                patterns = [
                    (r'__version__\s*=\s*["\'][^"\']*["\']', f'__version__ = "{new_version}"'),
                    (r'version\s*=\s*["\'][^"\']*["\']', f'version = "{new_version}"'),
                    (r'Version:\s*[^\n]*', f'Version: {new_version}'),
                ]
               
                updated = False
                for pattern, replacement in patterns:
                    if re.search(pattern, content):
                        content = re.sub(pattern, replacement, content)
                        updated = True
               
                if updated:
                    with open(file_path, 'w') as f:
                        f.write(content)
                    print(f"   ✅ Updated {file_path.name}")
               
            except Exception as e:
                print(f"   ❌ Failed to update {file_path}: {e}")
                return False
       
        return True
   
    def create_git_tag(self) -> bool:
        """Create and push git tag"""
        print(f"🏷️  Creating git tag v{self.version}...")
        
        try:
            # Check if tag already exists
            result = subprocess.run(
                ["git", "tag", "-l", f"v{self.version}"],
                capture_output=True, text=True
            )
           
            if result.stdout.strip():
                print(f"   ⚠️  Tag v{self.version} already exists")
                return True
           
            # Create tag
            subprocess.run([
                "git", "tag", "-a", f"v{self.version}",
                "-m", f"Release version {self.version}"
            ], check=True)
           
            # Push tag
            subprocess.run([
                "git", "push", "origin", f"v{self.version}"
            ], check=True)
           
            print(f"   ✅ Tag v{self.version} created and pushed")
            return True
           
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Git tag failed: {e}")
            return False
   
    def run_deployment(self, targets: List[str]) -> bool:
        """Run deployment process"""
        print("🚀 Starting deployment process...")
        print(f"Version: {self.version}")
        print(f"Targets: {', '.join(targets)}")
        print("=" * 50)
       
        # Validate artifacts
        artifacts = self.validate_artifacts()
        if not artifacts:
            return False
       
        # Run deployment steps
        success = True
        completed_targets = []
       
        for target in targets:
            print(f"\n{'='*10} Deploying to {target.upper()} {'='*10}")
           
            if target == "github":
                if self.create_github_release(artifacts):
                    completed_targets.append(target)
                else:
                    success = False
                   
            elif target == "pypi":
                if self.deploy_to_pypi():
                    completed_targets.append(target)
                else:
                    success = False
                   
            elif target == "s3":
                if self.deploy_to_s3(artifacts):
                    completed_targets.append(target)
                else:
                    success = False
                   
            elif target == "docker":
                if self.build_docker_image():
                    completed_targets.append(target)
                else:
                    success = False
                   
            else:
                print(f"❌ Unknown target: {target}")
                success = False
       
        # Send notifications
        self.send_notifications(success, completed_targets)
       
        # Summary
        print("\n" + "=" * 50)
        if success:
            print("🎉 Deployment completed successfully!")
            print(f"✅ Deployed to: {', '.join(completed_targets)}")
        else:
            print("💥 Deployment failed!")
            print(f"✅ Completed: {', '.join(completed_targets)}")
            print(f"❌ Failed: {', '.join(set(targets) - set(completed_targets))}")
       
        return success
   
    def setup_deployment_config(self):
        """Interactive deployment configuration setup"""
        print("⚙️  Deployment Configuration Setup")
        print("=" * 40)
       
        # GitHub configuration
        print("\n🐙 GitHub Configuration")
        self.config["github"]["owner"] = input(f"GitHub owner [{self.config['github']['owner']}]: ").strip() or self.config["github"]["owner"]
        self.config["github"]["repo"] = input(f"GitHub repo [{self.config['github']['repo']}]: ").strip() or self.config["github"]["repo"]
       
        token = input("GitHub token (leave empty to skip): ").strip()
        if token:
            self.config["github"]["token"] = token
       
        # PyPI configuration
        print("\n🐍 PyPI Configuration")
        pypi_username = input(f"PyPI username [{self.config['pypi']['username']}]: ").strip()
        if pypi_username:
            self.config["pypi"]["username"] = pypi_username
       
        pypi_password = input("PyPI password/token (leave empty to skip): ").strip()
        if pypi_password:
            self.config["pypi"]["password"] = pypi_password
       
        # Docker configuration
        print("\n🐳 Docker Configuration")
        docker_username = input("Docker username (leave empty to skip): ").strip()
        if docker_username:
            self.config["docker"]["username"] = docker_username
            docker_password = input("Docker password: ").strip()
            if docker_password:
                self.config["docker"]["password"] = docker_password
       
        # Notification configuration
        print("\n📢 Notification Configuration")
        slack_webhook = input("Slack webhook URL (leave empty to skip): ").strip()
        if slack_webhook:
            self.config["notification"]["slack_webhook"] = slack_webhook
       
        discord_webhook = input("Discord webhook URL (leave empty to skip): ").strip()
        if discord_webhook:
            self.config["notification"]["discord_webhook"] = discord_webhook
       
        # Save configuration
        self.save_config()
        print("\n✅ Configuration saved!")

def main():
    """Main deployment entry point"""
    import argparse
   
    parser = argparse.ArgumentParser(description="Deploy Options Calculator Pro")
    parser.add_argument(
        "targets",
        nargs="*",
        choices=["github", "pypi", "s3", "docker", "all"],
        default=["github"],
        help="Deployment targets"
    )
    parser.add_argument(
        "--version",
        help="Override version number"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Setup deployment configuration"
    )
    parser.add_argument(
        "--tag",
        action="store_true",
        help="Create git tag only"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate artifacts without deploying"
    )
   
    args = parser.parse_args()
   
    deployer = DeploymentManager()
   
    if args.version:
       deployer.version = args.version
   
    if args.config:
        deployer.setup_deployment_config()
        return
   
    if args.tag:
        success = deployer.create_git_tag()
        sys.exit(0 if success else 1)
   
    if args.dry_run:
        artifacts = deployer.validate_artifacts()
        print(f"✅ Found {len(artifacts)} valid artifacts")
        return
   
    # Expand "all" target
    targets = args.targets
    if "all" in targets:
        targets = ["github", "pypi", "docker"]
   
    success = deployer.run_deployment(targets)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
   main()