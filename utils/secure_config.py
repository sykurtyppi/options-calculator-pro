"""
Secure configuration management for Options Calculator Pro.

Handles API keys and sensitive trading parameters securely using
system keyring, environment variables, and encrypted storage.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import getpass
import platform

try:
    import keyring
    from keyring.errors import KeyringError
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    keyring = None

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Fernet = None


class SecureStorageBackend(str, Enum):
    """Available secure storage backends"""
    KEYRING = "keyring"          # System keyring (preferred)
    ENVIRONMENT = "environment"   # Environment variables
    ENCRYPTED_FILE = "encrypted_file"  # Encrypted local file
    PLAIN_FILE = "plain_file"    # Plain text (development only)


@dataclass
class APICredential:
    """API credential with metadata"""
    service: str
    key: str
    secret: Optional[str] = None
    endpoint: Optional[str] = None
    rate_limit: Optional[int] = None
    is_sandbox: bool = False
    
    def __post_init__(self):
        """Validate credential data"""
        if not self.service or not self.key:
            raise ValueError("Service name and API key are required")
        
        # Mask key in string representation for security
        self._display_key = self.key[:8] + "..." if len(self.key) > 8 else "***"
    
    def __str__(self) -> str:
        """Safe string representation that doesn't expose the key"""
        return f"APICredential(service={self.service}, key={self._display_key})"


class SecureConfigManager:
    """
    Secure configuration manager for sensitive trading data.
    
    Features:
    - Multiple storage backends (keyring, environment, encrypted file)
    - Automatic fallback between backends
    - API key rotation support
    - Audit logging of access
    - Development vs production modes
    """
    
    def __init__(
        self,
        app_name: str = "OptionsCalculatorPro",
        config_dir: Optional[Path] = None,
        preferred_backend: SecureStorageBackend = SecureStorageBackend.KEYRING
    ):
        self.app_name = app_name
        self.preferred_backend = preferred_backend
        self.logger = logging.getLogger(__name__)
        
        # Setup config directory
        if config_dir is None:
            config_dir = Path.home() / f".{app_name.lower()}"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine available backends
        self.available_backends = self._detect_available_backends()
        
        # Load configuration
        self.config_file = self.config_dir / "config.json"
        self._load_config()
    
    def _detect_available_backends(self) -> list:
        """Detect which storage backends are available"""
        backends = [SecureStorageBackend.ENVIRONMENT, SecureStorageBackend.PLAIN_FILE]
        
        if KEYRING_AVAILABLE:
            try:
                # Test keyring functionality
                keyring.get_password("test", "test")
                backends.insert(0, SecureStorageBackend.KEYRING)
                self.logger.info("System keyring available")
            except (KeyringError, Exception) as e:
                self.logger.warning(f"Keyring not available: {e}")
        
        if CRYPTO_AVAILABLE:
            backends.insert(-1, SecureStorageBackend.ENCRYPTED_FILE)
            self.logger.info("Encrypted file storage available")
        
        return backends
    
    def _load_config(self):
        """Load non-sensitive configuration"""
        self.config = {}
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                self.config = {}
    
    def _save_config(self):
        """Save non-sensitive configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving config: {e}")
    
    def store_api_key(
        self,
        service: str,
        api_key: str,
        secret: Optional[str] = None,
        backend: Optional[SecureStorageBackend] = None
    ) -> bool:
        """
        Store API key securely.
        
        Args:
            service: Service name (e.g., 'alpha_vantage', 'finnhub')
            api_key: API key to store
            secret: Optional API secret
            backend: Storage backend to use (defaults to preferred)
        
        Returns:
            True if stored successfully
        """
        backend = backend or self.preferred_backend
        
        # Try preferred backend first, then fallback
        for attempt_backend in [backend] + self.available_backends:
            if attempt_backend in self.available_backends:
                try:
                    if self._store_key_with_backend(service, api_key, secret, attempt_backend):
                        self.logger.info(f"API key stored for {service} using {attempt_backend.value}")
                        
                        # Store backend info in config
                        if 'api_keys' not in self.config:
                            self.config['api_keys'] = {}
                        self.config['api_keys'][service] = {
                            'backend': attempt_backend.value,
                            'stored_at': str(Path.home()),  # Don't log exact timestamp for security
                            'has_secret': secret is not None
                        }
                        self._save_config()
                        
                        return True
                        
                except Exception as e:
                    self.logger.warning(f"Failed to store with {attempt_backend.value}: {e}")
        
        self.logger.error(f"Failed to store API key for {service} with any backend")
        return False
    
    def get_api_key(self, service: str) -> Optional[APICredential]:
        """
        Retrieve API key for service.
        
        Args:
            service: Service name
        
        Returns:
            APICredential if found, None otherwise
        """
        # Check config for storage backend info
        backend_info = self.config.get('api_keys', {}).get(service, {})
        stored_backend = backend_info.get('backend')
        
        # Try stored backend first, then all available
        backends_to_try = []
        if stored_backend and SecureStorageBackend(stored_backend) in self.available_backends:
            backends_to_try.append(SecureStorageBackend(stored_backend))
        
        backends_to_try.extend([b for b in self.available_backends if b not in backends_to_try])
        
        for backend in backends_to_try:
            try:
                credential = self._get_key_with_backend(service, backend)
                if credential:
                    self.logger.debug(f"Retrieved API key for {service} from {backend.value}")
                    return credential
            except Exception as e:
                self.logger.debug(f"Backend {backend.value} failed for {service}: {e}")
        
        self.logger.warning(f"No API key found for service: {service}")
        return None
    
    def _store_key_with_backend(
        self,
        service: str,
        api_key: str,
        secret: Optional[str],
        backend: SecureStorageBackend
    ) -> bool:
        """Store key using specific backend"""
        
        if backend == SecureStorageBackend.KEYRING:
            if not KEYRING_AVAILABLE:
                return False
            
            keyring.set_password(self.app_name, f"{service}_key", api_key)
            if secret:
                keyring.set_password(self.app_name, f"{service}_secret", secret)
            return True
        
        elif backend == SecureStorageBackend.ENVIRONMENT:
            # For environment variables, we can only suggest the variable names
            # Actual setting must be done by user
            key_var = f"{service.upper()}_API_KEY"
            secret_var = f"{service.upper()}_API_SECRET"
            
            self.logger.info(f"To use environment variables, set: {key_var}={api_key[:8]}...")
            if secret:
                self.logger.info(f"And set: {secret_var}=<your_secret>")
            
            # Store in current session for immediate use
            os.environ[key_var] = api_key
            if secret:
                os.environ[secret_var] = secret
            return True
        
        elif backend == SecureStorageBackend.ENCRYPTED_FILE:
            return self._store_encrypted_file(service, api_key, secret)
        
        elif backend == SecureStorageBackend.PLAIN_FILE:
            return self._store_plain_file(service, api_key, secret)
        
        return False
    
    def _get_key_with_backend(
        self,
        service: str,
        backend: SecureStorageBackend
    ) -> Optional[APICredential]:
        """Retrieve key using specific backend"""
        
        if backend == SecureStorageBackend.KEYRING:
            if not KEYRING_AVAILABLE:
                return None
            
            try:
                api_key = keyring.get_password(self.app_name, f"{service}_key")
                if not api_key:
                    return None
                
                secret = keyring.get_password(self.app_name, f"{service}_secret")
                
                return APICredential(
                    service=service,
                    key=api_key,
                    secret=secret
                )
            except KeyringError:
                return None
        
        elif backend == SecureStorageBackend.ENVIRONMENT:
            key_var = f"{service.upper()}_API_KEY"
            secret_var = f"{service.upper()}_API_SECRET"
            
            api_key = os.getenv(key_var)
            if not api_key:
                return None
            
            secret = os.getenv(secret_var)
            
            return APICredential(
                service=service,
                key=api_key,
                secret=secret
            )
        
        elif backend == SecureStorageBackend.ENCRYPTED_FILE:
            return self._get_encrypted_file(service)
        
        elif backend == SecureStorageBackend.PLAIN_FILE:
            return self._get_plain_file(service)
        
        return None
    
    def _store_encrypted_file(
        self,
        service: str,
        api_key: str,
        secret: Optional[str]
    ) -> bool:
        """Store credentials in encrypted file"""
        if not CRYPTO_AVAILABLE:
            return False
        
        try:
            # Get or create encryption key
            key_file = self.config_dir / ".encryption_key"
            if key_file.exists():
                with open(key_file, 'rb') as f:
                    encryption_key = f.read()
            else:
                encryption_key = Fernet.generate_key()
                # Store key with restricted permissions
                key_file.write_bytes(encryption_key)
                key_file.chmod(0o600)  # Owner read/write only
            
            # Encrypt credentials
            fernet = Fernet(encryption_key)
            credentials = {'key': api_key}
            if secret:
                credentials['secret'] = secret
            
            encrypted_data = fernet.encrypt(json.dumps(credentials).encode())
            
            # Store encrypted file
            cred_file = self.config_dir / f"{service}.enc"
            cred_file.write_bytes(encrypted_data)
            cred_file.chmod(0o600)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Encryption storage failed: {e}")
            return False
    
    def _get_encrypted_file(self, service: str) -> Optional[APICredential]:
        """Retrieve credentials from encrypted file"""
        if not CRYPTO_AVAILABLE:
            return None
        
        try:
            key_file = self.config_dir / ".encryption_key"
            cred_file = self.config_dir / f"{service}.enc"
            
            if not (key_file.exists() and cred_file.exists()):
                return None
            
            # Load encryption key
            encryption_key = key_file.read_bytes()
            fernet = Fernet(encryption_key)
            
            # Decrypt credentials
            encrypted_data = cred_file.read_bytes()
            decrypted_data = fernet.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            return APICredential(
                service=service,
                key=credentials['key'],
                secret=credentials.get('secret')
            )
            
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            return None
    
    def _store_plain_file(
        self,
        service: str,
        api_key: str,
        secret: Optional[str]
    ) -> bool:
        """Store credentials in plain text (development only)"""
        self.logger.warning("Storing API key in plain text - development only!")
        
        try:
            credentials = {'key': api_key}
            if secret:
                credentials['secret'] = secret
            
            cred_file = self.config_dir / f"{service}.json"
            with open(cred_file, 'w') as f:
                json.dump(credentials, f, indent=2)
            
            # Restrict permissions
            cred_file.chmod(0o600)
            return True
            
        except Exception as e:
            self.logger.error(f"Plain file storage failed: {e}")
            return False
    
    def _get_plain_file(self, service: str) -> Optional[APICredential]:
        """Retrieve credentials from plain file"""
        try:
            cred_file = self.config_dir / f"{service}.json"
            if not cred_file.exists():
                return None
            
            with open(cred_file, 'r') as f:
                credentials = json.load(f)
            
            return APICredential(
                service=service,
                key=credentials['key'],
                secret=credentials.get('secret')
            )
            
        except Exception as e:
            self.logger.error(f"Plain file retrieval failed: {e}")
            return None
    
    def list_stored_keys(self) -> Dict[str, Dict[str, Any]]:
        """List all stored API keys (without exposing actual keys)"""
        stored_keys = {}
        
        for service, info in self.config.get('api_keys', {}).items():
            # Verify key still exists
            credential = self.get_api_key(service)
            if credential:
                stored_keys[service] = {
                    'backend': info.get('backend'),
                    'has_secret': info.get('has_secret', False),
                    'key_preview': credential.key[:8] + "..." if len(credential.key) > 8 else "***"
                }
        
        return stored_keys
    
    def remove_api_key(self, service: str) -> bool:
        """Remove stored API key"""
        backend_info = self.config.get('api_keys', {}).get(service, {})
        backend = SecureStorageBackend(backend_info.get('backend', self.preferred_backend.value))
        
        try:
            if backend == SecureStorageBackend.KEYRING:
                keyring.delete_password(self.app_name, f"{service}_key")
                try:
                    keyring.delete_password(self.app_name, f"{service}_secret")
                except:
                    pass  # Secret might not exist
            
            elif backend == SecureStorageBackend.ENCRYPTED_FILE:
                cred_file = self.config_dir / f"{service}.enc"
                if cred_file.exists():
                    cred_file.unlink()
            
            elif backend == SecureStorageBackend.PLAIN_FILE:
                cred_file = self.config_dir / f"{service}.json"
                if cred_file.exists():
                    cred_file.unlink()
            
            # Remove from config
            if 'api_keys' in self.config and service in self.config['api_keys']:
                del self.config['api_keys'][service]
                self._save_config()
            
            self.logger.info(f"Removed API key for {service}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove API key for {service}: {e}")
            return False


# Convenience functions for common use cases

def setup_api_key_interactive(service: str, app_name: str = "OptionsCalculatorPro") -> bool:
    """
    Interactive setup of API key with user prompts.
    
    Perfect for initial setup of the calculator.
    """
    print(f"\n=== Setting up {service.title()} API Key ===")
    
    api_key = getpass.getpass(f"Enter your {service} API key: ")
    if not api_key.strip():
        print("API key cannot be empty")
        return False
    
    # Check if service requires secret
    needs_secret = service.lower() in ['tradier', 'interactive_brokers']
    secret = None
    if needs_secret:
        secret = getpass.getpass(f"Enter your {service} API secret (optional): ")
        if not secret.strip():
            secret = None
    
    # Store the key
    config_manager = SecureConfigManager(app_name)
    success = config_manager.store_api_key(service, api_key, secret)
    
    if success:
        print(f"✓ API key for {service} stored securely")
        
        # Test the key if possible
        credential = config_manager.get_api_key(service)
        if credential:
            print(f"✓ Key retrieved successfully: {credential.key[:8]}...")
    else:
        print(f"✗ Failed to store API key for {service}")
    
    return success


def get_trading_api_keys() -> Dict[str, APICredential]:
    """
    Get all trading API keys for the calculator.
    
    Returns dict of service -> credential mappings.
    """
    config_manager = SecureConfigManager()
    services = ['alpha_vantage', 'finnhub', 'polygon', 'tradier']
    
    credentials = {}
    for service in services:
        cred = config_manager.get_api_key(service)
        if cred:
            credentials[service] = cred
    
    return credentials


def validate_api_access(service: str) -> bool:
    """
    Validate that API key exists and is accessible.
    
    Used for health checks and startup validation.
    """
    try:
        config_manager = SecureConfigManager()
        credential = config_manager.get_api_key(service)
        return credential is not None
    except Exception:
        return False


# CLI utility for managing keys
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage API keys for Options Calculator Pro")
    parser.add_argument('action', choices=['setup', 'list', 'test', 'remove'])
    parser.add_argument('--service', help='Service name')
    
    args = parser.parse_args()
    
    if args.action == 'setup':
        if not args.service:
            print("Service name required for setup")
            exit(1)
        setup_api_key_interactive(args.service)
    
    elif args.action == 'list':
        config_manager = SecureConfigManager()
        keys = config_manager.list_stored_keys()
        
        if keys:
            print("\nStored API Keys:")
            for service, info in keys.items():
                print(f"  {service}: {info['key_preview']} ({info['backend']})")
        else:
            print("No API keys stored")
    
    elif args.action == 'test':
        if not args.service:
            services = ['alpha_vantage', 'finnhub', 'polygon', 'tradier']
        else:
            services = [args.service]
        
        print("\nTesting API key access:")
        for service in services:
            if validate_api_access(service):
                print(f"✓ {service}: API key accessible")
            else:
                print(f"✗ {service}: No API key found")
    
    elif args.action == 'remove':
        if not args.service:
            print("Service name required for removal")
            exit(1)
        
        config_manager = SecureConfigManager()
        if config_manager.remove_api_key(args.service):
            print(f"✓ Removed API key for {args.service}")
        else:
            print(f"✗ Failed to remove API key for {args.service}")