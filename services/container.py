"""
Dependency Injection Container for Options Calculator Pro.

Manages service lifecycles, handles dependency resolution, and provides
clean service instantiation without tight coupling.
"""

import logging
from typing import Dict, Type, TypeVar, Optional, Any, Callable
from threading import Lock
from enum import Enum

from services.interfaces import (
    IMarketDataService, IVolatilityService, IOptionsService,
    IMLService, IAsyncAPIService, IConfigService, ICacheService,
    OptionsCalculatorError, ConfigurationError
)

T = TypeVar('T')
logger = logging.getLogger(__name__)


class ServiceLifecycle(Enum):
    """Service lifecycle modes"""
    SINGLETON = "singleton"      # Single instance per container
    TRANSIENT = "transient"      # New instance per request
    SCOPED = "scoped"           # Single instance per scope


class ServiceRegistration:
    """Service registration information"""
    
    def __init__(
        self,
        interface: Type[T],
        implementation: Type[T],
        lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
        factory: Optional[Callable[..., T]] = None,
        dependencies: Optional[Dict[str, Type]] = None
    ):
        self.interface = interface
        self.implementation = implementation
        self.lifecycle = lifecycle
        self.factory = factory
        self.dependencies = dependencies or {}
        self.instance: Optional[T] = None


class ServiceContainer:
    """
    Dependency injection container for managing service dependencies.
    
    Provides clean service instantiation, lifecycle management, and
    dependency resolution without tight coupling between components.
    """
    
    def __init__(self):
        self._registrations: Dict[Type, ServiceRegistration] = {}
        self._instances: Dict[Type, Any] = {}
        self._lock = Lock()
        self.logger = logging.getLogger(__name__)
    
    def register(
        self,
        interface: Type[T],
        implementation: Type[T],
        lifecycle: ServiceLifecycle = ServiceLifecycle.SINGLETON,
        factory: Optional[Callable[..., T]] = None,
        dependencies: Optional[Dict[str, Type]] = None
    ) -> 'ServiceContainer':
        """
        Register a service with the container.
        
        Args:
            interface: Abstract interface type
            implementation: Concrete implementation type
            lifecycle: Service lifecycle management
            factory: Optional factory function for custom instantiation
            dependencies: Explicit dependency mapping for constructor
        
        Returns:
            Self for method chaining
        """
        with self._lock:
            registration = ServiceRegistration(
                interface=interface,
                implementation=implementation,
                lifecycle=lifecycle,
                factory=factory,
                dependencies=dependencies
            )
            
            self._registrations[interface] = registration
            self.logger.debug(f"Registered {interface.__name__} -> {implementation.__name__}")
            
        return self
    
    def register_instance(self, interface: Type[T], instance: T) -> 'ServiceContainer':
        """
        Register an existing instance as a singleton.
        
        Args:
            interface: Service interface type
            instance: Pre-created service instance
            
        Returns:
            Self for method chaining
        """
        with self._lock:
            registration = ServiceRegistration(
                interface=interface,
                implementation=type(instance),
                lifecycle=ServiceLifecycle.SINGLETON
            )
            registration.instance = instance
            
            self._registrations[interface] = registration
            self._instances[interface] = instance
            self.logger.debug(f"Registered instance {interface.__name__}")
            
        return self
    
    def get_service(self, service_type: Type[T]) -> T:
        """
        Resolve and return a service instance.
        
        Args:
            service_type: Type of service to resolve
            
        Returns:
            Service instance
            
        Raises:
            ConfigurationError: If service not registered or resolution fails
        """
        with self._lock:
            if service_type not in self._registrations:
                raise ConfigurationError(f"Service {service_type.__name__} not registered")
            
            registration = self._registrations[service_type]
            
            # Return singleton if already created
            if registration.lifecycle == ServiceLifecycle.SINGLETON and registration.instance:
                return registration.instance
            
            # Create new instance
            instance = self._create_instance(registration)
            
            # Store singleton instance
            if registration.lifecycle == ServiceLifecycle.SINGLETON:
                registration.instance = instance
                self._instances[service_type] = instance
            
            return instance
    
    def _create_instance(self, registration: ServiceRegistration) -> Any:
        """
        Create service instance with dependency resolution.
        
        Args:
            registration: Service registration info
            
        Returns:
            Created service instance
        """
        try:
            # Use factory if provided
            if registration.factory:
                dependencies = self._resolve_dependencies(registration.dependencies)
                return registration.factory(**dependencies)

            # Use explicit dependencies if provided, otherwise resolve from constructor
            if registration.dependencies:
                constructor_args = self._resolve_dependencies(registration.dependencies)
            else:
                constructor_args = self._resolve_constructor_dependencies(
                    registration.implementation
                )
            
            # Create instance
            instance = registration.implementation(**constructor_args)
            
            self.logger.debug(f"Created instance of {registration.implementation.__name__}")
            return instance
            
        except Exception as e:
            error_msg = f"Failed to create {registration.implementation.__name__}: {str(e)}"
            self.logger.error(error_msg)
            raise ConfigurationError(error_msg) from e
    
    def _resolve_constructor_dependencies(self, implementation: Type) -> Dict[str, Any]:
        """
        Resolve constructor dependencies through type hints.
        
        Args:
            implementation: Implementation class to analyze
            
        Returns:
            Dictionary of resolved dependencies
        """
        dependencies = {}
        
        # Get constructor type hints
        if hasattr(implementation.__init__, '__annotations__'):
            annotations = implementation.__init__.__annotations__
            
            for param_name, param_type in annotations.items():
                if param_name == 'return':  # Skip return type annotation
                    continue
                
                if param_type in self._registrations:
                    dependencies[param_name] = self.get_service(param_type)
                else:
                    self.logger.warning(
                        f"Cannot resolve dependency {param_name}: {param_type.__name__} "
                        f"for {implementation.__name__}"
                    )
        
        return dependencies
    
    def _resolve_dependencies(self, dependencies: Dict[str, Type]) -> Dict[str, Any]:
        """
        Resolve explicit dependency mapping.
        
        Args:
            dependencies: Dependency name -> type mapping
            
        Returns:
            Resolved dependency instances
        """
        resolved = {}
        
        for name, dep_type in dependencies.items():
            resolved[name] = self.get_service(dep_type)
        
        return resolved
    
    def is_registered(self, service_type: Type) -> bool:
        """Check if service type is registered"""
        with self._lock:
            return service_type in self._registrations
    
    def get_registrations(self) -> Dict[str, str]:
        """Get summary of all service registrations"""
        with self._lock:
            return {
                interface.__name__: registration.implementation.__name__
                for interface, registration in self._registrations.items()
            }
    
    def clear(self) -> None:
        """Clear all registrations and instances"""
        with self._lock:
            # Cleanup instances that have close/dispose methods
            for instance in self._instances.values():
                if hasattr(instance, 'close'):
                    try:
                        instance.close()
                    except Exception as e:
                        self.logger.warning(f"Error closing service: {e}")
                elif hasattr(instance, 'dispose'):
                    try:
                        instance.dispose()
                    except Exception as e:
                        self.logger.warning(f"Error disposing service: {e}")
            
            self._registrations.clear()
            self._instances.clear()
            self.logger.debug("Container cleared")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.clear()


def create_default_container() -> ServiceContainer:
    """
    Create container with default service registrations.
    
    This factory function sets up the standard service configuration
    for the options calculator application.
    
    Returns:
        Configured service container
    """
    from services.market_data import MarketDataService
    from services.volatility_service import VolatilityService
    from services.options_service import OptionsService
    from services.ml_service import MLService
    from services.async_api import AsyncMarketDataClient
    from services.cache_adapter import CacheServiceAdapter
    from utils.config_manager import ConfigManager
    from utils.ttl_cache import MultiTTLCache

    container = ServiceContainer()

    # Cache service (using professional adapter)
    container.register_instance(ICacheService, CacheServiceAdapter())
    
    # Config service
    container.register(
        IConfigService, 
        ConfigManager,
        lifecycle=ServiceLifecycle.SINGLETON
    )
    
    # Market data service
    container.register(
        IMarketDataService,
        MarketDataService,
        lifecycle=ServiceLifecycle.SINGLETON
        # No dependencies - MarketDataService.__init__(parent=None) takes no injected dependencies
    )
    
    # Volatility service
    container.register(
        IVolatilityService,
        VolatilityService,
        dependencies={
            'config_manager': IConfigService,
            'market_data_service': IMarketDataService
        }
    )
    
    # Options service
    container.register(
        IOptionsService,
        OptionsService,
        dependencies={
            'market_data': IMarketDataService,
            'volatility_service': IVolatilityService
        }
    )
    
    # ML service
    container.register(
        IMLService,
        MLService,
        dependencies={
            'market_data': IMarketDataService
        }
    )
    
    # Async API service
    container.register(
        IAsyncAPIService,
        AsyncMarketDataClient,
        dependencies={
            'config': IConfigService
        }
    )
    
    logger.info("Default service container created with all standard services")
    return container


# Global container instance for application use
_default_container: Optional[ServiceContainer] = None
_container_lock = Lock()


def get_container() -> ServiceContainer:
    """
    Get or create the global default container.
    
    Returns:
        Global service container instance
    """
    global _default_container
    
    with _container_lock:
        if _default_container is None:
            _default_container = create_default_container()
        
        return _default_container


def set_container(container: ServiceContainer) -> None:
    """
    Set the global default container.
    
    Args:
        container: Service container to use as default
    """
    global _default_container
    
    with _container_lock:
        if _default_container:
            _default_container.clear()
        
        _default_container = container
        logger.info("Global container updated")


def reset_container() -> None:
    """Reset the global container to default configuration"""
    global _default_container
    
    with _container_lock:
        if _default_container:
            _default_container.clear()
        
        _default_container = create_default_container()
        logger.info("Container reset to default configuration")