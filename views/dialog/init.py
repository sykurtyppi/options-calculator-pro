"""
Professional Options Calculator - Dialogs Module
Dialog windows for settings, about, and other modal interfaces
"""

from .settings_dialog import SettingsDialog
from .about_dialog import AboutDialog

__all__ = [
    'SettingsDialog',
    'AboutDialog'
]

__version__ = "1.0.0"