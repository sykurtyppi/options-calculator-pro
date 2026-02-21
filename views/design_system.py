"""
Shared design-system tokens and UI style helpers for desktop views.

This module centralizes palette, spacing, typography, and component states
so screen-level views stay visually consistent.
"""

from PySide6.QtGui import QFont

# Typography tokens
FONT_FAMILY = "Segoe UI"
FONT_MONO = "Consolas"

# Spacing tokens
SPACE_XS = 6
SPACE_SM = 10
SPACE_MD = 14
SPACE_LG = 20

# Radius tokens
RADIUS_SM = 4
RADIUS_MD = 6
RADIUS_LG = 10

# Color tokens
COLOR_BG_APP = "#030712"
COLOR_BG_PANEL = "#111827"
COLOR_BG_PANEL_ALT = "#0f172a"
COLOR_BORDER = "#1e293b"
COLOR_BORDER_SOFT = "#334155"
COLOR_TEXT = "#e2e8f0"
COLOR_TEXT_MUTED = "#94a3b8"
COLOR_TEXT_SUBTLE = "#64748b"
COLOR_ACCENT = "#0ea5e9"
COLOR_ACCENT_HOVER = "#0284c7"
COLOR_SUCCESS = "#10b981"
COLOR_WARN = "#f59e0b"
COLOR_ERROR = "#ef4444"


def app_font(size: int = 11, bold: bool = False, mono: bool = False) -> QFont:
    """Return shared app font."""
    family = FONT_MONO if mono else FONT_FAMILY
    weight = QFont.Weight.Bold if bold else QFont.Weight.Normal
    return QFont(family, size, weight)


def panel_style() -> str:
    """Standard panel style."""
    return f"""
        QFrame {{
            background-color: {COLOR_BG_PANEL};
            border: 1px solid {COLOR_BORDER};
            border-radius: {RADIUS_MD}px;
        }}
    """


def status_strip_style(level: str = "info") -> str:
    """Status strip style by level."""
    palette = {
        "info": ("#111827", "#1f2937", "#93c5fd"),
        "success": ("#052e16", "#14532d", "#86efac"),
        "warn": ("#422006", "#78350f", "#fcd34d"),
        "error": ("#3f0b13", "#7f1d1d", "#fca5a5"),
        "neutral": ("#0b1220", "#1f2937", COLOR_TEXT_MUTED),
    }
    bg, border, fg = palette.get(level, palette["info"])
    return f"""
        QLabel {{
            background-color: {bg};
            border: 1px solid {border};
            border-radius: {RADIUS_MD}px;
            color: {fg};
            padding: 8px 12px;
            font-weight: bold;
        }}
    """


def summary_strip_style() -> str:
    """Summary strip style."""
    return f"""
        QLabel {{
            color: {COLOR_TEXT_MUTED};
            background-color: {COLOR_BG_PANEL_ALT};
            border: 1px solid {COLOR_BORDER};
            border-radius: {RADIUS_MD}px;
            padding: 7px 10px;
        }}
    """


def button_style(kind: str = "secondary") -> str:
    """Button styles for primary/secondary/tertiary actions."""
    styles = {
        "primary": f"""
            QPushButton {{
                background-color: {COLOR_ACCENT};
                border: none;
                border-radius: {RADIUS_MD}px;
                color: #ffffff;
                font-weight: bold;
                padding: 8px 16px;
            }}
            QPushButton:hover {{ background-color: {COLOR_ACCENT_HOVER}; }}
            QPushButton:pressed {{ background-color: #0369a1; }}
            QPushButton:disabled {{ background-color: {COLOR_BORDER_SOFT}; color: {COLOR_TEXT_MUTED}; }}
        """,
        "secondary": f"""
            QPushButton {{
                background-color: #1f2937;
                border: 1px solid {COLOR_BORDER_SOFT};
                border-radius: {RADIUS_MD}px;
                color: #cbd5e1;
                font-weight: bold;
                padding: 8px 14px;
            }}
            QPushButton:hover {{ background-color: {COLOR_BORDER_SOFT}; }}
            QPushButton:disabled {{ color: {COLOR_TEXT_SUBTLE}; border-color: #1f2937; }}
        """,
        "tertiary": f"""
            QPushButton {{
                background-color: {COLOR_BG_PANEL};
                border: 1px solid {COLOR_BORDER_SOFT};
                border-radius: {RADIUS_MD}px;
                color: #cbd5e1;
                font-weight: bold;
                padding: 8px 14px;
            }}
            QPushButton:hover {{ background-color: #1f2937; }}
        """,
    }
    return styles.get(kind, styles["secondary"])


def input_style(error: bool = False) -> str:
    """Shared input style."""
    border_color = COLOR_ERROR if error else COLOR_BORDER_SOFT
    return f"""
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {COLOR_BG_PANEL};
            border: 1px solid {border_color};
            border-radius: {RADIUS_SM}px;
            padding: 6px 8px;
            color: {COLOR_TEXT};
            font-size: 11px;
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {COLOR_ACCENT};
        }}
    """


def table_style() -> str:
    """Shared table style."""
    return f"""
        QTableWidget {{
            background-color: {COLOR_BG_PANEL};
            border: 1px solid {COLOR_BORDER};
            gridline-color: {COLOR_BORDER};
            color: {COLOR_TEXT};
        }}
        QTableWidget::item {{
            padding: 8px;
            border-bottom: 1px solid {COLOR_BORDER};
        }}
        QTableWidget::item:selected {{
            background-color: {COLOR_ACCENT};
            color: #ffffff;
        }}
        QHeaderView::section {{
            background-color: #1f2937;
            color: {COLOR_TEXT};
            padding: 8px;
            border: none;
            font-weight: bold;
        }}
    """


def mono_text_style() -> str:
    """Shared monospaced text area style."""
    return f"""
        QTextEdit {{
            background-color: #111111;
            border: 1px solid {COLOR_BORDER};
            color: #cbd5e1;
            font-family: '{FONT_MONO}', monospace;
            font-size: 11px;
            padding: 8px;
        }}
    """


def app_theme_stylesheet() -> str:
    """Global application theme stylesheet."""
    return f"""
        QWidget {{
            background-color: {COLOR_BG_APP};
            color: {COLOR_TEXT};
            font-family: '{FONT_FAMILY}', 'Helvetica', sans-serif;
            font-size: 12px;
        }}

        QScrollArea {{
            border: none;
            background-color: transparent;
        }}

        QScrollBar:vertical {{
            background-color: {COLOR_BG_PANEL};
            width: 12px;
            border-radius: 6px;
            margin: 2px;
        }}

        QScrollBar::handle:vertical {{
            background-color: {COLOR_BORDER_SOFT};
            border-radius: 5px;
            min-height: 20px;
            margin: 1px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: #475569;
        }}

        QScrollBar:horizontal {{
            background-color: {COLOR_BG_PANEL};
            height: 12px;
            border-radius: 6px;
            margin: 2px;
        }}

        QScrollBar::handle:horizontal {{
            background-color: {COLOR_BORDER_SOFT};
            border-radius: 5px;
            min-width: 20px;
            margin: 1px;
        }}

        QScrollBar::handle:horizontal:hover {{
            background-color: #475569;
        }}

        QLabel {{
            color: {COLOR_TEXT};
        }}

        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {COLOR_BG_PANEL};
            border: 1px solid {COLOR_BORDER_SOFT};
            border-radius: {RADIUS_SM}px;
            color: {COLOR_TEXT};
            padding: 6px 8px;
            font-size: 11px;
        }}

        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border-color: {COLOR_ACCENT};
        }}

        QToolTip {{
            background-color: {COLOR_BG_PANEL};
            color: {COLOR_TEXT};
            border: 1px solid {COLOR_BORDER_SOFT};
            border-radius: {RADIUS_SM}px;
            padding: 6px;
            font-size: 11px;
        }}
    """
