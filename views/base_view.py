"""
Base view classes for the Professional Options Calculator
PySide6 implementation with professional styling
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QLabel, QPushButton, QLineEdit, QTextEdit, QProgressBar,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QScrollArea, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QMessageBox, QFileDialog,
    QApplication, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, Signal
from PySide6.QtGui import QFont, QPalette, QIcon, QPixmap, QAction
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseView(QWidget):
    """Base class for all views with common functionality"""
    
    # Common signals
    status_changed = Signal(str)
    error_occurred = Signal(str, str)  # title, message
    data_requested = Signal(dict)  # request parameters
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_styling()
        self.setup_ui()
        self.connect_signals()
        
    def setup_styling(self):
        """Setup professional styling"""
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', 'San Francisco', Arial, sans-serif;
                font-size: 10pt;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4a4a4a;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #60a3d9;
            }
            
            QPushButton {
                background-color: #0d7377;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 80px;
            }
            
            QPushButton:hover {
                background-color: #14a085;
            }
            
            QPushButton:pressed {
                background-color: #0a5d61;
            }
            
            QPushButton:disabled {
                background-color: #4a4a4a;
                color: #888888;
            }
            
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #404040;
                border: 2px solid #5a5a5a;
                border-radius: 4px;
                padding: 6px;
                min-height: 20px;
            }
            
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #60a3d9;
            }
            
            QTextEdit {
                background-color: #353535;
                border: 2px solid #5a5a5a;
                border-radius: 4px;
                padding: 8px;
            }
            
            QProgressBar {
                border: 2px solid #5a5a5a;
                border-radius: 4px;
                text-align: center;
                background-color: #404040;
            }
            
            QProgressBar::chunk {
                background-color: #0d7377;
                border-radius: 2px;
            }
            
            QTableWidget {
                gridline-color: #5a5a5a;
                background-color: #353535;
                alternate-background-color: #404040;
                selection-background-color: #60a3d9;
            }
            
            QHeaderView::section {
                background-color: #4a4a4a;
                padding: 8px;
                border: 1px solid #5a5a5a;
                font-weight: bold;
            }
            
            QTabWidget::pane {
                border: 2px solid #4a4a4a;
                border-radius: 4px;
            }
            
            QTabBar::tab {
                background-color: #404040;
                padding: 8px 16px;
                margin-right: 2px;
                margin-bottom: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: #60a3d9;
                color: #ffffff;
            }
            
            QTabBar::tab:hover {
                background-color: #5a5a5a;
            }
            
            QScrollBar:vertical {
                background-color: #404040;
                width: 16px;
                border-radius: 8px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #60a3d9;
                border-radius: 8px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #7bb3e0;
            }
        """)
    
    def setup_ui(self):
        """Setup UI - to be implemented by subclasses"""
        pass
    
    def connect_signals(self):
        """Connect signals - to be implemented by subclasses"""
        pass
    
    def emit_status(self, message: str):
        """Emit status update"""
        self.status_changed.emit(message)
        logger.info(f"View status: {message}")
    
    def emit_error(self, title: str, message: str):
        """Emit error signal"""
        self.error_occurred.emit(title, message)
        logger.error(f"View error - {title}: {message}")
    
    def show_message(self, title: str, message: str, msg_type: str = "info"):
        """Show message box"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        
        if msg_type == "info":
            msg_box.setIcon(QMessageBox.Information)
        elif msg_type == "warning":
            msg_box.setIcon(QMessageBox.Warning)
        elif msg_type == "error":
            msg_box.setIcon(QMessageBox.Critical)
        elif msg_type == "question":
            msg_box.setIcon(QMessageBox.Question)
            msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        
        return msg_box.exec()
    
    def create_section_header(self, title: str) -> QLabel:
        """Create a styled section header"""
        header = QLabel(title)
        header.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #60a3d9;
                padding: 10px 0px;
                border-bottom: 2px solid #4a4a4a;
                margin-bottom: 10px;
            }
        """)
        return header
    
    def create_info_label(self, text: str, color: str = "#cccccc") -> QLabel:
        """Create an info label with custom styling"""
        label = QLabel(text)
        label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 9pt;
                padding: 4px;
            }}
        """)
        label.setWordWrap(True)
        return label
    
    def create_metric_display(self, label_text: str, value_text: str, 
                            value_color: str = "#ffffff") -> QHBoxLayout:
        """Create a metric display with label and value"""
        layout = QHBoxLayout()
        
        label = QLabel(f"{label_text}:")
        label.setStyleSheet("font-weight: bold; color: #cccccc;")
        
        value = QLabel(value_text)
        value.setStyleSheet(f"color: {value_color}; font-weight: bold;")
        
        layout.addWidget(label)
        layout.addWidget(value)
        layout.addStretch()
        
        return layout
    
    def create_loading_widget(self) -> QWidget:
        """Create a loading indicator widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 0)  # Indeterminate progress
        progress.setTextVisible(False)
        
        # Loading label
        label = QLabel("Loading...")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: #60a3d9; font-weight: bold;")
        
        layout.addWidget(progress)
        layout.addWidget(label)
        
        return widget


class ProfessionalTable(QTableWidget):
    """Professional styled table widget with enhanced functionality"""
    
    row_selected = Signal(int)
    data_changed = Signal()
    
    def __init__(self, headers: List[str], parent=None):
        super().__init__(0, len(headers), parent)
        self.headers = headers
        self.setup_table()
        
    def setup_table(self):
        """Setup table appearance and behavior"""
        # Set headers
        self.setHorizontalHeaderLabels(self.headers)
        
        # Configure table behavior
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSortingEnabled(True)
        
        # Configure header
        header = self.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Interactive)
        
        # Style the table
        self.setStyleSheet("""
            QTableWidget {
                gridline-color: #4a4a4a;
                background-color: #353535;
                alternate-background-color: #3a3a3a;
                selection-background-color: #60a3d9;
                selection-color: #ffffff;
            }
            
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #4a4a4a;
            }
            
            QTableWidget::item:selected {
                background-color: #60a3d9;
            }
            
            QHeaderView::section {
                background-color: #4a4a4a;
                padding: 10px;
                border: none;
                border-right: 1px solid #5a5a5a;
                font-weight: bold;
                color: #ffffff;
            }
        """)
        
        # Connect signals
        self.itemSelectionChanged.connect(self._on_selection_changed)
    
    def add_row(self, data: List[str], row_data: Dict[str, Any] = None, 
                row_color: str = None):
        """Add a row to the table"""
        row_position = self.rowCount()
        self.insertRow(row_position)
        
        for col, value in enumerate(data):
            item = QTableWidgetItem(str(value))
            
            # Set row color if specified
            if row_color:
                item.setBackground(QColor(row_color))
            
            # Store additional data
            if row_data:
                item.setData(Qt.UserRole, row_data)
            
            self.setItem(row_position, col, item)
    
    def clear_table(self):
        """Clear all table data"""
        self.setRowCount(0)
        self.data_changed.emit()
    
    def get_selected_row_data(self) -> Optional[Dict[str, Any]]:
        """Get data from selected row"""
        current_row = self.currentRow()
        if current_row >= 0:
            item = self.item(current_row, 0)
            if item:
                return item.data(Qt.UserRole)
        return None
    
    def _on_selection_changed(self):
        """Handle selection change"""
        current_row = self.currentRow()
        if current_row >= 0:
            self.row_selected.emit(current_row)


class StatusBar(QWidget):
    """Professional status bar with progress indicator"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup status bar UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #cccccc;
                font-size: 9pt;
                padding: 4px;
            }
        """)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #5a5a5a;
                border-radius: 3px;
                text-align: center;
                background-color: #404040;
                color: #ffffff;
                font-size: 8pt;
            }
            
            QProgressBar::chunk {
                background-color: #0d7377;
                border-radius: 2px;
            }
        """)
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.progress_bar)
        
    def set_status(self, message: str):
        """Set status message"""
        self.status_label.setText(message)
        
    def show_progress(self, show: bool = True):
        """Show or hide progress bar"""
        self.progress_bar.setVisible(show)
        
    def set_progress(self, value: int):
        """Set progress value (0-100)"""
        self.progress_bar.setValue(value)
        if not self.progress_bar.isVisible():
            self.show_progress(True)


class CollapsibleGroupBox(QWidget):
    """Collapsible group box widget"""
    
    toggled = Signal(bool)
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.is_collapsed = False
        self.setup_ui()
        
    def setup_ui(self):
        """Setup collapsible group UI"""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header button
        self.header_button = QPushButton(f"▼ {self.title}")
        self.header_button.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 8px;
                font-weight: bold;
                background-color: #4a4a4a;
                border: none;
                border-radius: 4px;
            }
            
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)
        self.header_button.clicked.connect(self.toggle_collapsed)
        
        # Content frame
        self.content_frame = QFrame()
        self.content_frame.setFrameStyle(QFrame.Box)
        self.content_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                margin-top: 2px;
            }
        """)
        
        # Content layout
        self.content_layout = QVBoxLayout(self.content_frame)
        
        self.main_layout.addWidget(self.header_button)
        self.main_layout.addWidget(self.content_frame)
        
    def toggle_collapsed(self):
        """Toggle collapsed state"""
        self.is_collapsed = not self.is_collapsed
        self.content_frame.setVisible(not self.is_collapsed)
        
        # Update button text
        arrow = "▶" if self.is_collapsed else "▼"
        self.header_button.setText(f"{arrow} {self.title}")
        
        self.toggled.emit(self.is_collapsed)
    
    def add_widget(self, widget: QWidget):
        """Add widget to content area"""
        self.content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        """Add layout to content area"""
        self.content_layout.addLayout(layout)


class MetricsDisplayWidget(QWidget):
    """Widget for displaying analysis metrics in a professional layout"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.metrics = {}
        self.setup_ui()
        
    def setup_ui(self):
        """Setup metrics display UI"""
        self.layout = QGridLayout(self)
        self.layout.setSpacing(10)
        
        # Title
        title = QLabel("Analysis Metrics")
        title.setStyleSheet("""
            QLabel {
                font-size: 12pt;
                font-weight: bold;
                color: #60a3d9;
                padding: 5px 0px;
            }
        """)
        self.layout.addWidget(title, 0, 0, 1, 4)
        
        self.current_row = 1
        
    def add_metric(self, label: str, value: str, color: str = "#ffffff", 
                   tooltip: str = ""):
        """Add a metric to the display"""
        # Label
        label_widget = QLabel(f"{label}:")
        label_widget.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #cccccc;
                padding: 4px;
            }
        """)
        
        # Value
        value_widget = QLabel(value)
        value_widget.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-weight: bold;
                padding: 4px;
            }}
        """)
        
        if tooltip:
            label_widget.setToolTip(tooltip)
            value_widget.setToolTip(tooltip)
        
        # Add to grid (2 columns per row)
        col = (len(self.metrics) % 2) * 2
        row = self.current_row + (len(self.metrics) // 2)
        
        self.layout.addWidget(label_widget, row, col)
        self.layout.addWidget(value_widget, row, col + 1)
        
        self.metrics[label] = value_widget
        
    def update_metric(self, label: str, value: str, color: str = None):
        """Update an existing metric"""
        if label in self.metrics:
            self.metrics[label].setText(value)
            if color:
                current_style = self.metrics[label].styleSheet()
                # Update color in stylesheet
                new_style = current_style.replace(
                    current_style.split('color: ')[1].split(';')[0], 
                    color
                )
                self.metrics[label].setStyleSheet(new_style)
    
    def clear_metrics(self):
        """Clear all metrics"""
        for widget in self.metrics.values():
            widget.deleteLater()
        self.metrics.clear()
        self.current_row = 1


class LoadingOverlay(QWidget):
    """Loading overlay widget"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup loading overlay UI"""
        layout = QVBoxLayout(self)
        
        # Semi-transparent background
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 0, 0, 150);
            }
        """)
        
        # Center content
        center_widget = QWidget()
        center_widget.setFixedSize(200, 100)
        center_widget.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 2px solid #60a3d9;
                border-radius: 10px;
            }
        """)
        
        center_layout = QVBoxLayout(center_widget)
        
        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 0)  # Indeterminate
        progress.setTextVisible(False)
        
        # Loading text
        text = QLabel("Loading...")
        text.setAlignment(Qt.AlignCenter)
        text.setStyleSheet("""
            QLabel {
                color: #60a3d9;
                font-weight: bold;
                font-size: 12pt;
            }
        """)
        
        center_layout.addWidget(progress)
        center_layout.addWidget(text)
        
        layout.addWidget(center_widget, 0, Qt.AlignCenter)
        
    def show_overlay(self, parent_widget: QWidget):
        """Show overlay on parent widget"""
        self.setParent(parent_widget)
        self.resize(parent_widget.size())
        self.move(0, 0)
        self.show()
        self.raise_()
        
    def hide_overlay(self):
        """Hide overlay"""
        self.hide()
        self.setParent(None)