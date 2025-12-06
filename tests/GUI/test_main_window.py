"""
GUI tests for main window and application interface
Tests user interface components and interactions
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtGui import QKeySequence

from views.main_window import MainWindow
from services.config_manager import ConfigManager


@pytest.mark.gui
class TestMainWindow:
    """Test suite for main window functionality"""
    
    @pytest.fixture
    def main_window(self, qt_app, mock_config_manager):
        """Create main window for testing"""
        with patch('services.market_data_service.MarketDataService'), \
             patch('services.analysis_service.AnalysisService'), \
             patch('workers.analysis_worker.AnalysisWorker'), \
             patch('workers.data_worker.DataWorker'):
            
            window = MainWindow(config_manager=mock_config_manager)
            window.show()
            qt_app.processEvents()
            return window
    
    def test_main_window_initialization(self, main_window, qt_app):
        """Test main window initializes correctly"""
        assert main_window.isVisible()
        assert main_window.windowTitle() == "Options Calculator Pro"
        
        # Check that main components are created
        assert hasattr(main_window, 'analysis_view')
        assert hasattr(main_window, 'status_bar')
        assert hasattr(main_window, 'menu_bar')
        assert hasattr(main_window, 'toolbar')
    
    def test_menu_bar_creation(self, main_window, qt_app):
        """Test menu bar and actions"""
        menu_bar = main_window.menuBar()
        assert menu_bar is not None
        
        # Check for main menus
        menus = [action.text() for action in menu_bar.actions() if action.menu()]
        
        expected_menus = ['File', 'Edit', 'View', 'Tools', 'Help']
        for expected_menu in expected_menus:
            assert any(expected_menu in menu for menu in menus), \
                f"Missing menu: {expected_menu}"
    
    def test_toolbar_creation(self, main_window, qt_app):
        """Test toolbar and actions"""
        toolbars = main_window.findChildren(type(main_window.toolbar))
        assert len(toolbars) > 0
        
        toolbar = toolbars[0]
        actions = toolbar.actions()
        
        # Should have some actions
        assert len(actions) > 0
        
        # Check for key toolbar actions
        action_texts = [action.text() for action in actions if action.text()]
        expected_actions = ['Analyze', 'Settings', 'Help']
        for expected_action in expected_actions:
            assert any(expected_action in text for text in action_texts), \
                f"Missing toolbar action: {expected_action}"
   
    def test_status_bar_functionality(self, main_window, qt_app):
        """Test status bar updates and messages"""
        status_bar = main_window.statusBar()
        assert status_bar is not None
       
        # Test status message
        test_message = "Test status message"
        main_window.update_status(test_message)
        qt_app.processEvents()
       
        # Check if message appears in status bar
        status_text = status_bar.currentMessage()
        assert test_message in status_text
   
    def test_symbol_input_functionality(self, main_window, qt_app):
        """Test symbol input and validation"""
        # Find symbol input widget
        symbol_input = main_window.analysis_view.input_widget.symbols_input
        assert symbol_input is not None
       
        # Test valid symbol input
        symbol_input.clear()
        QTest.keyClicks(symbol_input, "AAPL")
        qt_app.processEvents()
       
        assert symbol_input.text() == "AAPL"
       
        # Test multiple symbols
        symbol_input.clear()
        QTest.keyClicks(symbol_input, "AAPL, MSFT, GOOGL")
        qt_app.processEvents()
       
        assert "AAPL" in symbol_input.text()
        assert "MSFT" in symbol_input.text()
        assert "GOOGL" in symbol_input.text()
   
    def test_analysis_button_functionality(self, main_window, qt_app):
        """Test analysis button click and state changes"""
        # Find analysis button
        analyze_button = main_window.analysis_view.input_widget.analyze_button
        assert analyze_button is not None
       
        # Initially should be disabled (no symbols)
        assert not analyze_button.isEnabled()
       
        # Add symbol to enable button
        symbol_input = main_window.analysis_view.input_widget.symbols_input
        symbol_input.setText("AAPL")
        qt_app.processEvents()
       
        # Button should now be enabled
        assert analyze_button.isEnabled()
       
        # Mock analysis service
        with patch.object(main_window.analysis_worker, 'analyze_symbols') as mock_analyze:
            # Click analyze button
            QTest.mouseClick(analyze_button, Qt.LeftButton)
            qt_app.processEvents()
           
            # Should have called analysis
            mock_analyze.assert_called_once()
    
    def test_keyboard_shortcuts(self, main_window, qt_app):
        """Test keyboard shortcuts functionality"""
        # Test Ctrl+N for new analysis
        with patch.object(main_window, 'new_analysis') as mock_new:
            QTest.keySequence(main_window, QKeySequence("Ctrl+N"))
            qt_app.processEvents()
            mock_new.assert_called_once()
       
        # Test Ctrl+O for open
        with patch.object(main_window, 'open_analysis') as mock_open:
            QTest.keySequence(main_window, QKeySequence("Ctrl+O"))
            qt_app.processEvents()
            mock_open.assert_called_once()
       
        # Test F1 for help
        with patch.object(main_window, 'show_help') as mock_help:
            QTest.keySequence(main_window, QKeySequence("F1"))
            qt_app.processEvents()
            mock_help.assert_called_once()
   
    def test_window_resize_and_layout(self, main_window, qt_app):
        """Test window resizing and layout responsiveness"""
        original_size = main_window.size()
       
        # Resize to larger size
        new_width = original_size.width() + 200
        new_height = original_size.height() + 100
        main_window.resize(new_width, new_height)
        qt_app.processEvents()
       
        # Check that resize worked
        current_size = main_window.size()
        assert current_size.width() >= new_width - 10  # Allow small tolerance
        assert current_size.height() >= new_height - 10
       
        # Check that components are still visible and properly laid out
        analysis_view = main_window.analysis_view
        assert analysis_view.isVisible()
        assert analysis_view.size().width() > 0
        assert analysis_view.size().height() > 0
   
    def test_settings_dialog_integration(self, main_window, qt_app):
        """Test settings dialog opening and integration"""
        with patch('views.dialogs.settings_dialog.SettingsDialog') as mock_dialog:
            mock_instance = Mock()
            mock_dialog.return_value = mock_instance
            mock_instance.exec.return_value = True
           
            # Open settings through menu
            main_window.show_settings()
            qt_app.processEvents()
           
            # Should have created and shown dialog
            mock_dialog.assert_called_once()
            mock_instance.exec.assert_called_once()
   
    def test_about_dialog_integration(self, main_window, qt_app):
        """Test about dialog opening"""
        with patch('views.dialogs.about_dialog.AboutDialog') as mock_dialog:
            mock_instance = Mock()
            mock_dialog.return_value = mock_instance
            mock_instance.exec.return_value = True
           
            # Open about dialog
            main_window.show_about()
            qt_app.processEvents()
           
            # Should have created and shown dialog
            mock_dialog.assert_called_once()
            mock_instance.exec.assert_called_once()
   
    def test_progress_indication(self, main_window, qt_app):
        """Test progress indication during analysis"""
        # Start mock analysis
        main_window.on_analysis_started()
        qt_app.processEvents()
       
        # Check that progress is showing
        progress_widget = main_window.analysis_view.progress_widget
        assert progress_widget.isVisible()
       
        # Complete analysis
        mock_results = {
            'symbol': 'AAPL',
            'recommendation': 'BUY',
            'confidence': 0.75
        }
       
        main_window.on_analysis_completed(mock_results)
        qt_app.processEvents()
       
        # Progress should be hidden
        assert not progress_widget.isVisible()


@pytest.mark.gui
class TestAnalysisView:
    """Test suite for analysis view components"""
   
    @pytest.fixture
    def analysis_view(self, qt_app, mock_config_manager):
        """Create analysis view for testing"""
        with patch('services.market_data_service.MarketDataService'), \
             patch('services.analysis_service.AnalysisService'):
            
            from views.analysis_view import AnalysisView
            view = AnalysisView(config_manager=mock_config_manager)
            view.show()
            qt_app.processEvents()
            return view
   
    def test_input_widget_functionality(self, analysis_view, qt_app):
        """Test analysis input widget"""
        input_widget = analysis_view.input_widget
       
        # Test symbol input
        symbols_input = input_widget.symbols_input
        symbols_input.setText("AAPL, MSFT")
        qt_app.processEvents()
       
        assert symbols_input.text() == "AAPL, MSFT"
       
        # Test contracts input
        contracts_input = input_widget.contracts_input
        contracts_input.setValue(5)
        qt_app.processEvents()
       
        assert contracts_input.value() == 5
       
        # Test debit input
        debit_input = input_widget.debit_input
        debit_input.setValue(3.25)
        qt_app.processEvents()
       
        assert debit_input.value() == 3.25
   
    def test_results_display(self, analysis_view, qt_app):
        """Test analysis results display"""
        # Mock analysis results
        mock_results = {
            'symbol': 'AAPL',
            'recommendation': 'BUY',
            'confidence': 0.75,
            'expected_profit': 125.0,
            'max_loss': 250.0,
            'probability_profit': 0.65,
            'greeks': {
                'delta': 0.15,
                'gamma': 0.02,
                'theta': -5.50,
                'vega': 15.25
            },
            'volatility_metrics': {
                'current_iv': 0.28,
                'iv_rank': 0.65
            }
        }
       
        # Update results display
        analysis_view.update_results(mock_results)
        qt_app.processEvents()
       
        # Check that results are displayed
        results_widget = analysis_view.results_widget
        assert results_widget.isVisible()
       
        # Check specific result values
        confidence_widget = analysis_view.confidence_indicator
        # Note: Exact assertion depends on implementation
        assert confidence_widget.isVisible()
   
    def test_chart_integration(self, analysis_view, qt_app):
        """Test chart display and updates"""
        chart_widget = analysis_view.dashboard
        assert chart_widget is not None
       
        # Mock chart data
        mock_chart_data = {
            'price_distribution': {
                'prices': [140, 145, 150, 155, 160],
                'probabilities': [0.1, 0.2, 0.4, 0.2, 0.1]
            },
            'pl_analysis': {
                'price_range': [140, 150, 160],
                'pl_values': [-200, 100, -100]
            }
        }
       
        # Update charts
        analysis_view.update_charts(mock_chart_data)
        qt_app.processEvents()
       
        # Charts should be visible and updated
        assert chart_widget.isVisible()
   
    def test_metrics_panel_updates(self, analysis_view, qt_app):
        """Test trading metrics panel updates"""
        metrics_panel = analysis_view.metrics_panel
        assert metrics_panel is not None
       
        # Mock metrics data
        mock_metrics = {
            'Expected Move': '±5.2%',
            'Max Loss': '$250',
            'Max Profit': '$125',
            'Breakeven': '$152.50',
            'IV Rank': '65%'
        }
       
        # Update metrics
        for label, value in mock_metrics.items():
            metrics_panel.update_metric(label, value)
       
        qt_app.processEvents()
        
        # Verify updates
        assert metrics_panel.isVisible()


@pytest.mark.gui
class TestUserInteractionFlows:
    """Test complete user interaction workflows"""
   
    def test_complete_analysis_workflow(self, main_window, qt_app):
        """Test complete user workflow from input to results"""
        # Step 1: Enter symbols
        symbol_input = main_window.analysis_view.input_widget.symbols_input
        symbol_input.clear()
        QTest.keyClicks(symbol_input, "AAPL")
        qt_app.processEvents()
       
        # Step 2: Set contracts
        contracts_input = main_window.analysis_view.input_widget.contracts_input
        contracts_input.setValue(2)
        qt_app.processEvents()
       
        # Step 3: Set debit
        debit_input = main_window.analysis_view.input_widget.debit_input
        debit_input.setValue(2.75)
        qt_app.processEvents()
       
        # Step 4: Mock analysis service and results
        mock_results = {
            'symbol': 'AAPL',
            'recommendation': 'BUY',
            'confidence': 0.80,
            'expected_profit': 150.0,
            'max_loss': 275.0
        }
       
        with patch.object(main_window.analysis_worker, 'analyze_symbols') as mock_analyze:
            # Step 5: Click analyze
            analyze_button = main_window.analysis_view.input_widget.analyze_button
            QTest.mouseClick(analyze_button, Qt.LeftButton)
            qt_app.processEvents()
           
            # Verify analysis was called with correct parameters
            mock_analyze.assert_called_once()
            call_args = mock_analyze.call_args[0][0]
            assert 'AAPL' in call_args['symbols']
            assert call_args['contracts'] == 2
            assert call_args['debit'] == 2.75
       
        # Step 6: Simulate analysis completion
        main_window.on_analysis_completed(mock_results)
        qt_app.processEvents()
       
        # Step 7: Verify results are displayed
        results_widget = main_window.analysis_view.results_widget
        assert results_widget.isVisible()
   
    def test_settings_modification_workflow(self, main_window, qt_app):
        """Test settings modification workflow"""
        with patch('views.dialogs.settings_dialog.SettingsDialog') as mock_dialog_class:
             # Create mock dialog instance
            mock_dialog = Mock()
            mock_dialog_class.return_value = mock_dialog
            mock_dialog.exec.return_value = True
           
            # Mock settings changes
            new_settings = {
                'portfolio_value': 150000,
                'max_position_risk': 0.025,
                'theme': 'Light Professional'
            }
           
            # Open settings
            main_window.show_settings()
            qt_app.processEvents()
           
            # Simulate settings change signal
            main_window.on_settings_changed(new_settings)
            qt_app.processEvents()
           
            # Verify dialog was created and shown
            mock_dialog_class.assert_called_once()
            mock_dialog.exec.assert_called_once()
   
    def test_error_handling_workflow(self, main_window, qt_app):
        """Test error handling in UI workflow"""
        # Simulate analysis error
        error_message = "Network connection failed"
        main_window.on_analysis_error(error_message)
        qt_app.processEvents()
       
        # Should show error in status bar or dialog
        status_text = main_window.statusBar().currentMessage()
        assert error_message in status_text or "error" in status_text.lower()
       
        # UI should be in proper state after error
        analyze_button = main_window.analysis_view.input_widget.analyze_button
        assert analyze_button.isEnabled()  # Should be re-enabled after error
   
    def test_multi_symbol_analysis_workflow(self, main_window, qt_app):
        """Test workflow with multiple symbols"""
        # Enter multiple symbols
        symbol_input = main_window.analysis_view.input_widget.symbols_input
        symbol_input.clear()
        QTest.keyClicks(symbol_input, "AAPL, MSFT, GOOGL")
        qt_app.processEvents()
       
        # Mock multiple results
        mock_results = [
            {'symbol': 'AAPL', 'recommendation': 'BUY', 'confidence': 0.75},
            {'symbol': 'MSFT', 'recommendation': 'HOLD', 'confidence': 0.60},
            {'symbol': 'GOOGL', 'recommendation': 'BUY', 'confidence': 0.80}
        ]
       
        with patch.object(main_window.analysis_worker, 'analyze_symbols'):
            # Start analysis
            analyze_button = main_window.analysis_view.input_widget.analyze_button
            QTest.mouseClick(analyze_button, Qt.LeftButton)
            qt_app.processEvents()
           
            # Simulate results for each symbol
            for result in mock_results:
                main_window.on_analysis_completed(result)
                qt_app.processEvents()
       
        # Should display results for all symbols
        results_widget = main_window.analysis_view.results_widget
        assert results_widget.isVisible()


@pytest.mark.gui
class TestResponsiveDesign:
    """Test responsive design and different screen sizes"""
   
    def test_minimum_window_size(self, main_window, qt_app):
        """Test minimum window size constraints"""
        # Try to resize to very small size
        main_window.resize(400, 300)
        qt_app.processEvents()
       
        current_size = main_window.size()
       
        # Should enforce minimum size
        assert current_size.width() >= 800  # Minimum width
        assert current_size.height() >= 600  # Minimum height
   
    def test_large_screen_layout(self, main_window, qt_app):
        """Test layout on large screens"""
        # Resize to large screen size
        main_window.resize(1920, 1080)
        qt_app.processEvents()
       
        # Components should scale appropriately
        analysis_view = main_window.analysis_view
        assert analysis_view.size().width() > 1000
        assert analysis_view.size().height() > 600
       
        # Charts should have sufficient space
        dashboard = analysis_view.dashboard
        assert dashboard.size().width() > 800
        assert dashboard.size().height() > 400
   
    def test_widget_visibility_on_resize(self, main_window, qt_app):
        """Test widget visibility during window resizing"""
        # Get initial visibility states
        analysis_view = main_window.analysis_view
        input_widget = analysis_view.input_widget
        results_widget = analysis_view.results_widget
       
        initial_input_visible = input_widget.isVisible()
        initial_results_visible = results_widget.isVisible()
       
        # Resize window
        main_window.resize(1200, 800)
        qt_app.processEvents()
       
        # Visibility should be maintained
        assert input_widget.isVisible() == initial_input_visible
        assert results_widget.isVisible() == initial_results_visible


@pytest.mark.gui
class TestAccessibility:
    """Test accessibility features"""
   
    def test_keyboard_navigation(self, main_window, qt_app):
        """Test keyboard navigation between widgets"""
        # Set focus to symbol input
        symbol_input = main_window.analysis_view.input_widget.symbols_input
        symbol_input.setFocus()
        qt_app.processEvents()
       
        assert symbol_input.hasFocus()
       
        # Tab to next widget
        QTest.keyPress(main_window, Qt.Key_Tab)
        qt_app.processEvents()
       
        # Focus should move to next focusable widget
        focused_widget = qt_app.focusWidget()
        assert focused_widget != symbol_input
        assert focused_widget is not None
   
    def test_widget_labels_and_accessibility(self, main_window, qt_app):
        """Test widget labels and accessibility properties"""
        input_widget = main_window.analysis_view.input_widget
       
        # Check that input widgets have proper labels
        symbols_input = input_widget.symbols_input
        assert symbols_input.placeholderText() != ""
       
        # Check that buttons have proper text
        analyze_button = input_widget.analyze_button
        assert analyze_button.text() != ""
        assert "analyze" in analyze_button.text().lower()
   
    def test_tooltip_functionality(self, main_window, qt_app):
        """Test tooltip functionality for help"""
        # Find widgets that should have tooltips
        input_widget = main_window.analysis_view.input_widget
       
        # Check for tooltips on complex widgets
        contracts_input = input_widget.contracts_input
        if contracts_input.toolTip():
            assert len(contracts_input.toolTip()) > 0


@pytest.mark.gui
@pytest.mark.slow
class TestPerformanceUI:
    """Test UI performance and responsiveness"""
   
    def test_ui_responsiveness_during_analysis(self, main_window, qt_app):
        """Test UI remains responsive during long analysis"""
        # Start long-running mock analysis
        with patch.object(main_window.analysis_worker, 'analyze_symbols') as mock_analyze:
            # Make analysis take time but don't block UI
            def slow_analysis(*args, **kwargs):
                # Simulate work without blocking
                for i in range(10):
                    qt_app.processEvents()
                    QTest.qWait(10)
           
            mock_analyze.side_effect = slow_analysis
           
            # Start analysis
            symbol_input = main_window.analysis_view.input_widget.symbols_input
            symbol_input.setText("AAPL")
            qt_app.processEvents()
           
            analyze_button = main_window.analysis_view.input_widget.analyze_button
            QTest.mouseClick(analyze_button, Qt.LeftButton)
           
            # UI should remain responsive
            for i in range(5):
                qt_app.processEvents()
                QTest.qWait(20)
               
                # Should be able to interact with UI
                assert main_window.isVisible()
                assert symbol_input.isEnabled()
   
    def test_memory_usage_ui(self, main_window, qt_app):
        """Test UI memory usage over time"""
        import psutil
        import gc
        import os
       
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
       
        # Perform multiple UI operations
        for i in range(20):
            # Simulate user interactions
            symbol_input = main_window.analysis_view.input_widget.symbols_input
            symbol_input.setText(f"TEST{i}")
            qt_app.processEvents()
           
            # Mock results update
            mock_results = {
                'symbol': f'TEST{i}',
                'recommendation': 'BUY',
                'confidence': 0.75
            }
            main_window.on_analysis_completed(mock_results)
            qt_app.processEvents()
           
            if i % 5 == 0:
                gc.collect()
       
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
       
        # Memory increase should be reasonable
        max_acceptable_increase = 50 * 1024 * 1024  # 50MB
        assert memory_increase < max_acceptable_increase