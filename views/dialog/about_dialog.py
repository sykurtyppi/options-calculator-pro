"""
About dialog for the Professional Options Calculator
Professional information and credits display
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QFrame, QScrollArea, QTabWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap, QIcon
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AboutDialog(QDialog):
    """Professional about dialog with comprehensive information"""
    
    link_clicked = Signal(str)  # Emits URL when link is clicked
    
    def __init__(self, version="1.0.0", parent=None):
        super().__init__(parent)
        self.version = version
        self.setup_ui()
        
    def setup_ui(self):
        """Setup about dialog UI"""
        self.setWindowTitle("About - Options Calculator Pro")
        self.setModal(True)
        self.resize(600, 500)
        
        # Apply professional styling
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            
            QLabel {
                color: #ffffff;
            }
            
            QTextEdit {
                background-color: #353535;
                border: 2px solid #4a4a4a;
                border-radius: 6px;
                padding: 10px;
                color: #ffffff;
            }
        """)
        
        layout = QVBoxLayout(self)
        
        # Header section
        self.create_header_section()
        layout.addWidget(self.header_frame)
        
        # Create tabbed content
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 2px solid #4a4a4a;
                border-radius: 6px;
                background-color: #2b2b2b;
            }
            
            QTabBar::tab {
                background-color: #404040;
                color: #ffffff;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            
            QTabBar::tab:selected {
                background-color: #60a3d9;
            }
        """)
        
        # Add content tabs
        self.create_about_tab()
        self.create_features_tab()
        self.create_credits_tab()
        self.create_license_tab()
        
        layout.addWidget(self.tab_widget)
        
        # Close button
        self.create_close_button()
        layout.addWidget(self.close_button_frame)
    
    def create_header_section(self):
        """Create header with logo and title"""
        self.header_frame = QFrame()
        header_layout = QVBoxLayout(self.header_frame)
        
        # Application title
        title = QLabel("Options Calculator Pro")
        title.setStyleSheet("""
            QLabel {
                font-size: 24pt;
                font-weight: bold;
                color: #60a3d9;
                margin: 10px;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)
        
        # Version and build info
        version_info = QLabel(f"Version {self.version} - Professional Edition")
        version_info.setStyleSheet("""
            QLabel {
                font-size: 12pt;
                color: #cccccc;
                margin-bottom: 10px;
            }
        """)
        version_info.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(version_info)
        
        # Tagline
        tagline = QLabel("Professional Options Analysis for Serious Traders")
        tagline.setStyleSheet("""
            QLabel {
                font-size: 11pt;
                font-style: italic;
                color: #999999;
                margin-bottom: 15px;
            }
        """)
        tagline.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(tagline)
    
    def create_about_tab(self):
        """Create about information tab"""
        about_widget = QFrame()
        layout = QVBoxLayout(about_widget)
        
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setHtml("""
        <h3 style="color: #60a3d9;">About Options Calculator Pro</h3>
        
        <p><strong>Options Calculator Pro</strong> is a comprehensive, professional-grade tool designed specifically 
        for options traders focusing on earnings calendar spreads and advanced options strategies.</p>
        
        <h4 style="color: #60a3d9;">Key Capabilities:</h4>
        <ul>
            <li><strong>Advanced Analytics:</strong> Monte Carlo simulations using the Heston stochastic volatility model</li>
            <li><strong>Machine Learning:</strong> AI-powered predictions based on historical trade outcomes</li>
            <li><strong>Risk Management:</strong> Professional position sizing using Kelly Criterion</li>
            <li><strong>Real-time Data:</strong> Multi-source market data with intelligent failover</li>
            <li><strong>Comprehensive Greeks:</strong> Full options Greeks analysis for calendar spreads</li>
            <li><strong>Backtesting:</strong> Historical strategy performance analysis</li>
        </ul>
        
        <h4 style="color: #60a3d9;">Target Users:</h4>
        <p>This application is designed for serious retail traders, professional traders, and quantitative analysts 
        who require sophisticated tools for options analysis and risk management.</p>
        
        <h4 style="color: #60a3d9;">Technology Stack:</h4>
        <ul>
            <li><strong>Framework:</strong> PySide6 (Qt for Python)</li>
            <li><strong>Analytics:</strong> NumPy, SciPy, Pandas</li>
            <li><strong>Machine Learning:</strong> Scikit-learn</li>
            <li><strong>Visualization:</strong> Matplotlib with professional themes</li>
            <li><strong>Data Sources:</strong> Yahoo Finance, Alpha Vantage, Finnhub</li>
        </ul>
        
        <p style="margin-top: 20px;"><em>Built with precision, designed for professionals.</em></p>
        """)
        
        layout.addWidget(about_text)
        self.tab_widget.addTab(about_widget, "About")
    
    def create_features_tab(self):
        """Create features overview tab"""
        features_widget = QFrame()
        layout = QVBoxLayout(features_widget)
        
        features_text = QTextEdit()
        features_text.setReadOnly(True)
        features_text.setHtml("""
        <h3 style="color: #60a3d9;">Comprehensive Feature Set</h3>
        
        <h4 style="color: #60a3d9;">📊 Analysis Engine</h4>
        <ul>
            <li><strong>Multi-threaded Analysis:</strong> Parallel processing for multiple symbols</li>
            <li><strong>Real-time Data Integration:</strong> Smart API management with automatic failover</li>
            <li><strong>Volatility Analysis:</strong> IV term structure, realized vs implied volatility</li>
            <li><strong>Earnings Integration:</strong> Automatic earnings date detection and IV crush modeling</li>
        </ul>
        
        <h4 style="color: #60a3d9;">🧠 Machine Learning</h4>
        <ul>
            <li><strong>Ensemble Models:</strong> Random Forest + Logistic Regression voting classifier</li>
            <li><strong>Feature Engineering:</strong> IV/RV ratios, term structure slopes, market regime indicators</li>
            <li><strong>Continuous Learning:</strong> Model retraining with new trade outcomes</li>
            <li><strong>Confidence Scoring:</strong> Probabilistic predictions with confidence intervals</li>
        </ul>
        
        <h4 style="color: #60a3d9;">📈 Risk Management</h4>
        <ul>
            <li><strong>Kelly Criterion:</strong> Optimal position sizing based on win probability and payoff ratios</li>
            <li><strong>Portfolio Integration:</strong> Position sizing relative to total portfolio value</li>
            <li><strong>Greeks Monitoring:</strong> Real-time Greeks calculations for risk assessment</li>
            <li><strong>Scenario Analysis:</strong> Monte Carlo simulation for various market conditions</li>
        </ul>
        
        <h4 style="color: #60a3d9;">📋 Trade Management</h4>
        <ul>
            <li><strong>Trade History:</strong> Comprehensive logging of all trades with outcomes</li>
            <li><strong>Performance Analytics:</strong> Win rate, Sharpe ratio, maximum drawdown analysis</li>
            <li><strong>Backtesting Engine:</strong> Historical strategy performance evaluation</li>
            <li><strong>Export Capabilities:</strong> CSV export for external analysis</li>
        </ul>
        
        <h4 style="color: #60a3d9;">🔍 Scanner & Watchlist</h4>
        <ul>
            <li><strong>Multi-symbol Scanning:</strong> Batch analysis of multiple stocks</li>
            <li><strong>Custom Filters:</strong> Configurable criteria for opportunity identification</li>
            <li><strong>Real-time Monitoring:</strong> Continuous watchlist scanning</li>
            <li><strong>Opportunity Alerts:</strong> Notifications when criteria are met</li>
        </ul>
        
        <h4 style="color: #60a3d9;">📊 Visualization</h4>
        <ul>
            <li><strong>Professional Charts:</strong> P/L diagrams, price distributions, IV term structure</li>
           <li><strong>Interactive Dashboards:</strong> Multi-metric displays with real-time updates</li>
           <li><strong>Export Quality:</strong> High-resolution chart exports for presentations</li>
           <li><strong>Dark Theme:</strong> Professional dark interface optimized for extended use</li>
       </ul>
       
       <h4 style="color: #60a3d9;">⚙️ Professional Tools</h4>
       <ul>
           <li><strong>Data Validation:</strong> Automated data quality checks and cross-validation</li>
           <li><strong>Cache Management:</strong> Intelligent caching with TTL and automatic cleanup</li>
           <li><strong>Multi-threading:</strong> Optimized for Apple Silicon and multi-core processors</li>
           <li><strong>Error Recovery:</strong> Robust error handling with automatic retries</li>
       </ul>
       """)
       
        layout.addWidget(features_text)
        self.tab_widget.addTab(features_widget, "Features")
   
    def create_credits_tab(self):
        """Create credits and acknowledgments tab"""
        credits_widget = QFrame()
        layout = QVBoxLayout(credits_widget)
       
        credits_text = QTextEdit()
        credits_text.setReadOnly(True)
        credits_text.setHtml("""
        <h3 style="color: #60a3d9;">Credits & Acknowledgments</h3>
       
        <h4 style="color: #60a3d9;">👨‍💻 Development Team</h4>
        <p><strong>Lead Developer:</strong> Tristan Alejandro<br>
        <em>Quantitative Finance & Software Engineering</em></p>
       
        <h4 style="color: #60a3d9;">📚 Academic & Research Foundations</h4>
        <ul>
            <li><strong>Black-Scholes-Merton Model:</strong> Fischer Black, Myron Scholes, Robert C. Merton</li>
            <li><strong>Heston Stochastic Volatility Model:</strong> Steven L. Heston (1993)</li>
            <li><strong>Kelly Criterion:</strong> John Larry Kelly Jr. (1956)</li>
            <li><strong>Monte Carlo Methods:</strong> Stanislaw Ulam, John von Neumann</li>
        </ul>
       
        <h4 style="color: #60a3d9;">🛠️ Technology Stack</h4>
        <ul>
            <li><strong>PySide6:</strong> Qt for Python - Professional GUI framework</li>
            <li><strong>NumPy:</strong> Fundamental package for scientific computing</li>
            <li><strong>SciPy:</strong> Scientific computing and technical computing</li>
            <li><strong>Pandas:</strong> Data analysis and manipulation library</li>
            <li><strong>Matplotlib:</strong> Comprehensive library for creating visualizations</li>
            <li><strong>Scikit-learn:</strong> Machine learning library for Python</li>
            <li><strong>yfinance:</strong> Yahoo Finance market data retrieval</li>
        </ul>
       
        <h4 style="color: #60a3d9;">🌐 Data Providers</h4>
        <ul>
            <li><strong>Yahoo Finance:</strong> Primary market data source</li>
            <li><strong>Alpha Vantage:</strong> Financial market data API</li>
            <li><strong>Finnhub:</strong> Real-time financial data API</li>
            <li><strong>CBOE:</strong> VIX and volatility index data</li>
        </ul>
        
        <h4 style="color: #60a3d9;">📖 Educational Resources</h4>
        <ul>
            <li><strong>"Options, Futures, and Other Derivatives"</strong> - John C. Hull</li>
            <li><strong>"Option Volatility and Pricing"</strong> - Sheldon Natenberg</li>
            <li><strong>"Dynamic Hedging"</strong> - Nassim Nicholas Taleb</li>
            <li><strong>"Quantitative Portfolio Theory"</strong> - Various Academic Papers</li> 
        </ul>
       
        <h4 style="color: #60a3d9;">🎨 Design Inspiration</h4>
        <ul>
            <li><strong>Bloomberg Terminal:</strong> Professional financial interface design</li>
            <li><strong>TradingView:</strong> Modern charting and analysis tools</li>
            <li><strong>Interactive Brokers TWS:</strong> Professional trading platform</li>
        </ul>
       
        <h4 style="color: #60a3d9;">🙏 Special Thanks</h4>
        <p>To the open-source community, academic researchers in quantitative finance, 
        and professional traders who have shared their knowledge and insights that made 
        this application possible.</p>
       
        <p style="margin-top: 20px; font-style: italic; color: #cccccc;">
        "Standing on the shoulders of giants" - Isaac Newton
        </p>
        """)
       
        layout.addWidget(credits_text)
        self.tab_widget.addTab(credits_widget, "Credits")
   
    def create_license_tab(self):
        """Create license and legal information tab"""
        license_widget = QFrame()
        layout = QVBoxLayout(license_widget)
       
        license_text = QTextEdit()
        license_text.setReadOnly(True)
        license_text.setHtml(f"""
        <h3 style="color: #60a3d9;">License & Legal Information</h3>
       
        <h4 style="color: #60a3d9;">📄 Software License</h4>
        <p><strong>Options Calculator Pro v{self.version}</strong></p>
        <p>Copyright © {datetime.now().year} Tristan Alejandro. All rights reserved.</p>
       
        <p>This software is provided for educational and professional use. The software is provided "as is" 
        without warranty of any kind, express or implied.</p>
       
        <h4 style="color: #60a3d9;">⚠️ Disclaimer</h4>
        <div style="background-color: #3d2914; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0;">
            <p><strong>IMPORTANT FINANCIAL DISCLAIMER:</strong></p>
            <ul>
                <li>This software is for educational and informational purposes only</li>
                <li>No content should be construed as investment advice</li>
                <li>Options trading involves substantial risk of loss</li>
                <li>Past performance does not guarantee future results</li>
                <li>You should consult with a financial advisor before making investment decisions</li>
                <li>The developers are not responsible for any financial losses</li>
            </ul>
        </div>
       
        <h4 style="color: #60a3d9;">🔒 Data Usage & Privacy</h4>
        <ul>
            <li><strong>Market Data:</strong> This application retrieves market data from third-party providers</li>
            <li><strong>Local Storage:</strong> All analysis and trade data is stored locally on your device</li>
            <li><strong>No Telemetry:</strong> This application does not collect or transmit user data</li>
            <li><strong>API Keys:</strong> Your API keys are stored locally and never transmitted to our servers</li>
        </ul>
       
        <h4 style="color: #60a3d9;">📋 Third-Party Licenses</h4>
        <p>This software incorporates several open-source libraries, each with their own licenses:</p>
        <ul>
            <li><strong>PySide6:</strong> LGPL v3 License</li>
            <li><strong>NumPy:</strong> BSD License</li>
            <li><strong>SciPy:</strong> BSD License</li>
            <li><strong>Pandas:</strong> BSD License</li>
            <li><strong>Matplotlib:</strong> Python Software Foundation License</li>
            <li><strong>Scikit-learn:</strong> BSD License</li>
        </ul>
       
        <h4 style="color: #60a3d9;">🌍 Regulatory Compliance</h4>
        <p><strong>For US Users:</strong> This software is not regulated by the SEC, CFTC, or FINRA. 
        It is a technical analysis tool only.</p>
       
        <p><strong>International Users:</strong> Please ensure compliance with your local financial regulations 
        before using this software for trading decisions.</p>
       
        <h4 style="color: #60a3d9;">📞 Support & Contact</h4>
        <p>For technical support, feature requests, or bug reports:</p>
        <ul>
            <li><strong>Email:</strong> support@optionscalculatorpro.com</li>
            <li><strong>GitHub:</strong> github.com/username/options-calculator-pro</li>
            <li><strong>Documentation:</strong> docs.optionscalculatorpro.com</li>
        </ul>
       
        <hr style="margin: 20px 0; border: 1px solid #4a4a4a;">
       
        <p style="font-size: 9pt; color: #888888; text-align: center;">
        This software is provided for educational purposes. Trading options involves substantial risk. 
        Please trade responsibly and never risk more than you can afford to lose.
        </p>
        """)
       
        layout.addWidget(license_text)
        self.tab_widget.addTab(license_widget, "License")
   
    def create_close_button(self):
        """Create close button frame"""
        self.close_button_frame = QFrame()
        button_layout = QHBoxLayout(self.close_button_frame)
       
        # Version info on left
        build_info = QLabel(f"Build: {datetime.now().strftime('%Y%m%d')} | Python 3.11+ | Qt 6")
        build_info.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 8pt;
                font-family: monospace;
            }
        """)
        button_layout.addWidget(build_info)
       
        button_layout.addStretch()
       
        # Links
        github_btn = QPushButton("🔗 GitHub")
        github_btn.setFlat(True)
        github_btn.setStyleSheet("""
            QPushButton {
                color: #60a3d9;
                border: none;
                text-decoration: underline;
                font-size: 10pt;
            }
            QPushButton:hover {
                color: #7bb3e0;
            }
        """)
        github_btn.clicked.connect(lambda: self.link_clicked.emit("https://github.com/username/options-calculator-pro"))
        button_layout.addWidget(github_btn)
       
        docs_btn = QPushButton("📖 Documentation")
        docs_btn.setFlat(True)
        docs_btn.setStyleSheet("""
            QPushButton {
                color: #60a3d9;
                border: none;
                text-decoration: underline;
                font-size: 10pt;
            }
            QPushButton:hover {
                color: #7bb3e0;
            }
        """)
        docs_btn.clicked.connect(lambda: self.link_clicked.emit("https://docs.optionscalculatorpro.com"))
        button_layout.addWidget(docs_btn)
       
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: #ffffff;
                border: none;
                padding: 8px 24px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
        """)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)