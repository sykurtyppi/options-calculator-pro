    def __init__(self, config_manager, market_data_service, ml_service, thread_manager, parent=None):
        super().__init__(parent)
        import logging
        self.logger = logging.getLogger(__name__)
        
        self.config_manager = config_manager
        self.market_data_service = market_data_service
        self.ml_service = ml_service
        self.thread_manager = thread_manager
        
        # Initialize services
        self.options_service = OptionsService(self.config_manager, self.market_data_service)
        self.volatility_service = VolatilityService(self.config_manager, self.market_data_service)
        self.greeks_calculator = GreeksCalculator(self.config_manager)
        
        self.logger.info("AnalysisController initialized")
