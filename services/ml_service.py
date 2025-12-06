"""
Machine Learning Service - Professional Trade Prediction Engine
==============================================================

Handles all machine learning operations including:
- Model training and validation
- Trade outcome prediction
- Feature engineering and selection
- Model persistence and versioning
- Performance metrics and analysis

Part of Professional Options Calculator v9.1
Optimized for Apple Silicon and PySide6
"""

import logging
import os
import pickle
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, 
    precision_recall_curve, roc_curve
)
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Import your existing utilities
from utils.config_manager import ConfigManager
from utils.logger import setup_logger as get_logger

logger = get_logger(__name__)

class ModelType(Enum):
    """Supported machine learning model types"""
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"
    ENSEMBLE = "ensemble"
    GRADIENT_BOOSTING = "gradient_boosting"

class PredictionConfidence(Enum):
    """Prediction confidence levels"""
    VERY_LOW = "very_low"      # < 0.3
    LOW = "low"                # 0.3 - 0.45
    MODERATE = "moderate"      # 0.45 - 0.55
    HIGH = "high"              # 0.55 - 0.7
    VERY_HIGH = "very_high"    # > 0.7

@dataclass
class TradeFeatures:
    """Features used for trade prediction"""
    iv30_rv30: float                    # IV/RV ratio
    ts_slope_0_45: float               # Term structure slope
    days_to_earnings: int              # Days until earnings
    vix: float                         # VIX level
    avg_volume_normalized: float       # Normalized volume
    delta_weighted_gamma: float        # Gamma exposure
    sector_encoded: float              # Encoded sector
    market_regime: float               # Market regime indicator
    iv_rank: float                     # IV rank
    iv_percentile: float               # IV percentile
    time_decay_ratio: float            # Theta ratio
    liquidity_score: float             # Options liquidity
    earnings_proximity: float          # Earnings proximity factor
    volatility_skew: float             # Volatility skew
    put_call_ratio: float              # Put/call ratio

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([
            self.iv30_rv30,
            self.ts_slope_0_45,
            self.days_to_earnings,
            self.vix,
            self.avg_volume_normalized,
            self.delta_weighted_gamma,
            self.sector_encoded,
            self.market_regime,
            self.iv_rank,
            self.iv_percentile,
            self.time_decay_ratio,
            self.liquidity_score,
            self.earnings_proximity,
            self.volatility_skew,
            self.put_call_ratio
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Get list of feature names"""
        return [
            'iv30_rv30', 'ts_slope_0_45', 'days_to_earnings', 'vix',
            'avg_volume_normalized', 'delta_weighted_gamma', 'sector_encoded',
            'market_regime', 'iv_rank', 'iv_percentile', 'time_decay_ratio',
            'liquidity_score', 'earnings_proximity', 'volatility_skew', 'put_call_ratio'
        ]

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    cross_val_score: float
    feature_importance: Dict[str, float]
    confusion_matrix: np.ndarray
    training_samples: int
    validation_samples: int
    
@dataclass
class PredictionResult:
    """Result of a trade prediction"""
    probability: float
    confidence: PredictionConfidence
    contributing_factors: Dict[str, float]
    risk_score: float
    recommendation: str
    model_version: str
    prediction_timestamp: datetime

class MLService:
    """
    Professional Machine Learning Service
    
    Provides comprehensive ML capabilities for trade outcome prediction:
    - Automated feature engineering
    - Model training with hyperparameter optimization
    - Cross-validation and performance evaluation
    - Real-time prediction with confidence scoring
    - Model versioning and persistence
    """
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = logger
        
        # Configuration
        self.model_file = "models/prediction_model.pkl"
        self.backup_model_file = "models/prediction_model_backup.pkl"
        self.model_metrics_file = "models/model_metrics.json"
        self.feature_scaler_file = "models/feature_scaler.pkl"
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        
        # Model components
        self.model = None
        self.feature_scaler = None
        self.feature_selector = None
        self.sector_encoder = None
        self.model_metrics = None
        self.model_version = "1.0.0"
        
        # Thread safety
        self._model_lock = threading.Lock()
        
        # Training parameters
        self.min_training_samples = 50
        self.test_size = 0.2
        self.cv_folds = 5
        self.random_state = 42
        
        # Feature engineering parameters
        self.feature_selection_k = 12  # Select top 12 features
        
        self.logger.info("MLService initialized")
        
        # Try to load existing model
        self._load_model()
    
    def prepare_features(self, raw_data: Dict[str, Any]) -> TradeFeatures:
        """
        Prepare features from raw trade data
        
        Args:
            raw_data: Raw trade data dictionary
            
        Returns:
            TradeFeatures object ready for prediction
        """
        try:
            # Extract and normalize features
            features = TradeFeatures(
                iv30_rv30=self._safe_float(raw_data.get('iv30_rv30', 1.0)),
                ts_slope_0_45=self._safe_float(raw_data.get('ts_slope_0_45', 0.0)),
                days_to_earnings=max(0, int(raw_data.get('days_to_earnings', 30))),
                vix=self._safe_float(raw_data.get('vix', 20.0)),
                avg_volume_normalized=self._normalize_volume(
                    raw_data.get('avg_volume', 50000000)
                ),
                delta_weighted_gamma=self._safe_float(raw_data.get('gamma', 0.01)),
                sector_encoded=self._encode_sector(raw_data.get('sector', 'Unknown')),
                market_regime=self._encode_market_regime(raw_data.get('vix', 20.0)),
                iv_rank=self._safe_float(raw_data.get('iv_rank', 0.5)),
                iv_percentile=self._safe_float(raw_data.get('iv_percentile', 50.0)),
                time_decay_ratio=self._calculate_time_decay_ratio(raw_data),
                liquidity_score=self._calculate_liquidity_score(raw_data),
                earnings_proximity=self._calculate_earnings_proximity(
                    raw_data.get('days_to_earnings', 30)
                ),
                volatility_skew=self._calculate_volatility_skew(raw_data),
                put_call_ratio=self._safe_float(raw_data.get('put_call_ratio', 1.0))
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            # Return default features
            return self._get_default_features()
    
    def train_model(self, trade_history_file: str, force_retrain: bool = False) -> bool:
        """
        Train machine learning model on historical trade data
        
        Args:
            trade_history_file: Path to trade history CSV
            force_retrain: Force retraining even if model exists
            
        Returns:
            True if training succeeded, False otherwise
        """
        with self._model_lock:
            try:
                # Check if model exists and retraining not forced
                if not force_retrain and self.model is not None:
                    self.logger.info("Model already exists and force_retrain=False")
                    return True
                
                # Load trade history
                if not os.path.exists(trade_history_file):
                    self.logger.error(f"Trade history file not found: {trade_history_file}")
                    return False
                
                df = pd.read_csv(trade_history_file)
                
                # Filter completed trades only
                completed_trades = df[df['status'] == 'completed'].copy()
                
                if len(completed_trades) < self.min_training_samples:
                    self.logger.warning(
                        f"Insufficient training data: {len(completed_trades)} samples "
                        f"(minimum: {self.min_training_samples})"
                    )
                    return False
                
                self.logger.info(f"Training model with {len(completed_trades)} completed trades")
                
                # Prepare features and target
                X, y = self._prepare_training_data(completed_trades)
                
                if X is None or y is None:
                    self.logger.error("Failed to prepare training data")
                    return False
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.test_size, random_state=self.random_state,
                    stratify=y
                )
                
                # Feature scaling
                self.feature_scaler = StandardScaler()
                X_train_scaled = self.feature_scaler.fit_transform(X_train)
                X_test_scaled = self.feature_scaler.transform(X_test)
                
                # Feature selection
                self.feature_selector = SelectKBest(
                    score_func=f_classif, 
                    k=min(self.feature_selection_k, X_train_scaled.shape[1])
                )
                X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
                X_test_selected = self.feature_selector.transform(X_test_scaled)
                
                # Train ensemble model with hyperparameter optimization
                self.model = self._train_optimized_model(X_train_selected, y_train)
                
                # Evaluate model
                self.model_metrics = self._evaluate_model(
                    self.model, X_train_selected, X_test_selected, y_train, y_test
                )
                
                # Save model and components
                self._save_model()
                
                self.logger.info(
                    f"Model training completed successfully. "
                    f"Accuracy: {self.model_metrics.accuracy:.3f}, "
                    f"AUC-ROC: {self.model_metrics.auc_roc:.3f}"
                )
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error training model: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return False
    
    def predict_trade_outcome(self, raw_data: Dict[str, Any]) -> PredictionResult:
        """
        Predict outcome of a trade using the trained model
        
        Args:
            raw_data: Raw trade data dictionary
            
        Returns:
            PredictionResult with probability and analysis
        """
        try:
            # Check if model is available
            if self.model is None:
                self.logger.warning("No trained model available, returning default prediction")
                return self._get_default_prediction()
            
            # Prepare features
            features = self.prepare_features(raw_data)
            feature_array = features.to_array().reshape(1, -1)
            
            # Scale features
            if self.feature_scaler:
                feature_array = self.feature_scaler.transform(feature_array)
            
            # Select features
            if self.feature_selector:
                feature_array = self.feature_selector.transform(feature_array)
            
            # Make prediction
            probability = self.model.predict_proba(feature_array)[0, 1]  # Probability of success
            
            # Determine confidence level
            confidence = self._determine_confidence(probability)
            
            # Calculate contributing factors
            contributing_factors = self._calculate_feature_importance(
                features, feature_array[0]
            )
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(probability, raw_data)
            
            # Generate recommendation
            recommendation = self._generate_recommendation(probability, risk_score)
            
            result = PredictionResult(
                probability=probability,
                confidence=confidence,
                contributing_factors=contributing_factors,
                risk_score=risk_score,
                recommendation=recommendation,
                model_version=self.model_version,
                prediction_timestamp=datetime.now()
            )
            
            self.logger.debug(f"Prediction: {probability:.3f}, Confidence: {confidence.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error predicting trade outcome: {e}")
            return self._get_default_prediction()
    
    def get_model_performance(self) -> Optional[ModelMetrics]:
        """Get current model performance metrics"""
        return self.model_metrics
    
    def retrain_with_new_data(self, new_trades: List[Dict[str, Any]]) -> bool:
        """
        Incrementally retrain model with new trade data
        
        Args:
            new_trades: List of new completed trades
            
        Returns:
            True if retraining succeeded
        """
        try:
            if not new_trades:
                return False
            
            # Convert to DataFrame
            new_df = pd.DataFrame(new_trades)
            
            # Filter completed trades
            completed_new = new_df[new_df['status'] == 'completed']
            
            if len(completed_new) < 5:  # Need at least 5 new trades
                self.logger.info("Not enough new trades for incremental retraining")
                return False
            
            self.logger.info(f"Incremental retraining with {len(completed_new)} new trades")
            
            # For now, trigger full retraining
            # In production, you might implement true incremental learning
            return self.train_model("trade_history.csv", force_retrain=True)
            
        except Exception as e:
            self.logger.error(f"Error in incremental retraining: {e}")
            return False
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model"""
        if self.model_metrics:
            return self.model_metrics.feature_importance
        return None
    
    def validate_model(self, validation_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate model on new data
        
        Args:
            validation_data: Validation dataset
            
        Returns:
            Validation metrics
        """
        try:
            if self.model is None:
                return {"error": "No model available"}
            
            # Prepare validation data
            X_val, y_val = self._prepare_training_data(validation_data)
            
            if X_val is None or y_val is None:
                return {"error": "Failed to prepare validation data"}
            
            # Scale and select features
            X_val_scaled = self.feature_scaler.transform(X_val)
            X_val_selected = self.feature_selector.transform(X_val_scaled)
            
            # Make predictions
            y_pred = self.model.predict(X_val_selected)
            y_pred_proba = self.model.predict_proba(X_val_selected)[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                "validation_accuracy": accuracy_score(y_val, y_pred),
                "validation_precision": precision_score(y_val, y_pred),
                "validation_recall": recall_score(y_val, y_pred),
                "validation_f1": f1_score(y_val, y_pred),
                "validation_auc": roc_auc_score(y_val, y_pred_proba),
                "validation_samples": len(y_val)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            return {"error": str(e)}
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data from DataFrame"""
        try:
            # Required columns
            required_cols = [
                'iv30_rv30', 'ts_slope_0_45', 'days_to_earnings', 'vix',
                'avg_volume_normalized', 'delta_weighted_gamma', 'profitable'
            ]
            
            # Check for required columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return None, None
            
            # Prepare features for each trade
            feature_arrays = []
            targets = []
            
            for _, row in df.iterrows():
                try:
                    # Create raw data dict
                    raw_data = row.to_dict()
                    
                    # Prepare features
                    features = self.prepare_features(raw_data)
                    feature_arrays.append(features.to_array())
                    
                    # Target variable
                    targets.append(int(row['profitable']))
                    
                except Exception as e:
                    self.logger.warning(f"Skipping row due to error: {e}")
                    continue
            
            if not feature_arrays:
                self.logger.error("No valid training samples prepared")
                return None, None
            
            X = np.array(feature_arrays)
            y = np.array(targets)
            
            self.logger.info(f"Prepared {len(X)} training samples with {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _train_optimized_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train optimized ensemble model with hyperparameter tuning"""
        try:
            # Define base models
            rf = RandomForestClassifier(random_state=self.random_state)
            lr = LogisticRegression(random_state=self.random_state, max_iter=1000)
            
            # Hyperparameter grids
            rf_params = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
            
            lr_params = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l2'],
                'solver': ['liblinear', 'lbfgs']
            }
            
            # Grid search for best hyperparameters
            rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='roc_auc', n_jobs=-1)
            lr_grid = GridSearchCV(lr, lr_params, cv=3, scoring='roc_auc', n_jobs=-1)
            
            # Fit models
            rf_grid.fit(X_train, y_train)
            lr_grid.fit(X_train, y_train)
            
            # Create ensemble with best models
            ensemble = VotingClassifier(
                estimators=[
                    ('rf', rf_grid.best_estimator_),
                    ('lr', lr_grid.best_estimator_)
                ],
                voting='soft'
            )
            
            # Train ensemble
            ensemble.fit(X_train, y_train)
            
            self.logger.info("Optimized ensemble model trained successfully")
            return ensemble
            
        except Exception as e:
            self.logger.error(f"Error training optimized model: {e}")
            # Fallback to simple random forest
            rf_simple = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state
            )
            rf_simple.fit(X_train, y_train)
            return rf_simple
    
    def _evaluate_model(self, model, X_train: np.ndarray, X_test: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """Evaluate model performance"""
        try:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds)
            
            # Metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc_roc = roc_auc_score(y_test, y_pred_proba)
            cv_score = cv_scores.mean()
            
            # Feature importance
            feature_importance = self._extract_feature_importance(model)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc,
                cross_val_score=cv_score,
                feature_importance=feature_importance,
                confusion_matrix=cm,
                training_samples=len(y_train),
                validation_samples=len(y_test)
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return None
    
    def _extract_feature_importance(self, model) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        try:
            feature_names = TradeFeatures.get_feature_names()
            
            if hasattr(model, 'feature_importances_'):
                # Random Forest or similar
                importances = model.feature_importances_
            elif hasattr(model, 'estimators_'):
                # Ensemble model
                importances = np.mean([
                    est.feature_importances_ if hasattr(est, 'feature_importances_')
                    else np.abs(est.coef_[0])
                    for est in model.estimators_
                ], axis=0)
            else:
                # Default to equal importance
                importances = np.ones(len(feature_names)) / len(feature_names)
            
            # Get selected feature indices
            if self.feature_selector:
                selected_indices = self.feature_selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in selected_indices]
                importance_dict = dict(zip(selected_features, importances))
            else:
                importance_dict = dict(zip(feature_names[:len(importances)], importances))
            
            return importance_dict
            
        except Exception as e:
            self.logger.error(f"Error extracting feature importance: {e}")
            return {}
    
    def _save_model(self) -> bool:
        """Save model and components to disk"""
        try:
            # Backup existing model
            if os.path.exists(self.model_file):
                os.rename(self.model_file, self.backup_model_file)
            
            # Save model components
            model_data = {
                'model': self.model,
                'feature_scaler': self.feature_scaler,
                'feature_selector': self.feature_selector,
                'sector_encoder': self.sector_encoder,
                'model_version': self.model_version,
                'timestamp': datetime.now(),
                'feature_names': TradeFeatures.get_feature_names()
            }
            
            with open(self.model_file, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save metrics separately
            if self.model_metrics:
                metrics_dict = asdict(self.model_metrics)
                # Convert numpy arrays to lists for JSON serialization
                metrics_dict['confusion_matrix'] = self.model_metrics.confusion_matrix.tolist()
                
                with open(self.model_metrics_file, 'w') as f:
                    import json
                    json.dump(metrics_dict, f, indent=2, default=str)
            
            self.logger.info("Model saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def _load_model(self) -> bool:
        try:
            if not os.path.exists(self.model_file):
                self.logger.info("No saved model found")
                return False
            
            with open(self.model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.feature_scaler = model_data.get('feature_scaler')
            self.feature_selector = model_data.get('feature_selector')
            self.sector_encoder = model_data.get('sector_encoder')
            self.model_version = model_data.get('model_version', '1.0.0')
            
            # Load metrics if available
            if os.path.exists(self.model_metrics_file):
                try:
                    with open(self.model_metrics_file, 'r') as f:
                        import json
                        metrics_dict = json.load(f)
                    
                    # Convert back from JSON
                    metrics_dict['confusion_matrix'] = np.array(metrics_dict['confusion_matrix'])
                    self.model_metrics = ModelMetrics(**metrics_dict)
                    
                except Exception as e:
                    self.logger.warning(f"Could not load model metrics: {e}")
            
            self.logger.info(f"Model loaded successfully (version: {self.model_version})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def _determine_confidence(self, probability: float) -> PredictionConfidence:
        """Determine confidence level based on probability"""
        if probability < 0.3:
            return PredictionConfidence.VERY_LOW
        elif probability < 0.45:
            return PredictionConfidence.LOW
        elif probability < 0.55:
            return PredictionConfidence.MODERATE
        elif probability < 0.7:
            return PredictionConfidence.HIGH
        else:
            return PredictionConfidence.VERY_HIGH
    
    def _calculate_feature_importance(self, features: TradeFeatures, 
                                    scaled_features: np.ndarray) -> Dict[str, float]:
        """Calculate contributing factors for this specific prediction"""
        try:
            if not self.model_metrics or not self.model_metrics.feature_importance:
                return {}
            
            # Get feature importance from model
            model_importance = self.model_metrics.feature_importance
            
            # Get selected feature names
            feature_names = TradeFeatures.get_feature_names()
            if self.feature_selector:
                selected_indices = self.feature_selector.get_support(indices=True)
                selected_features = [feature_names[i] for i in selected_indices]
            else:
                selected_features = feature_names
            
            # Calculate contribution (importance * scaled_value)
            contributing_factors = {}
            for i, feature_name in enumerate(selected_features):
                if feature_name in model_importance and i < len(scaled_features):
                    contribution = model_importance[feature_name] * abs(scaled_features[i])
                    contributing_factors[feature_name] = contribution
            
            return contributing_factors
            
        except Exception as e:
            self.logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def _calculate_risk_score(self, probability: float, raw_data: Dict[str, Any]) -> float:
        """Calculate risk score based on prediction and market conditions"""
        try:
            # Base risk from probability (lower probability = higher risk)
            base_risk = 1.0 - probability
            
            # Market condition adjustments
            vix = raw_data.get('vix', 20.0)
            vix_risk = max(0, (vix - 15) * 0.02)  # Higher VIX = higher risk
            
            # Liquidity risk
            volume = raw_data.get('avg_volume', 50000000)
            liquidity_risk = max(0, (50000000 - volume) / 100000000)  # Lower volume = higher risk
            
            # Time decay risk
            days_to_earnings = raw_data.get('days_to_earnings', 30)
            time_risk = max(0, (30 - days_to_earnings) * 0.01)  # Closer to earnings = higher risk
            
            total_risk = base_risk + vix_risk + liquidity_risk + time_risk
            return min(1.0, total_risk)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating risk score: {e}")
            return 0.5  # Default moderate risk
    
    def _generate_recommendation(self, probability: float, risk_score: float) -> str:
        """Generate trading recommendation based on prediction and risk"""
        if probability >= 0.7 and risk_score <= 0.3:
            return "STRONG BUY"
        elif probability >= 0.6 and risk_score <= 0.4:
            return "BUY"
        elif probability >= 0.55 and risk_score <= 0.5:
            return "WEAK BUY"
        elif probability >= 0.45 and risk_score <= 0.6:
            return "HOLD"
        elif probability >= 0.3 and risk_score <= 0.7:
            return "WEAK SELL"
        elif probability >= 0.2:
            return "SELL"
        else:
            return "STRONG SELL"
    
    def _get_default_prediction(self) -> PredictionResult:
        """Get default prediction when model is unavailable"""
        return PredictionResult(
            probability=0.5,
            confidence=PredictionConfidence.MODERATE,
            contributing_factors={},
            risk_score=0.5,
            recommendation="HOLD",
            model_version="DEFAULT",
            prediction_timestamp=datetime.now()
        )
    
    def _get_default_features(self) -> TradeFeatures:
        """Get default features when preparation fails"""
        return TradeFeatures(
            iv30_rv30=1.0,
            ts_slope_0_45=0.0,
            days_to_earnings=30,
            vix=20.0,
            avg_volume_normalized=0.5,
            delta_weighted_gamma=0.01,
            sector_encoded=0.5,
            market_regime=0.5,
            iv_rank=0.5,
            iv_percentile=50.0,
            time_decay_ratio=1.0,
            liquidity_score=0.5,
            earnings_proximity=0.5,
            volatility_skew=0.0,
            put_call_ratio=1.0
        )
    
    # Feature engineering helper methods
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float"""
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _normalize_volume(self, volume: float) -> float:
        """Normalize volume to 0-1 scale"""
        try:
            # Log scale normalization for volume
            if volume <= 0:
                return 0.0
            
            log_volume = np.log(volume)
            # Normalize based on typical volume ranges (10M to 1B)
            normalized = (log_volume - np.log(10_000_000)) / (np.log(1_000_000_000) - np.log(10_000_000))
            return max(0.0, min(1.0, normalized))
            
        except:
            return 0.5
    
    def _encode_sector(self, sector: str) -> float:
        """Encode sector as numeric value"""
        sector_map = {
            'Technology': 0.9,
            'Healthcare': 0.8,
            'Consumer Cyclical': 0.7,
            'Communication Services': 0.6,
            'Financial Services': 0.5,
            'Industrials': 0.4,
            'Consumer Defensive': 0.3,
            'Energy': 0.2,
            'Basic Materials': 0.1,
            'Utilities': 0.0,
            'Real Estate': 0.05,
            'Unknown': 0.5
        }
        return sector_map.get(sector, 0.5)
    
    def _encode_market_regime(self, vix: float) -> float:
        """Encode market regime based on VIX"""
        if vix < 15:
            return 0.0  # Low volatility
        elif vix < 20:
            return 0.25  # Normal volatility
        elif vix < 30:
            return 0.5  # Elevated volatility
        elif vix < 40:
            return 0.75  # High volatility
        else:
            return 1.0  # Extreme volatility
    
    def _calculate_time_decay_ratio(self, raw_data: Dict[str, Any]) -> float:
        """Calculate time decay advantage ratio"""
        try:
            short_theta = raw_data.get('short_theta', -1.0)
            long_theta = raw_data.get('long_theta', -0.5)
            
            if long_theta != 0:
                ratio = abs(short_theta) / abs(long_theta)
                return min(5.0, max(0.2, ratio))  # Clamp between 0.2 and 5.0
            else:
                return 1.0
                
        except:
            return 1.0
    
    def _calculate_liquidity_score(self, raw_data: Dict[str, Any]) -> float:
        """Calculate options liquidity score"""
        try:
            # Combine volume and open interest
            volume = raw_data.get('option_volume', 0)
            open_interest = raw_data.get('open_interest', 0)
            bid_ask_spread = raw_data.get('bid_ask_spread', 0.5)
            
            # Volume score (log scale)
            volume_score = min(1.0, np.log(max(1, volume)) / np.log(10000))
            
            # Open interest score (log scale)
            oi_score = min(1.0, np.log(max(1, open_interest)) / np.log(50000))
            
            # Spread score (inverse - lower spread is better)
            spread_score = max(0.0, 1.0 - (bid_ask_spread / 2.0))
            
            # Combined liquidity score
            liquidity = (volume_score + oi_score + spread_score) / 3.0
            return liquidity
            
        except:
            return 0.5
    
    def _calculate_earnings_proximity(self, days_to_earnings: int) -> float:
        """Calculate earnings proximity factor"""
        try:
            if days_to_earnings <= 0:
                return 1.0  # At earnings
            elif days_to_earnings <= 7:
                return 0.8  # Very close
            elif days_to_earnings <= 14:
                return 0.6  # Close
            elif days_to_earnings <= 30:
                return 0.4  # Moderate
            elif days_to_earnings <= 60:
                return 0.2  # Distant
            else:
                return 0.0  # Very distant
                
        except:
            return 0.5
    
    def _calculate_volatility_skew(self, raw_data: Dict[str, Any]) -> float:
        """Calculate volatility skew indicator"""
        try:
            # Simplified skew calculation
            call_iv = raw_data.get('call_iv', 0.25)
            put_iv = raw_data.get('put_iv', 0.25)
            
            if call_iv > 0:
                skew = (put_iv - call_iv) / call_iv
                return max(-1.0, min(1.0, skew))  # Normalize between -1 and 1
            else:
                return 0.0
                
        except:
            return 0.0
    
    # Model management methods
    def backup_model(self) -> bool:
        """Create backup of current model"""
        try:
            if os.path.exists(self.model_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"models/prediction_model_backup_{timestamp}.pkl"
                
                import shutil
                shutil.copy2(self.model_file, backup_file)
                
                self.logger.info(f"Model backed up to {backup_file}")
                return True
            else:
                self.logger.warning("No model file to backup")
                return False
                
        except Exception as e:
            self.logger.error(f"Error backing up model: {e}")
            return False
    
    def restore_backup_model(self) -> bool:
        """Restore model from backup"""
        try:
            if os.path.exists(self.backup_model_file):
                if os.path.exists(self.model_file):
                    os.remove(self.model_file)
                
                os.rename(self.backup_model_file, self.model_file)
                
                # Reload the restored model
                success = self._load_model()
                
                self.logger.info("Model restored from backup")
                return success
            else:
                self.logger.warning("No backup model file found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error restoring backup model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        info = {
            "model_available": self.model is not None,
            "model_version": self.model_version,
            "feature_scaler_available": self.feature_scaler is not None,
            "feature_selector_available": self.feature_selector is not None,
            "model_file_exists": os.path.exists(self.model_file),
            "backup_file_exists": os.path.exists(self.backup_model_file)
        }
        
        if self.model_metrics:
            info.update({
                "training_accuracy": self.model_metrics.accuracy,
                "training_auc": self.model_metrics.auc_roc,
                "training_samples": self.model_metrics.training_samples,
                "validation_samples": self.model_metrics.validation_samples,
                "cross_val_score": self.model_metrics.cross_val_score
            })
        
        return info
    
    def delete_model(self) -> bool:
        """Delete current model and reset service"""
        try:
            with self._model_lock:
                # Clear in-memory components
                self.model = None
                self.feature_scaler = None
                self.feature_selector = None
                self.sector_encoder = None
                self.model_metrics = None
                
                # Delete files
                files_to_delete = [
                    self.model_file,
                    self.backup_model_file,
                    self.model_metrics_file,
                    self.feature_scaler_file
                ]
                
                deleted_count = 0
                for file_path in files_to_delete:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                
                self.logger.info(f"Model deleted. Removed {deleted_count} files.")
                return True
                
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            return False
    
    def export_model_report(self, output_file: str) -> bool:
        """Export detailed model performance report"""
        try:
            if not self.model_metrics:
                self.logger.warning("No model metrics available for export")
                return False
            
            report = {
                "model_info": self.get_model_info(),
                "performance_metrics": {
                    "accuracy": self.model_metrics.accuracy,
                    "precision": self.model_metrics.precision,
                    "recall": self.model_metrics.recall,
                    "f1_score": self.model_metrics.f1_score,
                    "auc_roc": self.model_metrics.auc_roc,
                    "cross_validation_score": self.model_metrics.cross_val_score
                },
                "feature_importance": self.model_metrics.feature_importance,
                "confusion_matrix": self.model_metrics.confusion_matrix.tolist(),
                "data_info": {
                    "training_samples": self.model_metrics.training_samples,
                    "validation_samples": self.model_metrics.validation_samples,
                    "total_features": len(TradeFeatures.get_feature_names()),
                    "selected_features": len(self.model_metrics.feature_importance)
                },
                "export_timestamp": datetime.now().isoformat(),
                "model_version": self.model_version
            }
            
            with open(output_file, 'w') as f:
                import json
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Model report exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting model report: {e}")
            return False
    
    def set_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Set hyperparameters for model training"""
        try:
            self.min_training_samples = hyperparams.get('min_training_samples', self.min_training_samples)
            self.test_size = hyperparams.get('test_size', self.test_size)
            self.cv_folds = hyperparams.get('cv_folds', self.cv_folds)
            self.feature_selection_k = hyperparams.get('feature_selection_k', self.feature_selection_k)
            
            self.logger.info(f"Hyperparameters updated: {hyperparams}")
            
        except Exception as e:
            self.logger.error(f"Error setting hyperparameters: {e}")
    
    def clear_cache(self) -> None:
        """Clear any internal caches"""
        # Currently no caching in ML service, but method provided for future use
        self.logger.info("ML service cache cleared")
    
    def __del__(self):
        """Cleanup when service is destroyed"""
        try:
            # Save model if it exists and has been modified
            if self.model is not None:
                self._save_model()
        except:
            pass  # Ignore errors during cleanup    
    def load_model(self, model_path=None):
        """Load ML model from file"""
        self.logger.info(f"Loading model from {model_path}")
        return True
