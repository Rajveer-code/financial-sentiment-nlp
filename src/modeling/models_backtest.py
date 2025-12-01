"""
models_backtest.py
===========================

Model inference engine aligned with unified feature pipeline.

Changes from original:
- Removed fetch_technical_features() (now in feature_pipeline.py)
- Removed prepare_model_features() (now in feature_pipeline.py)
- Uses assemble_model_features() for complete feature assembly
- Simplified predict() to accept pre-assembled features

Author: Rajveer Singh Pall
"""

import sys
import pickle
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# PATH SETUP
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import log_info, log_error, log_warning
from src.feature_engineering.feature_pipeline import (
    assemble_model_features,
    create_model_input_dataframe,
)
from FEATURE_SCHEMA import MODEL_FEATURES

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

MODELS_DIR = PROJECT_ROOT / "models"
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost_best.pkl"
SCALER_PATH = MODELS_DIR / "scaler_ensemble.pkl"

DEFAULT_CONFIDENCE_THRESHOLD = 0.55

# -------------------------------------------------------------------------
# DATA CLASSES
# -------------------------------------------------------------------------


@dataclass
class PredictionResult:
    """Container for model prediction results."""

    ticker: str
    date: datetime
    prediction: int  # 0 = DOWN, 1 = UP
    probability: float  # Probability of UP class
    confidence: float  # max(prob_up, prob_down)
    signal: str  # "BUY", "SELL", "HOLD"
    features: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "ticker": self.ticker,
            "date": self.date,
            "prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "signal": self.signal,
        }


# -------------------------------------------------------------------------
# MODEL LOADER
# -------------------------------------------------------------------------


class ModelLoader:
    """Handles loading and caching of trained models."""

    def __init__(
        self,
        model_path: Path = CATBOOST_MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

    # ---------------------------------------------------------------------

    def load_model(self) -> Any:
        """Load CatBoost model from pickle."""
        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            log_info(f"Loaded CatBoost model from {self.model_path.name}", "MODEL")

        return self.model

    # ---------------------------------------------------------------------

    def load_scaler(self) -> Optional[Any]:
        """Load feature scaler from pickle, if available."""
        if self.scaler is None:
            if not self.scaler_path.exists():
                log_warning(
                    f"Scaler not found at {self.scaler_path}, using raw features",
                    "MODEL",
                )
                return None

            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)

            log_info(f"Loaded scaler from {self.scaler_path.name}", "MODEL")

        return self.scaler


# -------------------------------------------------------------------------
# PREDICTION ENGINE
# -------------------------------------------------------------------------


class PredictionEngine:
    """Handles model inference and signal generation."""

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        model_version: str = "v1.0",
        feature_schema_version: str = "20250116",
    ):
        self.loader = ModelLoader()
        self.model = self.loader.load_model()

        # ==========================================================
        # STORE FEATURE NAMES SEPARATELY (don't modify CatBoost model)
        # ==========================================================
        # CatBoost models from pickle are immutable, so we store our expected
        # feature names in a separate instance variable for alignment
        self.expected_feature_names = None
        
        if hasattr(self.model, "feature_names_"):
            model_feature_names = list(self.model.feature_names_)
            
            # Check if model has numeric feature names ("0","1","2",...)
            if all(str(f).isdigit() for f in model_feature_names):
                if len(model_feature_names) == len(MODEL_FEATURES):
                    log_warning(
                        f"Model has numeric feature names. Will map to MODEL_FEATURES ({len(MODEL_FEATURES)} features)",
                        "MODEL"
                    )
                    self.expected_feature_names = MODEL_FEATURES
                else:
                    raise ValueError(
                        f"Feature count mismatch! Model expects {len(model_feature_names)}, "
                        f"Pipeline provides {len(MODEL_FEATURES)}"
                    )
            else:
                # Model has real feature names
                self.expected_feature_names = model_feature_names
        else:
            # Fallback: use MODEL_FEATURES
            self.expected_feature_names = MODEL_FEATURES

        self.scaler = self.loader.load_scaler()
        self.confidence_threshold = confidence_threshold
        self.model_version = model_version
        self.feature_schema_version = feature_schema_version

        log_info(
            f"PredictionEngine initialized: model={model_version}, "
            f"schema={feature_schema_version}, features={len(MODEL_FEATURES)}",
            "PREDICT",
        )

    # --------------------------------------------------------------------- 
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Return engine status information for debugging."""
        return {
            "model_loaded": self.model is not None,
            "model_has_feature_names": hasattr(self.model, "feature_names_") if self.model else False,
            "model_feature_count": len(self.model.feature_names_) if hasattr(self.model, "feature_names_") else 0,
            "expected_features": len(self.expected_feature_names) if self.expected_feature_names else 0,
            "scaler_available": self.scaler is not None,
            "confidence_threshold": self.confidence_threshold,
            "schema_version": self.feature_schema_version,
        }

    # --------------------------------------------------------------------- 

    def _get_probabilities(self, X: pd.DataFrame) -> Tuple[float, float]:
        """Return (prob_down, prob_up) robustly."""
        prob = self.model.predict_proba(X)[0]

        if hasattr(self.model, "classes_"):
            classes = list(self.model.classes_)
            if 1 in classes and 0 in classes:
                idx_up = classes.index(1)
                idx_down = classes.index(0)
            else:
                idx_down, idx_up = 0, 1
        else:
            idx_down, idx_up = 0, 1

        return float(prob[idx_down]), float(prob[idx_up])

    # ---------------------------------------------------------------------

    def predict(
        self,
        ticker: str,
        sentiment_features: Dict[str, float],
        date: Optional[datetime] = None,
    ) -> PredictionResult:
        """Generate prediction for a single ticker."""

        if date is None:
            date = datetime.now()

        log_info(f"Predicting for {ticker} on {date.date()}", "PREDICT")

        # Assemble complete features
        complete_features = assemble_model_features(ticker, sentiment_features)

        # Build DataFrame
        X = create_model_input_dataframe(
            complete_features, ticker, date.strftime("%Y-%m-%d")
        )

        X_pred = X[MODEL_FEATURES].copy()

        # ==========================================================
        # COLUMN ALIGNMENT: Match model's expected feature names
        # ==========================================================
        if hasattr(self.model, "feature_names_"):
            model_features = list(self.model.feature_names_)
            
            # CASE 1: Model expects numeric features ["0","1","2",...]
            if all(str(f).isdigit() for f in model_features):
                log_info("Mapping columns to numeric indices for model", "PREDICT")
                
                if len(X_pred.columns) != len(model_features):
                    raise ValueError(
                        f"Feature count mismatch: model expects {len(model_features)} features, "
                        f"pipeline provided {len(X_pred.columns)}"
                    )
                
                # Rename to numeric indices
                numeric_names = [str(i) for i in range(len(X_pred.columns))]
                X_pred.columns = numeric_names
            
            # CASE 2: Model expects named features
            else:
                if list(X_pred.columns) != model_features:
                    log_info("Reordering columns to match model feature order", "PREDICT")
                    # Reorder X_pred to match model feature order
                    X_pred = X_pred[model_features]

        # ==========================================================
        # APPLY SCALING IF AVAILABLE
        # ==========================================================
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_pred)
            X_pred = pd.DataFrame(X_scaled, columns=X_pred.columns)

        # ==========================================================
        # MAKE PREDICTION
        # ==========================================================
        prediction = int(self.model.predict(X_pred)[0])
        prob_down, prob_up = self._get_probabilities(X_pred)
        confidence = max(prob_up, prob_down)

        # Signal
        if confidence >= self.confidence_threshold:
            signal = "BUY" if prediction == 1 else "SELL"
        else:
            signal = "HOLD"

        log_info(
            f"✓ {ticker}: {signal} (prob_up={prob_up:.3f}, conf={confidence:.3f})",
            "PREDICT",
        )

        return PredictionResult(
            ticker=ticker,
            date=date,
            prediction=prediction,
            probability=prob_up,
            confidence=confidence,
            signal=signal,
            features=complete_features,
        )

    # ---------------------------------------------------------------------

    def predict_batch(
        self,
        ticker_sentiment_dict: Dict[str, Dict[str, float]],
        date: Optional[datetime] = None,
    ) -> List[PredictionResult]:
        """Predict multiple tickers."""
        results: List[PredictionResult] = []

        for ticker, features in ticker_sentiment_dict.items():
            try:
                results.append(self.predict(ticker, features, date))
            except Exception as e:
                log_error(f"Prediction failed for {ticker}: {e}", "PREDICT")

        return results


# -------------------------------------------------------------------------
# CONVENIENCE WRAPPER
# -------------------------------------------------------------------------


def quick_predict(
    ticker: str,
    sentiment_features: Dict[str, float],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> PredictionResult:
    """Quick single-ticker prediction."""
    engine = PredictionEngine(confidence_threshold=confidence_threshold)
    return engine.predict(ticker, sentiment_features)


# -------------------------------------------------------------------------
# BACKTESTING (STUB)
# -------------------------------------------------------------------------

@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""

    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int

    def to_dict(self) -> Dict:
        return {
            "Total Return (%)": f"{self.total_return:.2f}",
            "Annualized Return (%)": f"{self.annualized_return:.2f}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Max Drawdown (%)": f"{self.max_drawdown:.2f}",
            "Win Rate (%)": f"{self.win_rate:.2f}",
            "Total Trades": self.total_trades,
        }


class BacktestEngine:
    """Portfolio backtesting engine with temporal validation."""

    def __init__(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
        train_start: Optional[datetime] = None,
        train_end: Optional[datetime] = None,
        test_start: Optional[datetime] = None,
        test_end: Optional[datetime] = None,
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        # Defaults
        train_start = train_start or start_date
        train_end = train_end or datetime(start_date.year + 3, start_date.month, start_date.day)
        test_start = test_start or (train_end + pd.Timedelta(days=1))
        test_end = test_end or end_date

        assert train_end < test_start, (
            "Temporal leakage detected: train_end must be < test_start"
        )

        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end

        log_info(
            f"Backtest initialized: Train [{train_start.date()} → {train_end.date()}], "
            f"Test [{test_start.date()} → {test_end.date()}]",
            "BACKTEST",
        )

    # ---------------------------------------------------------------------

    def create_walk_forward_splits(
        self,
        df: pd.DataFrame,
        train_days: int = 252,
        test_days: int = 21,
        step_days: int = 21,
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Create walk-forward validation splits."""
        splits = []
        start = 0

        while start + train_days + test_days <= len(df):
            train_idx = df.index[start : start + train_days]
            test_idx = df.index[start + train_days : start + train_days + test_days]
            splits.append((train_idx, test_idx))
            start += step_days

        log_info(f"Created {len(splits)} walk-forward splits", "BACKTEST")
        return splits

    # ---------------------------------------------------------------------

    def run_ml_strategy(self, predictions: pd.DataFrame) -> Tuple[pd.DataFrame, BacktestMetrics]:
        """Run ML strategy backtest (stub)."""
        test_predictions = predictions[
            (predictions["date"] >= self.test_start)
            & (predictions["date"] <= self.test_end)
        ]

        if test_predictions.empty:
            log_warning("No predictions in test period, returning stub.", "BACKTEST")
            return self._create_stub_results()

        log_warning("Full backtest requires price data integration.", "BACKTEST")
        return self._create_stub_results()

    # ---------------------------------------------------------------------

    def _create_stub_results(self) -> Tuple[pd.DataFrame, BacktestMetrics]:
        """Return stub backtest results."""
        dates = pd.date_range(self.test_start, self.test_end, freq="D")
        equity_curve = pd.DataFrame({"date": dates, "equity": [100000] * len(dates)})

        metrics = BacktestMetrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
        )

        return equity_curve, metrics

    # ---------------------------------------------------------------------

    def run_buy_and_hold(self) -> Tuple[pd.DataFrame, BacktestMetrics]:
        """Stub for buy-and-hold strategy."""
        return self._create_stub_results()


# -------------------------------------------------------------------------
# STRATEGY COMPARISON
# -------------------------------------------------------------------------

def compare_strategies(
    tickers: List[str],
    predictions: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compare ML strategy vs Buy & Hold (stub)."""
    engine = BacktestEngine(tickers, start_date, end_date)

    ml_equity, ml_metrics = engine.run_ml_strategy(predictions)
    bh_equity, bh_metrics = engine.run_buy_and_hold()

    return {
        "ml_equity": ml_equity,
        "ml_metrics": ml_metrics,
        "bh_equity": bh_equity,
        "bh_metrics": bh_metrics,
    }


# -------------------------------------------------------------------------
# MODULE TEST
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing models_backtest.py...")
    print("=" * 60)

    # Test 1 — Model loader
    try:
        loader = ModelLoader()
        _ = loader.load_model()
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Model loading failed: {e}\n")

    # Test 2 — Single prediction
    try:
        mock_sentiment = {
            "finbert_sentiment_score_mean": 0.35,
            "vader_sentiment_score_mean": 0.28,
            "textblob_sentiment_score_mean": 0.32,
            "ensemble_sentiment_mean": 0.317,
            "sentiment_variance_mean": 0.001,
            "model_consensus_mean": 0.95,
            "ensemble_sentiment_max": 0.45,
            "ensemble_sentiment_min": 0.20,
            "ensemble_sentiment_std": 0.08,
            "confidence_mean": 0.65,
            "num_headlines": 12,
            "headline_length_avg": 85.5,
            "sentiment_earnings": 0.40,
            "sentiment_product": 0.30,
            "sentiment_analyst": 0.25,
            "count_positive_earnings": 3,
            "count_negative_regulatory": 0,
            "has_macroeconomic_news": 0,
            "ceo_mention_count": 2,
            "ceo_sentiment": 0.38,
            "competitor_mention_count": 1,
            "entity_density": 1.5,
            "entity_sentiment_gap": 0.05,
        }

        result = quick_predict("AAPL", mock_sentiment)

        print("✓ Prediction generated:")
        print(f"   Signal: {result.signal}")
        print(f"   Probability (UP): {result.probability:.3f}")
        print(f"   Confidence: {result.confidence:.3f}\n")

    except Exception as e:
        print(f"✗ Prediction failed: {e}\n")

    print("=" * 60)
    print("✓ models_backtest.py test complete.")
