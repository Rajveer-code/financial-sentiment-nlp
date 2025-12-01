"""
test_prediction_engine_fix.py
=============================
Comprehensive test to verify the PredictionEngine fixes.

Tests:
1. Engine initialization without attribute errors
2. Feature assembly with sentiment features
3. Complete prediction pipeline
4. Error handling and logging

Author: Test Suite
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.modeling.models_backtest import PredictionEngine
from src.feature_engineering.nlp_pipeline import generate_sentiment_features
from src.utils.utils import safe_json_load, log_info, log_error
from FEATURE_SCHEMA import MODEL_FEATURES, normalize_feature_dict

# ============================================================
# TEST CONFIGURATION
# ============================================================

TICKER_FILE = PROJECT_ROOT / "config" / "tickers.json"
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================
# TEST 1: Engine Initialization
# ============================================================

def test_engine_initialization():
    """Test that PredictionEngine initializes without attribute errors."""
    print("\n" + "="*70)
    print("TEST 1: Engine Initialization")
    print("="*70)
    
    try:
        print("Loading PredictionEngine...")
        engine = PredictionEngine(confidence_threshold=0.55)
        
        # Get status
        status = engine.get_engine_status()
        print("\n✅ Engine initialized successfully!")
        print(f"   - Model loaded: {status['model_loaded']}")
        print(f"   - Model has feature names: {status['model_has_feature_names']}")
        print(f"   - Expected features: {status['expected_features']}")
        print(f"   - Scaler available: {status['scaler_available']}")
        
        return True, engine
    except Exception as e:
        print(f"❌ Engine initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


# ============================================================
# TEST 2: Mock Sentiment Features
# ============================================================

def test_mock_sentiment_features():
    """Test with mock sentiment features."""
    print("\n" + "="*70)
    print("TEST 2: Mock Sentiment Features")
    print("="*70)
    
    # Create realistic mock sentiment features
    mock_features = {
        # Sentiment features (23)
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
    
    print(f"✅ Created mock sentiment features ({len(mock_features)} features)")
    
    # Normalize
    normalized = normalize_feature_dict(mock_features)
    print(f"✅ Normalized to {len(normalized)} features (should be {len(MODEL_FEATURES)})")
    
    if len(normalized) != len(MODEL_FEATURES):
        print(f"❌ Feature count mismatch!")
        return False
    
    return True, normalized


# ============================================================
# TEST 3: Feature Assembly
# ============================================================

def test_feature_assembly():
    """Test complete feature assembly."""
    print("\n" + "="*70)
    print("TEST 3: Feature Assembly")
    print("="*70)
    
    try:
        from src.feature_engineering.feature_pipeline import (
            assemble_model_features,
            create_model_input_dataframe
        )
        
        # Use mock sentiment
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
        
        print("Assembling model features for AAPL...")
        complete_features = assemble_model_features("AAPL", mock_sentiment)
        
        print(f"✅ Assembled {len(complete_features)} features")
        print(f"   - Expected: {len(MODEL_FEATURES)}")
        
        if len(complete_features) != len(MODEL_FEATURES):
            print(f"❌ Feature count mismatch!")
            return False
        
        # Create DataFrame
        print("Creating model input DataFrame...")
        df = create_model_input_dataframe(complete_features, "AAPL")
        
        print(f"✅ DataFrame created with shape {df.shape}")
        print(f"   - Columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
        
        return True, (complete_features, df)
        
    except Exception as e:
        print(f"❌ Feature assembly failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


# ============================================================
# TEST 4: End-to-End Prediction
# ============================================================

def test_end_to_end_prediction(engine, features_tuple):
    """Test complete prediction pipeline."""
    print("\n" + "="*70)
    print("TEST 4: End-to-End Prediction")
    print("="*70)
    
    if not engine or not features_tuple:
        print("❌ Skipping - prerequisite test failed")
        return False
    
    complete_features, _ = features_tuple
    
    try:
        print("Making prediction for AAPL...")
        prediction = engine.predict("AAPL", complete_features)
        
        print(f"✅ Prediction successful!")
        print(f"   - Signal: {prediction.signal}")
        print(f"   - Prediction: {'UP' if prediction.prediction == 1 else 'DOWN'}")
        print(f"   - Probability: {prediction.probability:.1%}")
        print(f"   - Confidence: {prediction.confidence:.1%}")
        print(f"   - Features used: {len(prediction.features)}")
        
        return True, prediction
        
    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None


# ============================================================
# MAIN TEST RUNNER
# ============================================================

def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE PREDICTION ENGINE TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Initialization
    success, engine = test_engine_initialization()
    results["initialization"] = success
    
    if not success:
        print("\n❌ CRITICAL: Engine initialization failed. Cannot proceed with other tests.")
        print("\nTest Results Summary:")
        print(f"  - Initialization: ❌ FAILED")
        return
    
    # Test 2: Mock features
    success, features = test_mock_sentiment_features()
    results["mock_features"] = success
    
    if not success:
        print("\n❌ Mock features test failed.")
        print("\nTest Results Summary:")
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  - {test_name}: {status}")
        return
    
    # Test 3: Feature assembly
    success, features_tuple = test_feature_assembly()
    results["feature_assembly"] = success
    
    if not success:
        print("\n❌ Feature assembly test failed.")
        print("\nTest Results Summary:")
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  - {test_name}: {status}")
        return
    
    # Test 4: End-to-end prediction
    success, prediction = test_end_to_end_prediction(engine, features_tuple)
    results["end_to_end_prediction"] = success
    
    # Summary
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  - {test_name}: {status}")
    
    if all_passed:
        print("\n" + "🎉 "*20)
        print("ALL TESTS PASSED! ✅ PredictionEngine is working correctly.")
        print("🎉 "*20)
    else:
        print("\n❌ Some tests failed. Please review the output above.")


if __name__ == "__main__":
    main()
