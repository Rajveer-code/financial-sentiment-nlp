# Limitations & Future Work

This document transparently documents known limitations of the current implementation and areas for future improvement.

---

## 🔴 Critical Limitations

### 1. Lagged Sentiment Feature

**Issue**: During inference, `ensemble_sentiment_mean_lag1` uses a placeholder (0.0) or current sentiment approximation since historical predictions are not stored.

**Impact**: 
- **Training**: Used actual T-1 sentiment values
- **Inference**: Uses 0.0 (neutral) or current sentiment approximation
- **Result**: Distribution shift that may reduce model performance in production

**Current Workaround**: 
- Uses current sentiment as approximation (with warning logged)
- Falls back to 0.0 if no sentiment available

**Future Work**: 
- Implement prediction storage (database or CSV)
- Retrieve historical predictions for lagged features
- Ensure training/inference consistency

**Code Location**: `src/feature_engineering/feature_pipeline.py` → `calculate_lagged_features()`

---

### 2. News-Price Alignment Assumption

**Issue**: News from weekends/holidays is assigned to the next trading day's features.

**Assumption**: 
- News published on Sunday → Contributes to Monday features
- News published on Saturday → Contributes to Monday features
- News published on holiday → Contributes to next trading day features

**Impact**: 
- May introduce slight timing misalignment
- Weekend news may be less relevant by Monday

**Verification**: 
- `published_at` timestamps in `events_classified.csv` may be empty
- Manual verification recommended for critical dates

**Future Work**: 
- Implement explicit trading day calendar
- Add timestamp verification in data pipeline
- Consider time-of-day weighting (morning vs afternoon news)

**Code Location**: `src/feature_engineering/nlp_pipeline.py` → `align_news_to_trading_day()`

---

### 3. Event Classification Method

**Issue**: Event classification uses keyword matching. If training data used a different method (e.g., trained classifier), there may be a mismatch.

**Current Implementation**: 
- Uses keyword-based classification with probability scores
- Matches format of `events_classified.csv` (continuous scores)

**Verification Required**: 
- Confirm training data generation used the same keyword-based approach
- If not, either retrain model or include classifier in inference

**Future Work**: 
- Implement trained event classifier for better accuracy
- Add confidence scores to event classification
- Support fuzzy matching for event keywords

**Code Location**: `src/feature_engineering/nlp_pipeline.py` → `classify_event()`

---

## ⚠️ Moderate Limitations

### 4. Data Coverage

**Current Coverage**:
- Date range: 2025-07-23 to 2025-11-20 (sample data)
- Tickers: 7 tickers
- Total samples: ~600 predictions

**Limitations**:
- Limited to specific time period
- Small number of tickers
- May not cover all market regimes (bull/bear/high-vol)

**Future Work**:
- Expand to 3-5 years of data
- Include 20-30 tickers across sectors
- Ensure coverage of different market regimes

---

### 5. Technical Indicator Calculation

**Issue**: Some indicators (e.g., VWAP) use rolling windows in inference but may have used cumulative in training.

**Current Fix**: 
- VWAP now uses 20-day rolling window (fixed)
- Falls back to cumulative if insufficient data

**Verification**: 
- Ensure all indicators match training calculation method
- Test with edge cases (insufficient data, gaps)

**Future Work**:
- Standardize all indicator calculations
- Add unit tests for each indicator
- Document calculation methods

**Code Location**: `src/feature_engineering/feature_pipeline.py` → `fetch_technical_features()`

---

### 6. Model Complexity

**Issue**: CatBoost model parameters not documented.

**Current State**:
- Model file exists (`models/catboost_best.pkl`)
- Hyperparameters not explicitly logged

**Future Work**:
- Document hyperparameters in model metadata
- Add model version tracking
- Include training metrics in model file

---

## 📊 Known Issues

### 7. Missing Features in Current Model

**Issue**: Additional high-impact features are calculated but not included in current model:
- `volume_ratio`: Current volume / 20-day average
- `price_momentum_5d`: 5-day return
- `volatility_regime`: Current volatility / 20-day average

**Reason**: Maintain compatibility with existing trained model

**Future Work**:
- Retrain model with additional features
- Compare performance with/without new features
- A/B test in production

**Code Location**: `src/feature_engineering/feature_pipeline.py` → `fetch_technical_features()` (commented out)

---

### 8. Backtest Implementation

**Issue**: `BacktestEngine` is partially implemented (stub).

**Current State**:
- Temporal validation implemented
- Walk-forward splits implemented
- Full backtest logic (position sizing, transaction costs) not implemented

**Future Work**:
- Implement full backtest with:
  - Position sizing logic
  - Transaction costs (commissions, slippage)
  - Execution delay (predict at T, trade at T+1 open)
  - Portfolio rebalancing
- Add benchmark comparison (buy-and-hold, market index)

**Code Location**: `src/modeling/models_backtest.py` → `BacktestEngine.run_ml_strategy()`

---

## 🔧 Technical Debt

### 9. Error Handling

**Current State**: Basic error handling with retry logic

**Future Work**:
- Add comprehensive error handling for all API calls
- Implement circuit breakers for external services
- Add monitoring and alerting

---

### 10. Testing Coverage

**Current State**: 
- End-to-end tests implemented
- Schema validation tests
- Leakage detection tests

**Future Work**:
- Add unit tests for each feature engineering function
- Add integration tests for full pipeline
- Add performance benchmarks
- Add regression tests

---

## 📝 Documentation Gaps

### 11. Architecture Diagram

**Missing**: Visual architecture diagram showing data flow

**Future Work**: 
- Create architecture diagram (draw.io or mermaid)
- Add to README.md
- Document component interactions

---

### 12. API Documentation

**Missing**: Detailed API documentation for all functions

**Future Work**:
- Generate API docs (Sphinx or similar)
- Add usage examples
- Document all parameters and return values

---

## 🎯 Research Limitations

### 13. Reproducibility

**Current State**: 
- Random seeds set (seed=42)
- Feature schema documented
- Data generation script provided

**Limitations**:
- External API data may vary (news, prices)
- Model training not fully automated

**Future Work**:
- Containerize environment (Docker)
- Automate full pipeline (training + inference)
- Version control all dependencies

---

### 14. Evaluation Metrics

**Current State**: 
- Basic metrics (accuracy, Sharpe ratio)
- Calibration plots implemented
- Per-ticker breakdown implemented

**Future Work**:
- Add more advanced metrics (information ratio, Sortino ratio)
- Implement regime-specific evaluation
- Add statistical significance tests

---

## ✅ What's Working Well

1. **Leak-Free Implementation**: Target construction verified, temporal validation enforced
2. **Reproducibility**: Random seeds, feature schema, deterministic results
3. **Code Quality**: Type hints, docstrings, modular design
4. **Testing**: Comprehensive test suite with leakage detection
5. **Documentation**: Transparent about limitations, well-documented code

---

## 📋 Summary

**Critical Issues**: 3 (lagged sentiment, news alignment, event classification)
**Moderate Issues**: 3 (data coverage, technical indicators, model complexity)
**Known Issues**: 2 (missing features, backtest stub)
**Technical Debt**: 2 (error handling, testing)
**Documentation**: 2 (architecture, API docs)
**Research**: 2 (reproducibility, evaluation)

**Total**: 14 areas for improvement

**Priority**: Address critical limitations first, then moderate issues, then technical debt.

---

**Last Updated**: 2025-01-XX
**Status**: Transparent documentation of all known limitations

