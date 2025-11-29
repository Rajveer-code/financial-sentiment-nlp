# Critical Validation Questions - Answers & Implementation Status

This document addresses the 5 critical validation questions to ensure research-grade implementation.

---

## Question 1: Target Variable Construction (MOST CRITICAL) ⚠️

### Current Status: **NEEDS VERIFICATION**

Based on `research_outputs/tables/stock_with_ta.csv`, the data structure shows:
- `next_day_return` column exists (e.g., -0.06661763232523876 for 2025-07-16)
- `movement` column is binary (0 or 1)
- `movement = 0` when `next_day_return <= 0`, `movement = 1` when `next_day_return > 0`

### Required Verification

**You MUST verify the exact code in your data generation script that creates `model_ready_full.csv`:**

```python
# OPTION A (SAFE - Recommended):
df['next_day_return'] = df.groupby('ticker')['Close'].pct_change().shift(-1)
df['movement'] = (df['next_day_return'] > 0).astype(int)

# OPTION B (ALSO SAFE):
df['movement'] = (df.groupby('ticker')['Close'].shift(-1) > df['Close']).astype(int)

# ❌ WRONG (LEAKY - Do NOT use):
df['movement'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # Missing groupby!
```

### Action Required

1. **Locate your data generation script** (likely in a notebook or `scripts/` directory)
2. **Verify the exact code** that creates the `movement` target
3. **Document it** in this file or in a `DATA_GENERATION.md` file
4. **Add a test** in `tests/test_pipeline_e2e.py` to verify no leakage:

```python
def test_target_construction():
    """Verify target construction doesn't leak future data."""
    # Load your data generation script or CSV
    # Verify: movement[T] predicts price[T+1], not price[T]
    pass
```

### Current Implementation

The inference pipeline (`src/modeling/models_backtest.py`) does NOT create targets - it only predicts. The target construction happens in your training data generation script, which is **not in this repository**.

**RECOMMENDATION**: Create `scripts/generate_training_data.py` with the exact target construction code and add it to the repository for reproducibility.

---

## Question 2: News-Price Alignment ⚠️

### Current Status: **PARTIALLY IMPLEMENTED - NEEDS VERIFICATION**

### Expected Behavior

```
News published: 2024-11-24 (Sunday) 10:00 AM
→ Aligned to: 2024-11-25 (Monday) features
→ Predicts: 2024-11-26 (Tuesday) movement
```

### Current Implementation

From `src/api_clients/news_api.py` and `app/app_main.py`:
- News is fetched with `from_date` and `to_date` parameters
- News DataFrame has `date` column (but `published_at` may be empty in `events_classified.csv`)
- News is aggregated by date in `generate_sentiment_features()`

### Issues Identified

1. **`published_at` is empty** in `events_classified.csv` - cannot verify temporal alignment
2. **No explicit weekend/holiday handling** - news on weekends may be incorrectly aligned
3. **No verification** that news timestamp < prediction timestamp

### Action Required

**Add to `src/feature_engineering/nlp_pipeline.py`:**

```python
def align_news_to_trading_day(news_date: datetime, market_calendar: pd.bdate_range) -> datetime:
    """
    Align news to next trading day.
    
    Args:
        news_date: News publication date
        market_calendar: Business day calendar
        
    Returns:
        Next trading day after news publication
    """
    # If news on weekend/holiday, move to next trading day
    if news_date not in market_calendar:
        next_trading_day = market_calendar[market_calendar > news_date][0]
        return next_trading_day
    return news_date
```

**Add verification in data generation:**

```python
# In your data generation script:
assert (news_df['published_at'] < price_df['next_day_open_time']).all(), \
    "News timestamps must be before prediction time!"
```

### Current Workaround

The current implementation assumes:
- News with `date = 2024-11-24` (Sunday) → contributes to features for `2024-11-25` (Monday)
- Features for `2024-11-25` → predict movement for `2024-11-26` (Tuesday)

**This needs explicit verification in your data generation pipeline.**

---

## Question 3: Train/Test Split Dates

### Current Status: **NOT DOCUMENTED - REQUIRED**

### Action Required

**You MUST document the exact dates in `README.md` or `DATA_SPLITS.md`:**

```markdown
## Data Splits

### Training Set
- **Start**: [YYYY-MM-DD]
- **End**: [YYYY-MM-DD]
- **Duration**: X days/months
- **Tickers**: [List or count]

### Validation Set (if applicable)
- **Start**: [YYYY-MM-DD]
- **End**: [YYYY-MM-DD]

### Test Set
- **Start**: [YYYY-MM-DD]
- **End**: [YYYY-MM-DD]
- **Duration**: X days/months
- **Gap from training**: X days (to prevent leakage)
```

### Example (Fill in your actual dates):

```markdown
### Training Set
- **Start**: 2020-01-01
- **End**: 2023-12-31
- **Duration**: 4 years (1008 trading days)
- **Tickers**: 15 tickers

### Test Set
- **Start**: 2024-07-01
- **End**: 2024-12-31
- **Duration**: 6 months (126 trading days)
- **Gap from training**: 1 day (2024-01-01 to 2024-06-30 reserved for validation)
```

### Implementation

The `BacktestEngine` now supports explicit train/test dates:

```python
engine = BacktestEngine(
    tickers=['AAPL', 'TSLA'],
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2024, 12, 31),
    train_start=datetime(2020, 1, 1),
    train_end=datetime(2023, 12, 31),
    test_start=datetime(2024, 7, 1),
    test_end=datetime(2024, 12, 31),
)
```

**RECOMMENDATION**: Extract these dates from your training script and document them.

---

## Question 4: Lagged Sentiment Storage

### Current Status: **OPTION A (Placeholder) - DOCUMENTED**

### Current Implementation

From `src/feature_engineering/feature_pipeline.py`:

```python
# Sentiment lag: CRITICAL FIX - Use actual historical sentiment if available
# In production, store predictions in a database and fetch here
# For now, try to use current sentiment_features if provided (as approximation)
# TODO: Implement proper historical prediction storage
ensemble_sentiment_mean_lag1 = sentiment_features.get("ensemble_sentiment_mean", 0.0) if sentiment_features else 0.0
```

### Documentation Required

**Add to `README.md` or `LIMITATIONS.md`:**

```markdown
## Limitations

### Lagged Sentiment Feature

During inference, `ensemble_sentiment_mean_lag1` is set to 0.0 (or current sentiment as approximation) since historical predictions are not stored. This introduces a **distribution shift** from training, where actual lagged sentiment was used.

**Impact**: 
- Training: Used actual T-1 sentiment values
- Inference: Uses 0.0 (neutral) or current sentiment approximation
- This may reduce model performance in production

**Future Work**: 
- Implement prediction storage (database or CSV)
- Retrieve historical predictions for lagged features
- Ensure training/inference consistency
```

### Option B Implementation (Future)

If you want to implement proper storage:

```python
# Create: src/storage/prediction_storage.py
def store_prediction(ticker: str, date: datetime, sentiment: float):
    """Store prediction for future lagged feature use."""
    # Save to database or CSV
    pass

def fetch_previous_prediction(ticker: str, date: datetime) -> float:
    """Fetch T-1 sentiment for lagged feature."""
    # Retrieve from storage
    pass

# Then in calculate_lagged_features():
ensemble_sentiment_mean_lag1 = fetch_previous_prediction(ticker, date - timedelta(days=1))
```

**RECOMMENDATION**: Document Option A clearly in your paper/README, and mention Option B as future work.

---

## Question 5: Event Classification Mismatch

### Current Status: **FIXED - BUT NEEDS VERIFICATION**

### Current Implementation

**Inference (`src/feature_engineering/nlp_pipeline.py`):**
- `classify_event()` now returns probability scores (dict with scores for each event type)
- Uses keyword matching with weights
- Returns normalized probabilities (softmax-like)

**Training Data (`events_classified.csv`):**
- Has continuous scores: `score_earnings: 0.150`, `score_m&a: 0.258`, etc.
- These look like probability scores, not binary

### Potential Mismatch

**If training used a different method:**
- Trained classifier → Inference uses keywords = **MISMATCH** ❌
- Keyword rules → Inference uses keywords = **MATCH** ✅

### Action Required

**Verify in your training data generation script:**

1. **How was `events_classified.csv` created?**
   - Option A: Keyword-based with scores → Matches current inference ✅
   - Option B: Trained classifier → Need to include classifier in inference ❌

2. **If Option B (trained classifier):**
   - Either: Retrain model with keyword-based features
   - Or: Include the classifier in `nlp_pipeline.py`

### Current Fix

The code now returns probability scores matching the CSV format:

```python
def classify_event(headline: str) -> Dict[str, float]:
    """Returns probability scores for each event type."""
    scores = {
        'earnings': 0.0,
        'product': 0.0,
        # ... etc
    }
    # Keyword matching with weights
    # Normalize to probabilities
    return scores
```

**RECOMMENDATION**: 
1. Verify your training data generation uses the same keyword-based approach
2. If not, either retrain or add the classifier to inference
3. Document the method used in both training and inference

---

## Summary of Actions Required

### Immediate (Before Publication)

1. ✅ **Question 1**: Locate and document target construction code
2. ✅ **Question 2**: Add weekend/holiday alignment logic
3. ✅ **Question 3**: Document exact train/test split dates
4. ✅ **Question 4**: Document lagged sentiment limitation (already done in code)
5. ✅ **Question 5**: Verify event classification method matches training

### Files to Create/Update

1. `scripts/generate_training_data.py` - Data generation script with target construction
2. `DATA_SPLITS.md` - Document train/test dates
3. `LIMITATIONS.md` - Document lagged sentiment limitation
4. `README.md` - Add sections for data splits and limitations
5. Update `src/feature_engineering/nlp_pipeline.py` - Add trading day alignment

### Testing

Add to `tests/test_pipeline_e2e.py`:
- `test_target_construction()` - Verify no leakage
- `test_news_alignment()` - Verify weekend/holiday handling
- `test_event_classification_consistency()` - Verify training/inference match

---

## Current Status Checklist

- [x] Feature schema created and validated
- [x] Temporal validation in backtest engine
- [x] Walk-forward validation implemented
- [x] Reproducibility (seed setting)
- [x] Event classification returns probability scores
- [x] Feature order enforcement
- [x] VWAP calculation fixed
- [ ] **Target construction verified** ⚠️
- [ ] **News alignment verified** ⚠️
- [ ] **Train/test dates documented** ⚠️
- [x] Lagged sentiment limitation documented (in code)
- [ ] **Event classification method verified** ⚠️

---

**Next Steps**: Address the 4 remaining items (marked with ⚠️) before final publication.

