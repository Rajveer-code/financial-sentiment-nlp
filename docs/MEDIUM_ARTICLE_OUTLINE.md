# Medium Article Outline: "Building a Leak-Free Financial Sentiment Model"

## Article Structure

**Target Length**: 2000-2500 words
**Reading Time**: 8-10 minutes
**Target Audience**: ML engineers, quant researchers, data scientists

---

## Title Options

1. **"How I Built a Leak-Free Financial Sentiment Model (And Why It Matters)"**
2. **"The Hidden Data Leakage Traps in Financial ML (And How to Avoid Them)"**
3. **"From News to Trading Signals: Building a Production-Ready Sentiment Model"**

---

## Introduction (200 words)

### Hook
> "I spent 3 months building a financial sentiment model that achieved 65% accuracy in backtesting. Then I discovered it was completely broken—due to data leakage I didn't even know existed."

### Problem Statement
- Financial ML is riddled with subtle leakage traps
- Most tutorials skip critical validation steps
- Real-world performance often disappoints

### What You'll Learn
- How to detect and prevent data leakage in financial ML
- Proper temporal validation techniques
- Building a production-ready sentiment model
- Lessons from building an end-to-end system

---

## Part 1: The Leakage Problem (400 words)

### 1.1 What is Data Leakage in Financial ML?

**Definition**: Using future information to predict the past

**Common Examples**:
```python
# ❌ LEAKY: Using T+1 price to predict T+1 movement
df['movement'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# ✅ SAFE: Using T price to predict T+1 movement
df['next_day_return'] = df.groupby('ticker')['Close'].pct_change().shift(-1)
df['movement'] = (df['next_day_return'] > 0).astype(int)
```

### 1.2 Why Financial Data is Especially Tricky

- **Temporal dependencies**: Prices are sequential
- **Cross-ticker contamination**: Groupby matters!
- **News alignment**: Weekend news → which trading day?
- **Feature-target alignment**: Features at T predict movement at T+1

### 1.3 The Cost of Leakage

- **Backtest**: 65% accuracy (inflated)
- **Real-world**: 48% accuracy (below random!)
- **Result**: Wasted months, lost credibility

---

## Part 2: Building the System (600 words)

### 2.1 Architecture Overview

**Three Main Components**:
1. NLP Pipeline: News → Sentiment Features
2. Feature Engineering: Sentiment + Technical + Lagged
3. Model: CatBoost Classifier

**Key Design Decisions**:
- Modular pipelines (testable, debuggable)
- Centralized feature schema (FEATURE_SCHEMA.py)
- Temporal validation (walk-forward)

### 2.2 The NLP Pipeline

**Multi-Model Ensemble**:
```python
ensemble = 0.6 * finbert + 0.3 * vader + 0.1 * textblob
```

**Why This Works**:
- FinBERT: Domain-specific (financial text)
- VADER: Fast, rule-based
- TextBlob: Polarity + subjectivity

**Event Classification**:
- Keyword-based (matches training)
- Probability scores (not binary)
- 6 event types: earnings, product, analyst, regulatory, macro, M&A

### 2.3 Feature Engineering

**43 Features Total**:
- 24 sentiment features (ensemble, variance, events, entities)
- 15 technical indicators (RSI, MACD, Bollinger Bands, etc.)
- 4 lagged features (T-1 sentiment, returns, volume, volatility)

**Critical Fix**: VWAP calculation
```python
# ❌ WRONG: Cumulative (diverges from training)
vwap = (price * volume).cumsum() / volume.cumsum()

# ✅ CORRECT: Rolling window (matches training)
vwap = (price * volume).rolling(20).sum() / volume.rolling(20).sum()
```

### 2.4 The Model

**CatBoost Classifier**:
- Handles missing values
- Feature importance (SHAP)
- Probability outputs (for calibration)

**Preprocessing**:
- Feature scaling (StandardScaler)
- Feature order validation (prevents errors)
- Missing value handling

---

## Part 3: Preventing Leakage (500 words)

### 3.1 Target Construction

**The Critical Code**:
```python
# ✅ VERIFIED SAFE
df['next_day_return'] = df.groupby('ticker')['Close'].pct_change().shift(-1)
df['movement'] = (df['next_day_return'] > 0).astype(int)
```

**Why `groupby('ticker')` Matters**:
- Without it: Cross-ticker contamination
- With it: Each ticker's returns calculated independently

**Verification**:
- Unit tests check alignment
- Assertions prevent future mistakes
- Documentation in code

### 3.2 Temporal Validation

**Walk-Forward Validation**:
```python
# Train: 2020-2023
# Test: 2024-07 to 2024-12
# Gap: 1 day (prevents overlap)
```

**Why Not K-Fold?**:
- K-Fold shuffles data (temporal order lost)
- Walk-forward preserves time sequence
- More realistic performance estimate

### 3.3 News-Price Alignment

**The Weekend Problem**:
- News published: Sunday 10 AM
- Which trading day does it affect?
- Answer: Monday features → Tuesday prediction

**Implementation**:
```python
def align_news_to_trading_day(news_date):
    # Weekend/holiday news → next trading day
    return next_business_day(news_date)
```

### 3.4 Feature-Target Alignment

**Golden Rule**: Features at time T predict movement at time T+1

**Verification Checklist**:
- [ ] No T+1 data in features at T
- [ ] News timestamps < prediction timestamps
- [ ] Lagged features use T-1 (not T+1)
- [ ] Technical indicators use only past data

---

## Part 4: Results & Lessons (400 words)

### 4.1 Performance Metrics

**Model Performance**:
- Accuracy: 58% (8% above random)
- Sharpe Ratio: 0.9 (positive risk-adjusted returns)
- Calibration: Brier score 0.24 (well-calibrated)

**Not Spectacular, But Honest**:
- No inflated backtest numbers
- Realistic expectations
- Transparent limitations

### 4.2 What Worked

1. **Sentiment features are predictive**: Top feature in SHAP analysis
2. **Lagged features add value**: Previous day returns matter
3. **Event classification helps**: Earnings events are informative
4. **Proper validation**: Walk-forward prevents overfitting

### 4.3 What Didn't Work

1. **Lagged sentiment placeholder**: Distribution shift in inference
2. **Bear market performance**: Model struggles in downturns
3. **Low news coverage**: Accuracy drops for some tickers

### 4.4 Key Lessons

**Lesson 1: Leakage is Everywhere**
- Check target construction first
- Verify temporal alignment
- Test with unit tests

**Lesson 2: Validation Matters More Than Model**
- Simple model + proper validation > Complex model + leaky validation
- Walk-forward > K-Fold for time series

**Lesson 3: Transparency Builds Trust**
- Document limitations (LIMITATIONS.md)
- Show calibration plots
- Be honest about performance

---

## Part 5: Production Considerations (300 words)

### 5.1 Reproducibility

**Random Seeds**:
```python
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

**Feature Schema**:
- Centralized in FEATURE_SCHEMA.py
- Version tracking
- Validation functions

### 5.2 Monitoring

**What to Track**:
- Prediction accuracy over time
- Calibration drift
- Feature distributions
- Model confidence

### 5.3 Limitations

**Known Issues**:
- Lagged sentiment uses placeholder
- Limited to specific time period
- Transaction costs not fully modeled

**Future Work**:
- Prediction storage for lagged features
- Regime detection
- Expanded dataset

---

## Conclusion (200 words)

### Takeaways

1. **Data leakage is subtle**: Always verify target construction
2. **Temporal validation is critical**: Use walk-forward, not K-Fold
3. **Transparency matters**: Document limitations honestly
4. **Simple + correct > Complex + leaky**

### Final Thoughts

> "Building a leak-free financial ML model isn't about finding the perfect algorithm. It's about rigorous validation, transparent documentation, and honest evaluation. The model I built isn't perfect, but it's trustworthy—and that's what matters in production."

### Call to Action

- Check out the code: [GitHub link]
- Read the full documentation: [Docs link]
- Share your own leakage stories: [Twitter/Discord]

---

## Code Snippets to Include

### 1. Leak-Free Target Construction
```python
# ✅ SAFE
df['next_day_return'] = df.groupby('ticker')['Close'].pct_change().shift(-1)
df['movement'] = (df['next_day_return'] > 0).astype(int)
```

### 2. Walk-Forward Validation
```python
def create_walk_forward_splits(df, train_days=252, test_days=21):
    splits = []
    start = 0
    while start + train_days + test_days <= len(df):
        train_idx = df.index[start:start + train_days]
        test_idx = df.index[start + train_days:start + train_days + test_days]
        splits.append((train_idx, test_idx))
        start += test_days
    return splits
```

### 3. Feature Order Validation
```python
# Prevent prediction errors from column misalignment
assert list(X_pred.columns) == list(model.feature_names_), \
    "Feature order mismatch!"
```

---

## Visualizations to Include

1. **Architecture Diagram** (Mermaid)
2. **Calibration Plot** (showing well-calibrated probabilities)
3. **Feature Importance** (SHAP summary plot)
4. **Temporal Split Timeline** (train/val/test boundaries)
5. **Performance by Regime** (bar chart)

---

## SEO Keywords

- financial machine learning
- data leakage prevention
- sentiment analysis
- time series validation
- walk-forward validation
- financial NLP
- trading signals
- CatBoost
- FinBERT

---

## Social Media Hooks

**Twitter Thread**:
1. "I built a financial sentiment model with 65% accuracy. Then I discovered it was completely broken due to data leakage. Here's what I learned 🧵"

**LinkedIn Post**:
"Most financial ML models have hidden data leakage. Here's how to detect and prevent it—with code examples and real results."

---

## Estimated Writing Time

- **Outline**: ✅ Done
- **Draft**: 4-6 hours
- **Code snippets**: 1 hour
- **Visualizations**: 2 hours
- **Editing**: 2 hours
- **Total**: 9-11 hours

---

## Next Steps

1. Write first draft (focus on Part 1-3)
2. Add code snippets and visualizations
3. Get feedback from ML community
4. Publish on Medium
5. Share on Twitter/LinkedIn

---

**Last Updated**: 2025-01-XX
**Status**: Ready for writing

