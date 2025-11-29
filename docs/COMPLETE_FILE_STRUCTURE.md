# Complete File Structure with Descriptions

## 📁 Root Directory

```
financial-sentiment-nlp/
│
├── 📄 README.md                              # Main project documentation
├── 📄 LICENSE                                 # MIT License
├── 📄 requirements.txt                        # Python dependencies
├── 📄 FEATURE_SCHEMA.py                      # Central feature schema (42 features)
├── 📄 .gitignore                             # Git ignore rules
├── 📄 __init__.py                            # Root package init
│
├── 📁 app/                                    # Streamlit Application
│   └── app_main.py                           # Main dashboard (live predictions, news, charts)
│
├── 📁 config/                                 # Configuration
│   ├── api_keys.json                         # ⚠️ SENSITIVE - API keys (DO NOT PUSH)
│   ├── api_keys.example.json                 # ✅ Template for API keys
│   └── tickers.json                          # ✅ Ticker metadata (company names, CEOs, competitors)
│
├── 📁 docs/                                   # Documentation
│   ├── ARCHITECTURE.md                       # System architecture with Mermaid diagram
│   ├── PERFORMANCE_METRICS.md               # Performance evaluation summary
│   ├── MEDIUM_ARTICLE_OUTLINE.md            # Article structure for publication
│   ├── FILE_ARCHITECTURE.md                 # File structure documentation
│   ├── GITHUB_EXCLUSION_LIST.md             # Files to exclude from GitHub
│   └── COMPLETE_FILE_STRUCTURE.md           # This file
│
├── 📁 models/                                 # Trained Models
│   ├── catboost_best.pkl                     # ⚠️ LARGE - Trained CatBoost model (2-5 MB)
│   └── scaler_ensemble.pkl                   # ✅ Feature scaler (StandardScaler)
│
├── 📁 notebooks/                              # Jupyter Notebooks
│   └── generate_report.ipynb                 # Reproduces all research outputs
│
├── 📁 research_outputs/                       # Research Results
│   ├── 📁 figures/                           # Visualizations (20+ PNG files)
│   ├── 📁 stats/                             # Statistical analysis results
│   └── 📁 tables/                            # Processed data tables (25+ CSV files)
│
├── 📁 scripts/                                # Data Generation Scripts
│   ├── generate_training_data.py            # ✅ Main data generation (verified leak-free)
│   └── generate_training_data_template.py    # ✅ Template for reference
│
├── 📁 src/                                    # Source Code
│   ├── 📁 api_clients/                       # API Integration
│   ├── 📁 feature_engineering/               # Feature Engineering
│   ├── 📁 modeling/                          # Model Training & Inference
│   └── 📁 utils/                             # Utility Functions
│
├── 📁 tests/                                  # Test Suite
│   ├── test_pipeline_e2e.py                 # ✅ 10 comprehensive tests
│   └── test_api_keys.py                     # API key loading tests
│
└── 📁 Documentation/                         # Additional Documentation
    └── README.md                             # Project documentation
```

---

## 📁 Detailed Structure

### `/app` - Application Layer

```
app/
└── app_main.py                               # Streamlit dashboard
    ├── News fetching (Yahoo, NewsAPI)
    ├── Real-time sentiment analysis
    ├── Live predictions
    ├── Interactive charts (Plotly)
    └── PDF report generation
```

**Purpose**: User-facing application for live predictions and analysis.

---

### `/config` - Configuration

```
config/
├── api_keys.json                             # ⚠️ SENSITIVE - Actual API keys
├── api_keys.example.json                     # ✅ Template (safe to push)
└── tickers.json                              # ✅ Ticker metadata
    └── Company names, CEOs, competitors per ticker
```

**Purpose**: Configuration files for API keys and ticker metadata.

---

### `/docs` - Documentation

```
docs/
├── ARCHITECTURE.md                           # System architecture diagram
├── PERFORMANCE_METRICS.md                   # Performance summary
├── MEDIUM_ARTICLE_OUTLINE.md                # Article structure
├── FILE_ARCHITECTURE.md                     # File structure
├── GITHUB_EXCLUSION_LIST.md                 # Exclusion guide
└── COMPLETE_FILE_STRUCTURE.md               # This file
```

**Purpose**: Comprehensive documentation for users and researchers.

---

### `/models` - Model Artifacts

```
models/
├── catboost_best.pkl                         # ⚠️ LARGE - Trained CatBoost classifier
└── scaler_ensemble.pkl                        # ✅ StandardScaler for features
```

**Purpose**: Trained model files for inference.

**Note**: `catboost_best.pkl` is large (2-5 MB). Consider Git LFS or exclude.

---

### `/notebooks` - Analysis Notebooks

```
notebooks/
└── generate_report.ipynb                     # Report generation
    ├── Dataset coverage analysis
    ├── Sentiment summary tables
    ├── Model performance metrics
    ├── Statistical tests
    ├── ROC/PR curves
    ├── SHAP feature importance
    └── Sentiment decay analysis
```

**Purpose**: Reproducible analysis and report generation.

---

### `/research_outputs` - Research Results

#### `/research_outputs/figures` - Visualizations

```
figures/
├── figure1_roc_curve.png                     # ROC curve
├── figure2_pr_curve.png                     # Precision-Recall curve
├── figure3_confusion_matrix.png             # Confusion matrix
├── figure4_shap_summary.png                 # SHAP summary plot
├── figure5_shap_force_plot_sample0.png     # SHAP force plot
├── shap_waterfall_*.png                      # SHAP waterfall plots (3 files)
├── shap_dependence_*.png                     # SHAP dependence plots (5 files)
├── shap_interaction_heatmap.png             # SHAP interactions
├── shap_summary_extended.png                 # Extended SHAP summary
├── cumulative_returns.png                   # Cumulative returns
├── cumulative_returns_AAPL.png               # AAPL returns
├── sentiment_decay_curve.png                 # Sentiment decay
├── sentiment_decay_by_ticker.png            # Per-ticker decay
├── event_distribution.png                    # Event distribution
├── event_sentiment.png                       # Event sentiment
├── event_predictive_power.png               # Event predictive power
├── event_ticker_heatmap.png                  # Event-ticker heatmap
├── entity_mentions.png                       # Entity mentions
├── entity_by_ticker.png                     # Entity by ticker
├── entity_sentiment_impact.png              # Entity sentiment impact
├── feature_correlation.png                   # Feature correlation
├── target_correlation.png                    # Target correlation
└── README.md                                 # Figures documentation
```

**Total**: 20+ visualization files

#### `/research_outputs/stats` - Statistical Results

```
stats/
├── shap_feature_importance.csv              # SHAP importance scores
├── statistical_tests.csv                     # Test results (CSV)
├── statistical_tests.json                     # Test results (JSON)
└── README.md                                 # Stats documentation
```

**Purpose**: Statistical analysis results.

#### `/research_outputs/tables` - Processed Data

```
tables/
├── model_ready_full.csv                      # ⚠️ LARGE - Final training data (5-10 MB)
├── stock_with_ta.csv                         # ⚠️ LARGE - Stock + technical indicators
├── events_classified.csv                     # ⚠️ LARGE - Classified news events
├── sentiment_fused.csv                       # ⚠️ LARGE - Fused sentiment scores
├── sentiment_finbert.csv                     # FinBERT outputs
├── sentiment_vader.csv                       # VADER outputs
├── sentiment_textblob.csv                    # TextBlob outputs
├── sentiment_daily_agg.csv                   # Daily aggregated sentiment
├── sentiment_decay.csv                       # Sentiment decay
├── sentiment_decay_by_ticker.csv            # Per-ticker decay
├── event_sentiment_features.csv              # Event-specific sentiment
├── entity_sentiment_features.csv             # Entity-level sentiment
├── entities_extracted.csv                    # Extracted entities
├── news_yahoo.csv                           # ⚠️ LARGE - Yahoo news
├── news_newsapi.csv                          # ⚠️ LARGE - NewsAPI news
├── df_pred.csv                               # Model predictions
├── df_pred_inference.csv                     # Inference predictions
├── backtest_metrics.csv                      # Backtest performance
├── backtest_metrics_AAPL.csv                 # AAPL backtest
├── advanced_model_performance.csv            # Advanced metrics
├── baseline_performance.csv                  # Baseline comparison
├── shap_feature_importance.csv               # SHAP importance (duplicate)
├── statistical_tests.csv                     # Statistical tests (duplicate)
├── statistical_tests.json                     # Statistical tests JSON
└── README.md                                 # Tables documentation
```

**Total**: 25+ CSV files (some are large)

---

### `/scripts` - Data Generation

```
scripts/
├── generate_training_data.py                 # ✅ Main data generation
│   ├── Leak-free target construction
│   ├── News-price alignment
│   ├── Feature generation
│   └── Verification functions
└── generate_training_data_template.py         # ✅ Template for reference
```

**Purpose**: Scripts for generating training data with verified leak-free methods.

---

### `/src` - Source Code

#### `/src/api_clients` - API Integration

```
api_clients/
├── __init__.py
├── news_api.py                               # News API client
│   ├── Yahoo Finance integration
│   ├── NewsAPI integration
│   ├── Fallback mechanisms
│   └── DataFrame conversion
└── settings_ui.py                            # API key management UI
    └── Streamlit UI for API key configuration
```

**Purpose**: API clients for fetching news and market data.

#### `/src/feature_engineering` - Feature Engineering

```
feature_engineering/
├── __init__.py
├── nlp_pipeline.py                           # NLP Feature Generation
│   ├── FinBERT sentiment (transformer)
│   ├── VADER sentiment (lexicon-based)
│   ├── TextBlob sentiment (rule-based)
│   ├── Ensemble sentiment (weighted)
│   ├── Event classification (6 types)
│   ├── Entity extraction (CEO, competitors)
│   ├── Sentiment disagreement metrics
│   └── Output: 24 sentiment features
└── feature_pipeline.py                       # Technical + Lagged Features
    ├── Technical indicators (RSI, MACD, etc.)
    ├── Lagged features (T-1 sentiment, returns)
    ├── VWAP calculation (rolling window)
    ├── Error handling with retry logic
    └── Output: 19 features (15 technical + 4 lagged)
```

**Purpose**: Complete feature engineering pipeline (43 total features).

#### `/src/modeling` - Model Training & Inference

```
modeling/
├── __init__.py
├── models_backtest.py                        # Model Inference & Backtesting
│   ├── ModelLoader (loads CatBoost, scaler)
│   ├── PredictionEngine (inference)
│   ├── BacktestEngine (walk-forward validation)
│   ├── Feature order validation
│   └── Version tracking
└── evaluation.py                             # Advanced Evaluation
    ├── Calibration plots
    ├── Per-ticker breakdown
    ├── Regime analysis
    ├── Precision-Recall curves
    └── Comprehensive evaluation suite
```

**Purpose**: Model inference, backtesting, and evaluation.

#### `/src/utils` - Utilities

```
utils/
├── __init__.py
├── utils.py                                  # General Utilities
│   ├── JSON loading (UTF-8-BOM handling)
│   ├── Text cleaning
│   ├── Date formatting
│   ├── Logging helpers
│   └── Validation functions
└── api_key_manager.py                        # API Key Management
    └── File-based API key storage
```

**Purpose**: Shared utility functions used across the project.

---

### `/tests` - Test Suite

```
tests/
├── __init__.py
├── test_pipeline_e2e.py                     # ✅ Comprehensive E2E Tests
│   ├── test_schema (feature schema validation)
│   ├── test_utils (utility functions)
│   ├── test_ticker_metadata (metadata loading)
│   ├── test_nlp_pipeline (sentiment generation)
│   ├── test_feature_pipeline (feature assembly)
│   ├── test_model_prediction (model inference)
│   ├── test_full_pipeline (end-to-end)
│   ├── test_feature_schema (schema consistency)
│   ├── test_determinism (reproducibility)
│   └── test_no_future_leakage (leakage detection)
└── test_api_keys.py                         # API Key Tests
    └── API key loading and validation
```

**Purpose**: Comprehensive test suite ensuring system correctness.

---

## 📄 Root-Level Documentation Files

```
Root/
├── README.md                                 # ✅ Main project README
├── LICENSE                                   # ✅ MIT License
├── requirements.txt                          # ✅ Python dependencies
├── FEATURE_SCHEMA.py                         # ✅ Central feature schema
├── .gitignore                                # ✅ Git ignore rules
│
├── LIMITATIONS.md                            # ✅ Transparent limitations
├── VALIDATION_ANSWERS.md                     # ✅ Answers to 5 critical questions
├── CRITICAL_VALIDATION_CHECKLIST.md         # ✅ Pre-publication checklist
├── FIXES_SUMMARY.md                         # ✅ Summary of technical fixes
├── TASKS_COMPLETED.md                       # ✅ Completion record
├── TEST_RESULTS.md                          # ✅ Test results summary
├── TEST_FIX_SUMMARY.md                      # ✅ Test fix documentation
├── PRE_PUBLICATION_CHECKLIST.md             # ✅ Final checklist
├── DATA_SPLITS_TEMPLATE.md                  # ✅ Data splits template
└── RESEARCH_NOVELTY.md                      # ✅ Novel contributions for paper
```

---

## 📊 File Statistics

| Category | Count | Total Size (Est.) |
|----------|-------|-------------------|
| **Python Source** | ~15 files | ~200 KB |
| **Tests** | 2 files | ~50 KB |
| **Documentation** | 15+ files | ~500 KB |
| **Configuration** | 3 files | ~10 KB |
| **Models** | 2 files | ~3-5 MB |
| **Notebooks** | 1 file | ~50 KB |
| **Figures** | 20+ files | ~5-10 MB |
| **Tables (CSV)** | 25+ files | ~20-50 MB |
| **Total** | ~80+ files | ~30-70 MB |

---

## 🎯 Files by Purpose

### Core Application
- `app/app_main.py`
- `src/**/*.py`
- `FEATURE_SCHEMA.py`

### Configuration
- `config/tickers.json` ✅
- `config/api_keys.json` ❌ (sensitive)
- `config/api_keys.example.json` ✅

### Documentation
- `README.md`
- `docs/**/*.md`
- `*.md` (all markdown files)

### Research Outputs
- `research_outputs/figures/*.png` ✅
- `research_outputs/stats/*.csv` ✅
- `research_outputs/tables/*.csv` ⚠️ (some large)

### Models
- `models/scaler_ensemble.pkl` ✅
- `models/catboost_best.pkl` ⚠️ (large)

### Tests
- `tests/**/*.py` ✅

---

**Last Updated**: 2025-01-XX
**Total Files**: ~80+ (excluding cache)
**Repository Size**: ~30-70 MB (depending on included files)

