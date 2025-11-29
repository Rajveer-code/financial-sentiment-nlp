# Complete File Architecture

## Project Structure (Complete)

```
financial-sentiment-nlp/
│
├── 📁 app/                                    # Streamlit Application
│   └── app_main.py                           # Main dashboard application
│
├── 📁 config/                                 # Configuration Files
│   ├── api_keys.json                         # ⚠️ SENSITIVE - API keys (DO NOT PUSH)
│   └── tickers.json                          # ✅ Ticker metadata (company names, CEOs, competitors)
│
├── 📁 docs/                                   # Documentation
│   ├── ARCHITECTURE.md                       # System architecture diagram
│   ├── PERFORMANCE_METRICS.md               # Performance evaluation results
│   ├── MEDIUM_ARTICLE_OUTLINE.md            # Article structure
│   └── FILE_ARCHITECTURE.md                 # This file
│
├── 📁 models/                                 # Trained Model Artifacts
│   ├── catboost_best.pkl                    # ⚠️ LARGE - Trained CatBoost model (~2-5 MB)
│   └── scaler_ensemble.pkl                   # ✅ Feature scaler (small, can push)
│
├── 📁 notebooks/                              # Jupyter Notebooks
│   └── generate_report.ipynb                # Report generation notebook
│
├── 📁 research_outputs/                       # Research Results
│   ├── 📁 figures/                           # Generated Plots & Visualizations
│   │   ├── figure1_roc_curve.png            # ROC curve
│   │   ├── figure2_pr_curve.png             # Precision-Recall curve
│   │   ├── figure3_confusion_matrix.png     # Confusion matrix
│   │   ├── figure4_shap_summary.png         # SHAP summary plot
│   │   ├── figure5_shap_force_plot_sample0.png
│   │   ├── shap_waterfall_sample0.png       # SHAP waterfall plots
│   │   ├── shap_waterfall_sample301.png
│   │   ├── shap_waterfall_sample601.png
│   │   ├── shap_dependence_*.png            # SHAP dependence plots (5 files)
│   │   ├── shap_interaction_heatmap.png     # SHAP interaction heatmap
│   │   ├── shap_summary_extended.png        # Extended SHAP summary
│   │   ├── cumulative_returns.png           # Cumulative returns chart
│   │   ├── cumulative_returns_AAPL.png      # AAPL-specific returns
│   │   ├── sentiment_decay_curve.png        # Sentiment decay analysis
│   │   ├── sentiment_decay_by_ticker.png    # Per-ticker decay
│   │   ├── event_distribution.png           # Event type distribution
│   │   ├── event_sentiment.png              # Event sentiment analysis
│   │   ├── event_predictive_power.png       # Event predictive power
│   │   ├── event_ticker_heatmap.png         # Event-ticker heatmap
│   │   ├── entity_mentions.png              # Entity mention analysis
│   │   ├── entity_by_ticker.png            # Entity mentions by ticker
│   │   ├── entity_sentiment_impact.png     # Entity sentiment impact
│   │   ├── feature_correlation.png         # Feature correlation matrix
│   │   ├── target_correlation.png          # Target correlation
│   │   └── README.md                        # Figures documentation
│   │
│   ├── 📁 stats/                             # Statistical Analysis Results
│   │   ├── shap_feature_importance.csv      # SHAP feature importance scores
│   │   ├── statistical_tests.csv           # Statistical test results (CSV)
│   │   ├── statistical_tests.json          # Statistical test results (JSON)
│   │   └── README.md                        # Stats documentation
│   │
│   └── 📁 tables/                            # Processed Data Tables
│       ├── model_ready_full.csv            # ⚠️ LARGE - Final training data (~5-10 MB)
│       ├── stock_with_ta.csv               # ⚠️ LARGE - Stock data with technical indicators
│       ├── events_classified.csv           # ⚠️ LARGE - Classified news events
│       ├── sentiment_fused.csv            # ⚠️ LARGE - Fused sentiment scores
│       ├── sentiment_finbert.csv          # FinBERT sentiment outputs
│       ├── sentiment_vader.csv            # VADER sentiment outputs
│       ├── sentiment_textblob.csv         # TextBlob sentiment outputs
│       ├── sentiment_daily_agg.csv        # Daily aggregated sentiment
│       ├── sentiment_decay.csv            # Sentiment decay analysis
│       ├── sentiment_decay_by_ticker.csv  # Per-ticker sentiment decay
│       ├── event_sentiment_features.csv   # Event-specific sentiment
│       ├── entity_sentiment_features.csv  # Entity-level sentiment
│       ├── entities_extracted.csv         # Extracted entities
│       ├── news_yahoo.csv                 # ⚠️ LARGE - Yahoo news data
│       ├── news_newsapi.csv               # ⚠️ LARGE - NewsAPI data
│       ├── df_pred.csv                    # Model predictions
│       ├── df_pred_inference.csv          # Inference predictions
│       ├── backtest_metrics.csv           # Backtest performance metrics
│       ├── backtest_metrics_AAPL.csv     # AAPL-specific backtest
│       ├── advanced_model_performance.csv # Advanced model metrics
│       ├── baseline_performance.csv       # Baseline comparison
│       ├── shap_feature_importance.csv    # SHAP importance (duplicate of stats/)
│       ├── statistical_tests.csv          # Statistical tests (duplicate of stats/)
│       ├── statistical_tests.json         # Statistical tests JSON
│       └── README.md                       # Tables documentation
│
├── 📁 scripts/                               # Data Generation Scripts
│   ├── generate_training_data.py           # ✅ Main data generation (verified leak-free)
│   └── generate_training_data_template.py  # ✅ Template for reference
│
├── 📁 src/                                   # Source Code
│   ├── __init__.py                          # Package initialization
│   │
│   ├── 📁 api_clients/                      # API Integration Layer
│   │   ├── __init__.py
│   │   ├── news_api.py                     # News API client (Yahoo, NewsAPI)
│   │   └── settings_ui.py                  # API key management UI
│   │
│   ├── 📁 feature_engineering/              # Feature Engineering Pipeline
│   │   ├── __init__.py
│   │   ├── nlp_pipeline.py                 # NLP features (24 sentiment features)
│   │   └── feature_pipeline.py              # Technical + lagged features (19 features)
│   │
│   ├── 📁 modeling/                         # Model Training & Inference
│   │   ├── __init__.py
│   │   ├── models_backtest.py              # Model inference + backtest engine
│   │   └── evaluation.py                   # Advanced evaluation metrics
│   │
│   └── 📁 utils/                            # Utility Functions
│       ├── __init__.py
│       ├── utils.py                        # General utilities (JSON, text, dates)
│       └── api_key_manager.py              # API key management
│
├── 📁 tests/                                 # Test Suite
│   ├── __init__.py
│   ├── test_pipeline_e2e.py               # ✅ Comprehensive end-to-end tests (10 tests)
│   └── test_api_keys.py                    # API key loading tests
│
├── 📁 Documentation/                        # Additional Documentation
│   └── README.md                            # Project documentation
│
├── 📄 Root Level Files
│   ├── README.md                            # ✅ Main project README
│   ├── LICENSE                              # ✅ MIT License
│   ├── requirements.txt                     # ✅ Python dependencies
│   ├── FEATURE_SCHEMA.py                   # ✅ Central feature schema definition
│   ├── .gitignore                          # ✅ Git ignore rules
│   │
│   ├── 📄 Documentation Files
│   │   ├── LIMITATIONS.md                  # ✅ Transparent limitations documentation
│   │   ├── VALIDATION_ANSWERS.md           # ✅ Answers to 5 critical validation questions
│   │   ├── CRITICAL_VALIDATION_CHECKLIST.md # ✅ Pre-publication checklist
│   │   ├── FIXES_SUMMARY.md                # ✅ Summary of all technical fixes
│   │   ├── TASKS_COMPLETED.md              # ✅ Completion record
│   │   ├── TEST_RESULTS.md                 # ✅ Test results summary
│   │   ├── TEST_FIX_SUMMARY.md             # ✅ Test fix documentation
│   │   ├── PRE_PUBLICATION_CHECKLIST.md    # ✅ Final pre-publication checklist
│   │   └── DATA_SPLITS_TEMPLATE.md         # ✅ Template for data splits documentation
│   │
│   └── 📄 Python Cache (Auto-generated)
│       └── __pycache__/                    # ⚠️ DO NOT PUSH - Python bytecode cache
│
└── 📁 .github/ (if exists)                  # GitHub Actions
    └── workflows/                          # CI/CD workflows
```

---

## File Count Summary

| Category | Count | Notes |
|----------|-------|-------|
| **Python Source Files** | ~15 | Core application code |
| **Test Files** | 2 | Comprehensive test suite |
| **Documentation Files** | 12+ | READMEs, guides, checklists |
| **Configuration Files** | 2 | API keys (sensitive), tickers |
| **Model Files** | 2 | Trained models (.pkl) |
| **Data Files (CSV)** | 25+ | Research outputs, processed data |
| **Image Files (PNG)** | 20+ | Plots, visualizations |
| **Notebooks** | 1 | Jupyter notebook |
| **Total Files** | ~80+ | Excluding cache |

---

## File Size Estimates

| File Type | Estimated Size | Should Push? |
|-----------|---------------|--------------|
| `models/catboost_best.pkl` | 2-5 MB | ⚠️ Optional (can use Git LFS) |
| `models/scaler_ensemble.pkl` | < 100 KB | ✅ Yes |
| `research_outputs/tables/model_ready_full.csv` | 5-10 MB | ⚠️ Optional (can use Git LFS) |
| `research_outputs/tables/*.csv` (others) | 1-5 MB each | ⚠️ Optional (research outputs) |
| `research_outputs/figures/*.png` | 100-500 KB each | ✅ Yes (documentation) |
| `config/api_keys.json` | < 1 KB | ❌ NO (sensitive) |
| Python source files | < 50 KB each | ✅ Yes |
| Documentation | < 100 KB each | ✅ Yes |

---

## Directory Purposes

### `/app` - Application Layer
- **Purpose**: User-facing Streamlit application
- **Files**: Main dashboard, UI components
- **Status**: ✅ Push to GitHub

### `/config` - Configuration
- **Purpose**: Configuration files (API keys, ticker metadata)
- **Files**: 
  - `api_keys.json` - ❌ DO NOT PUSH (sensitive)
  - `tickers.json` - ✅ Push (public metadata)
- **Status**: Partial push (exclude sensitive files)

### `/docs` - Documentation
- **Purpose**: Comprehensive documentation
- **Files**: Architecture, metrics, article outlines
- **Status**: ✅ Push to GitHub

### `/models` - Model Artifacts
- **Purpose**: Trained model files
- **Files**: CatBoost model, scaler
- **Status**: ⚠️ Optional (large files, consider Git LFS)

### `/notebooks` - Analysis Notebooks
- **Purpose**: Jupyter notebooks for analysis
- **Files**: Report generation notebook
- **Status**: ✅ Push to GitHub

### `/research_outputs` - Research Results
- **Purpose**: All research outputs (figures, tables, stats)
- **Files**: 
  - Figures: ✅ Push (documentation)
  - Tables: ⚠️ Optional (large CSVs, consider Git LFS)
  - Stats: ✅ Push (small files)
- **Status**: Partial push (exclude very large files)

### `/scripts` - Data Generation
- **Purpose**: Scripts for generating training data
- **Files**: Data generation with verified leak-free methods
- **Status**: ✅ Push to GitHub

### `/src` - Source Code
- **Purpose**: Core application code
- **Files**: All Python modules
- **Status**: ✅ Push to GitHub (exclude `__pycache__`)

### `/tests` - Test Suite
- **Purpose**: Comprehensive testing
- **Files**: End-to-end tests, unit tests
- **Status**: ✅ Push to GitHub (exclude `__pycache__`)

---

## File Naming Conventions

### Python Files
- `snake_case.py` - Standard Python naming
- `__init__.py` - Package initialization

### Documentation
- `UPPERCASE.md` - Important documentation (FEATURE_SCHEMA, README)
- `Title_Case.md` - Detailed guides (LIMITATIONS, VALIDATION_ANSWERS)

### Data Files
- `snake_case.csv` - Processed data tables
- `snake_case.json` - Configuration and results

### Model Files
- `model_name.pkl` - Pickled model artifacts

---

**Last Updated**: 2025-01-XX
**Total Files**: ~80+ (excluding cache)
**Total Size**: ~50-100 MB (with all data files)

