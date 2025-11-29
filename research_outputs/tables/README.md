# Tables

This folder stores all tabular datasets that support the analysis, model training, feature engineering, and evaluation presented in the project and research paper.  
They are grouped below into meaningful categories so future readers and reviewers can easily understand the purpose of each file.

---

# 1. 📘 **Core Processed Datasets**

These are the main datasets used in feature engineering, event classification, and model training.

### **`events_classified.csv`**
Fully processed news dataset with:
- event type classifications  
- sentiment scores  
- confidence scores  
- enriched metadata  

Used directly in model-ready merges.

### **`entity_sentiment_features.csv`**
Entity-level sentiment signals used in the paper:
- CEO sentiment  
- product sentiment  
- competitor sentiment  
- entity density  
- sentiment gaps  

### **`event_sentiment_features.csv`**
Aggregate sentiment statistics grouped by event category:
- earnings sentiment  
- product sentiment  
- analyst event sentiment  
- regulatory sentiment  
- etc.  

### **`entities_extracted.csv`**
Raw entity recognition output:
- entity names  
- mention counts  
- contextual polarity  

### **`sentiment_fused.csv`**
Complete merged sentiment dataset combining:
- FinBERT  
- VADER  
- TextBlob  
- ensemble sentiment  
- consensus confidence  

### **Sentiment model outputs:**
- **`sentiment_finbert.csv`**
- **`sentiment_vader.csv`**
- **`sentiment_textblob.csv`**

These contain raw sentiment outputs before fusion.

### **News-source datasets**
- **`news_newspai.csv`** (NewsAPI feed)  
- **`news_yahoo.csv`** (Yahoo Finance + RSS feed)

---

# 2. 📈 **Feature Engineering & Market Data**

### **`stock_with_ta.csv`**
Price data merged with technical indicators:
- MACD  
- RSI  
- ATR  
- OBV  
- volatility lags  
- daily return lags  
- CMF, etc.

### **`sentiment_daily_agg.csv`**
Daily-level aggregated sentiment signals:
- daily mean  
- daily variance  
- sentiment agreement  
- sentiment range, etc.

### **`sentiment_decay.csv` / `sentiment_decay_by_ticker.csv`**
Output of sentiment-decay analysis used in:
- decay curve plots  
- half-life calculations  
- ticker-level sensitivity studies  

### **`shap_feature_importance.csv`**
Tabular version of SHAP importances for:
- ranking features  
- reproducible model explainability  

### **`model_ready_full.csv`**
Final feature matrix used for model training and rolling cross-validation:
- all price + sentiment + entity + event features  
- target variable (movement)  

---

# 3. 🧪 **Model Predictions & Inference Outputs**

### **`df_pred.csv`**
Full prediction dataset for all tickers and test periods:
- predicted movement  
- model probabilities  
- confidence-weighted sentiment  

### **`df_pred_inference.csv`**
Cleaned version focusing only on:
- actual vs predicted  
- final fold evaluation  

Used for:
- PR curve  
- ROC curve  
- confusion matrix  

---

# 4. 📊 **Backtest & Performance Metrics**

### **`advanced_model_performance.csv`**
Performance metrics for advanced/ensemble models:
- precision  
- recall  
- F1  
- ROC-AUC  
- PR-AUC  
- calibration metrics  

### **`baseline_performance.csv`**
Baseline model comparison table:
- moving average  
- momentum  
- naive classifier  

### **`backtest_metrics.csv`**
Rolling backtest performance across all tickers:
- cumulative return  
- volatility  
- hit rate  
- drawdowns  

### **`backtest_metrics_AAPL.csv`**
Ticker-specific breakdown for AAPL:
- used for case-study visuals  
- cumulative returns chart  

---

# 5. 🧬 **Statistical Outputs**

(*These also appear in the `stats/` folder. Kept here for convenience.*)

- **`statistical_tests.csv`**  
- **`statistical_tests.json`**

Contain all hypothesis tests:
- event predictive power  
- sentiment correlations  
- movement distribution tests  
- p-values & effect sizes  

---

# ✔ Notes
- All tables here are **directly referenced in the research paper**.  
- None of these files contain API keys or sensitive credentials.  
- If additional datasets are generated, please update this README to keep the repository consistent.

