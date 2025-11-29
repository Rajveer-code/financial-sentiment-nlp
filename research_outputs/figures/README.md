# Figures

This folder contains all visual outputs used in the research paper and supporting analysis.  
Each figure is stored as a high-resolution PNG and grouped conceptually into categories that match the study’s flow:  
sentiment analysis, event classification, feature behavior, SHAP explainability, and model performance.

Below is a clear description of every figure in this directory.

---

## 📈 1. Market Performance & Backtesting

### **`cumulative_returns.png`**
Shows the cumulative strategy return compared to baseline over the entire evaluation period.

### **`cumulative_returns_AAPL.png`**
Ticker-specific cumulative return curve for AAPL.

---

## 📰 2. Entity & Event Analysis

### **`entity_by_ticker.png`**
Heatmap showing entity (CEO/Product/Competitor) presence across tickers.

### **`entity_mentions.png`**
Counts of entity mentions extracted from headlines.

### **`entity_sentiment_impact.png`**
Comparison of sentiment when CEO/Product/Competitor is mentioned vs not mentioned.

### **`event_distribution.png`**
Distribution of event types across the dataset  
(Earnings, Product, Analyst, Regulatory, Macroeconomic, M&A).

### **`event_predictive_power.png`**
Movement-up rates for each event type (Negative/Neutral/Positive).

### **`event_sentiment.png`**
Sentiment distribution for each event category.

### **`event_ticker_heatmap.png`**
Heatmap showing event frequency per ticker.

---

## 🧠 3. Feature Relationships & Correlations

### **`feature_correlation.png`**
Full correlation matrix of engineered features.

### **`target_correlation.png`**
Bar chart showing correlation between each feature and the target movement.

---

## ⏳ 4. Sentiment Decay Analysis

### **`sentiment_decay_curve.png`**
Global decay curve showing how news sentiment influence decays over time.

### **`sentiment_decay_by_ticker.png`**
Decay curves computed individually for each ticker.

---

## 🔍 5. SHAP Explainability (Feature Contribution)

### **`figure4_shap_summary.png`**
Main SHAP summary plot for the final CatBoost model  
(feature importance + distribution).

### **`shap_summary_extended.png`**
Extended variant with more features shown.

### **`shap_interaction_heatmap.png`**
SHAP interaction matrix (pairwise feature interactions).

### **`shap_dependence_CMF.png`**
Dependence plot for CMF contributions.

### **`shap_dependence_MACD.png`**
Dependence plot for MACD contributions.

### **`shap_dependence_OBV.png`**
Dependence plot for OBV contributions.

### **`shap_dependence_volatility_lag1.png`**
Dependence plot for lagged volatility.

### **`shap_dependence_daily_return_lag1.png`**
Dependence plot for lagged daily return.

### **`shap_waterfall_sample0.png`**  
### **`shap_waterfall_sample301.png`**  
### **`shap_waterfall_sample601.png`**  
Waterfall explanations for individual predictions.

### **`figure5_shap_force_plot_sample0.png`**
Force plot for a representative sample (localized explanation).

---

## 🧪 6. Model Performance Metrics

### **`figure1_roc_curve.png`**
ROC curve for the final test fold (AUC displayed).

### **`figure2_pr_curve.png`**
Precision-Recall curve for the final test fold.

### **`figure3_confusion_matrix.png`**
Confusion matrix showing prediction performance.

---

## ✔ Notes

- All figures were generated from code in `notebooks/` and the datasets in `research_outputs/tables/`.  
- High resolution is preserved for publication or poster use.  
- If new figures are added, please follow the same naming system and update this README.

