# Statistical Outputs

This folder contains all of the statistical summaries and model-level diagnostic data used in the research paper.  
These files provide the quantitative backbone for the claims made in the analysis, including feature strength, hypothesis testing, and model reliability.

Each file is described below.

---

## 📊 1. `shap_feature_importance.csv`
A ranked table of SHAP feature importances extracted from the final CatBoost model.

**What it contains:**
- Feature names  
- Mean |SHAP| value  
- Relative contribution (%)  
- Ranking from strongest → weakest

**Used for:**
- Feature importance discussion  
- SHAP summary plot  
- Model interpretability section

---

## 🧪 2. `statistical_tests.csv`
This is a tabular version of all the hypothesis tests run during the analysis.

**Typical contents include:**
- Event-type predictive power tests  
- Sentiment vs. movement correlation tests  
- t-tests, chi-square tests, KS tests  
- p-values  
- Test statistics  
- Interpretation flag (significant / not significant)

**Used in:**
- Methodology section  
- Statistical validity discussion  
- Appendix for reproducibility  

---

## 🧬 3. `statistical_tests.json`
A structured JSON representation of the same statistical results.  
Useful for programmatic access and reproducibility.

**Contents mirror the CSV version**, but with nested keys such as:
```json
{
  "event_predictive_power": { ... },
  "sentiment_vs_movement": { ... },
  "distribution_tests": { ... }
}
```

**Used for:**
- Automated reporting scripts  
- Regenerating tables  
- Supplementary material for reviewers

---

## ✔ Notes
- These files are generated automatically from analysis notebooks.  
- All values here are directly referenced in the Results and Discussion sections of the research paper.  
- If new tests are introduced, add the outputs to this folder and update this README for consistency.

