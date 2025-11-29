# Files to Exclude from GitHub

## 🚫 Critical: DO NOT PUSH

### 1. Sensitive Data Files

```
config/api_keys.json                    # Contains API keys (sensitive)
```

**Reason**: Contains actual API keys that should never be committed to version control.

**Action**: 
- Create `config/api_keys.example.json` with placeholder values
- Add `config/api_keys.json` to `.gitignore`

---

### 2. Python Cache Files

```
__pycache__/                            # All Python bytecode cache directories
**/__pycache__/                         # Recursive cache directories
*.pyc                                   # Compiled Python files
*.pyo                                   # Optimized Python files
```

**Reason**: Auto-generated files, not needed in repository.

**Action**: Already in `.gitignore` (verify it's comprehensive)

---

### 3. Large Model Files (Optional - Use Git LFS if needed)

```
models/catboost_best.pkl                # Large trained model (2-5 MB)
```

**Reason**: Large binary file. Can use Git LFS if you want to version control it.

**Options**:
- **Option A**: Exclude from GitHub, provide download link
- **Option B**: Use Git LFS (Git Large File Storage)
- **Option C**: Include if < 100 MB (GitHub limit)

**Recommendation**: Use Git LFS or exclude, provide download instructions in README.

---

### 4. Large Data Files (Optional - Consider Git LFS)

```
research_outputs/tables/model_ready_full.csv      # Large training data (5-10 MB)
research_outputs/tables/stock_with_ta.csv         # Large stock data (2-5 MB)
research_outputs/tables/events_classified.csv      # Large event data (2-5 MB)
research_outputs/tables/sentiment_fused.csv        # Large sentiment data (1-3 MB)
research_outputs/tables/news_yahoo.csv             # Large news data (1-3 MB)
research_outputs/tables/news_newsapi.csv           # Large news data (1-3 MB)
```

**Reason**: Large CSV files can bloat repository. Research outputs can be regenerated.

**Options**:
- **Option A**: Exclude, provide data generation script
- **Option B**: Use Git LFS for large files
- **Option C**: Include sample data only (first 100 rows)

**Recommendation**: 
- Keep small/medium CSVs (< 1 MB) ✅
- Use Git LFS for large CSVs (> 5 MB) ⚠️
- Or exclude and document how to generate

---

## ⚠️ Consider Excluding (Large Files)

### Research Output Tables (Large CSVs)

If total size > 50 MB, consider excluding:

```
research_outputs/tables/model_ready_full.csv
research_outputs/tables/stock_with_ta.csv
research_outputs/tables/events_classified.csv
research_outputs/tables/sentiment_fused.csv
research_outputs/tables/news_yahoo.csv
research_outputs/tables/news_newsapi.csv
```

**Alternative**: Include sample versions (first 100-1000 rows) for demonstration.

---

## ✅ Must Include in GitHub

### Source Code
```
src/**/*.py                              # All Python source files
app/**/*.py                              # Application code
scripts/**/*.py                          # Data generation scripts
tests/**/*.py                            # Test files
```

### Configuration (Public)
```
config/tickers.json                      # Ticker metadata (public)
FEATURE_SCHEMA.py                        # Feature schema
requirements.txt                         # Dependencies
```

### Documentation
```
README.md                                # Main README
LICENSE                                  # License file
docs/**/*.md                             # All documentation
*.md                                     # All markdown files
```

### Research Outputs (Small/Medium)
```
research_outputs/figures/*.png           # All visualization images
research_outputs/stats/*.csv             # Statistical results
research_outputs/stats/*.json            # Statistical results
research_outputs/tables/*.csv            # Small/medium tables (< 1 MB each)
research_outputs/**/README.md            # Documentation
```

### Models (Small)
```
models/scaler_ensemble.pkl               # Small scaler file
```

### Notebooks
```
notebooks/*.ipynb                        # Jupyter notebooks
```

---

## 📋 Recommended .gitignore

Create/update `.gitignore` with:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Sensitive files
config/api_keys.json

# Large model files (optional - uncomment if excluding)
# models/catboost_best.pkl

# Large data files (optional - uncomment if excluding)
# research_outputs/tables/model_ready_full.csv
# research_outputs/tables/stock_with_ta.csv
# research_outputs/tables/events_classified.csv
# research_outputs/tables/sentiment_fused.csv
# research_outputs/tables/news_*.csv

# Environment
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage
htmlcov/

# Distribution
dist/
build/
*.egg-info/
```

---

## 🎯 Recommended Strategy

### Minimal Repository (Recommended for GitHub)

**Include**:
- ✅ All source code
- ✅ All documentation
- ✅ Configuration templates (not actual keys)
- ✅ Small model files (< 1 MB)
- ✅ Research figures (images)
- ✅ Small/medium data files (< 1 MB each)
- ✅ Test suite

**Exclude**:
- ❌ API keys (sensitive)
- ❌ Python cache
- ⚠️ Large model files (use Git LFS or exclude)
- ⚠️ Very large data files (> 5 MB, use Git LFS or exclude)

**Total Repository Size**: ~10-20 MB (without large files)

---

## 📦 Git LFS Setup (If Including Large Files)

If you want to include large files, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.pkl"
git lfs track "research_outputs/tables/*.csv"

# Add .gitattributes
git add .gitattributes
```

---

## 🔍 Pre-Push Checklist

Before pushing to GitHub:

- [ ] Remove `config/api_keys.json` (or add to .gitignore)
- [ ] Create `config/api_keys.example.json` with placeholders
- [ ] Verify `.gitignore` includes `__pycache__/`
- [ ] Decide on large files (exclude or use Git LFS)
- [ ] Test repository size: `du -sh .` (should be < 100 MB ideally)
- [ ] Verify no sensitive data in any files
- [ ] Check file sizes: `find . -size +5M` (files > 5 MB)

---

**Last Updated**: 2025-01-XX
**Status**: Ready for GitHub push

