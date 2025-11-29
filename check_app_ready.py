"""
Quick check to verify app_main.py is ready to run.
"""
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Checking app_main.py readiness...\n")

# Check 1: Syntax
print("1. Checking syntax...")
try:
    with open("app/app_main.py", "r", encoding="utf-8") as f:
        compile(f.read(), "app/app_main.py", "exec")
    print("   [OK] Syntax is valid")
except SyntaxError as e:
    print(f"   [ERROR] Syntax error: {e}")
    sys.exit(1)
except UnicodeDecodeError as e:
    print(f"   [ERROR] Encoding issue: {e}")
    sys.exit(1)

# Check 2: Imports
print("\n2. Checking imports...")
try:
    from src.utils.utils import safe_json_load, log_info, log_error
    from src.api_clients.news_api import fetch_news_dataframe_for_ticker
    from src.api_clients.settings_ui import render_api_settings
    from src.feature_engineering.nlp_pipeline import generate_sentiment_features
    from src.modeling.models_backtest import PredictionEngine, quick_predict, compare_strategies
    print("   [OK] All imports successful")
except ImportError as e:
    print(f"   [ERROR] Import error: {e}")
    sys.exit(1)

# Check 3: Config files
print("\n3. Checking config files...")
ticker_file = PROJECT_ROOT / "config" / "tickers.json"
if ticker_file.exists():
    tickers = safe_json_load(ticker_file)
    print(f"   [OK] tickers.json found ({len(tickers)} tickers)")
else:
    print("   [WARNING] tickers.json not found (app will fail)")

# Check 4: Model files
print("\n4. Checking model files...")
model_file = PROJECT_ROOT / "models" / "catboost_best.pkl"
scaler_file = PROJECT_ROOT / "models" / "scaler_ensemble.pkl"

if model_file.exists():
    print(f"   [OK] Model file found ({model_file.stat().st_size / 1024 / 1024:.1f} MB)")
else:
    print("   [WARNING] Model file not found (predictions will fail)")

if scaler_file.exists():
    print(f"   [OK] Scaler file found ({scaler_file.stat().st_size / 1024:.1f} KB)")
else:
    print("   [WARNING] Scaler file not found (predictions will fail)")

# Check 5: Dependencies
print("\n5. Checking key dependencies...")
dependencies = {
    "streamlit": "st",
    "pandas": "pd",
    "numpy": "np",
    "plotly": "go",
    "yfinance": "yf",
    "torch": "torch",
    "transformers": "transformers",
}

missing = []
for module, alias in dependencies.items():
    try:
        __import__(module)
        print(f"   [OK] {module}")
    except ImportError:
        print(f"   [ERROR] {module} (not installed)")
        missing.append(module)

if missing:
    print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
    print("   Install with: pip install " + " ".join(missing))
else:
    print("\n✅ All dependencies available")

# Summary
print("\n" + "="*50)
if not missing and ticker_file.exists():
    print("[SUCCESS] app_main.py is READY TO RUN!")
    print("\nTo run the app:")
    print("   streamlit run app/app_main.py")
else:
    print("[WARNING] app_main.py has some issues:")
    if missing:
        print(f"   - Install missing dependencies: {', '.join(missing)}")
    if not ticker_file.exists():
        print("   - Create config/tickers.json")
    if not model_file.exists() or not scaler_file.exists():
        print("   - Model files missing (predictions will fail)")

