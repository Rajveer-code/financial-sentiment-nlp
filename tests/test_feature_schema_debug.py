import sys
from pathlib import Path

# --- ADD PROJECT ROOT ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# --- IMPORT AFTER PATCH ---
from FEATURE_SCHEMA import MODEL_FEATURES
from src.feature_engineering.feature_pipeline import create_model_input_dataframe

# Test data
features = {feat: 0.5 for feat in MODEL_FEATURES}
df = create_model_input_dataframe(features, "AAPL", "2024-01-01")

print("Columns:", list(df.columns))
print("First 5:", df.columns[:5])
