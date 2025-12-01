#!/usr/bin/env python
"""
FINAL VALIDATION TEST
=====================
Tests that:
1. Price chart data has mixed red/green candles (not all green)
2. NewsAPI is being used to fetch real articles
3. All API providers initialize correctly
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 70)
print("FINAL VALIDATION TEST")
print("=" * 70)

# =========================================================================
# TEST 1: Provider Initialization
# =========================================================================
print("\n[TEST 1] Provider Initialization")
print("-" * 70)

from src.api_clients.news_api import get_default_providers

providers = get_default_providers()
print(f"[OK] {len(providers)} providers initialized")
provider_names = [p.name for p in providers]
print(f"  Providers: {', '.join(provider_names)}")

# Verify NewsAPI is first
if providers and providers[0].name == "newsapi":
    print("[OK] NewsAPI is first (priority)")
else:
    print(f"[WARN] NewsAPI is not first. First is: {providers[0].name if providers else 'None'}")

# =========================================================================
# TEST 2: NewsAPI News Fetching
# =========================================================================
print("\n[TEST 2] NewsAPI News Fetching")
print("-" * 70)

from src.api_clients.news_api import fetch_news_with_fallback

ticker = "AAPL"
from_date = datetime.now() - timedelta(days=3)
to_date = datetime.now()

articles = fetch_news_with_fallback(
    ticker=ticker,
    from_date=from_date,
    to_date=to_date,
    max_articles=10
)

print(f"[OK] Fetched {len(articles)} articles for {ticker}")

if articles:
    provider = articles[0].get("provider", "unknown")
    print(f"  Provider used: {provider}")
    
    if provider == "newsapi":
        print("[OK] SUCCESS: NewsAPI is providing the articles!")
    else:
        print(f"[WARN] Articles from {provider}, not NewsAPI")
else:
    print("[WARN] No articles fetched!")

# =========================================================================
# TEST 3: Fallback Price Data (Check for Mixed Red/Green Candles)
# =========================================================================
print("\n[TEST 3] Fallback Price Data (Mixed Red/Green Candles)")
print("-" * 70)

from app.app_main import get_fallback_realistic_data

df = get_fallback_realistic_data(ticker=ticker, days=30)
print(f"[OK] Generated {len(df)} days of fallback price data")

# Check Open vs Close for red/green distribution
df["color"] = df.apply(lambda row: "RED" if row["Open"] > row["Close"] else "GREEN", axis=1)
red_count = (df["color"] == "RED").sum()
green_count = (df["color"] == "GREEN").sum()

print(f"  Distribution: {red_count} RED candles, {green_count} GREEN candles")

if red_count > 0 and green_count > 0:
    print("[OK] Mixed red/green candles!")
elif red_count == 0:
    print("[ERROR] All candles are GREEN (Open <= Close always)")
elif green_count == 0:
    print("[ERROR] All candles are RED (Open > Close always)")

# Show sample
print(f"\n  Sample data (first 5 rows):")
for idx, row in df.head().iterrows():
    color = row["color"]
    o = row["Open"]
    c = row["Close"]
    print(f"    {idx}: O={o:.2f}, C={c:.2f} -> {color}")

# =========================================================================
# TEST 4: API Key Configuration
# =========================================================================
print("\n[TEST 4] API Key Configuration")
print("-" * 70)

from src.utils.api_key_manager import load_api_keys

keys = load_api_keys()
configured = {k: v is not None for k, v in keys.items()}

for key_name, is_configured in configured.items():
    status = "[OK] Configured" if is_configured else "[ERROR] Missing"
    print(f"  {key_name}: {status}")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

all_passed = (
    len(providers) > 0 and 
    providers[0].name == "newsapi" and
    len(articles) > 0 and 
    articles[0].get("provider") == "newsapi" and
    red_count > 0 and 
    green_count > 0
)

if all_passed:
    print("[OK] ALL TESTS PASSED! App is ready to use.")
    print("\nThe dashboard should now:")
    print("  1. Show mixed red/green candlesticks (not all green)")
    print("  2. Fetch real articles from NewsAPI")
    print("  3. Display NewsAPI credits being consumed")
else:
    print("[WARN] Some tests did not pass. See above for details.")

print("=" * 70)
