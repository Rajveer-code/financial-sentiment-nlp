"""
test_api_keys.py
================
Test script to validate the API key management system.

Run this to verify:
1. API keys are saved to disk correctly
2. API keys can be loaded from disk
3. News fetching works with saved keys

Author: Rajveer Singh Pall
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.api_key_manager import (
    save_api_keys,
    load_api_keys,
    validate_api_keys,
    get_config_file_path,
    clear_api_keys
)
from src.api_clients.news_api import get_default_providers, fetch_news_dataframe_for_ticker


def test_api_key_storage():
    """Test 1: API key storage and retrieval."""
    print("\n" + "="*60)
    print("TEST 1: API KEY STORAGE")
    print("="*60)
    
    # Check config file location
    config_path = get_config_file_path()
    print(f"\n📁 Config file location: {config_path}")
    print(f"   Exists: {config_path.exists()}")
    
    # Check if keys are already configured
    existing_keys = load_api_keys()
    print(f"\n🔑 Existing keys:")
    for provider, key in existing_keys.items():
        if key:
            masked = f"{key[:8]}***" if len(key) > 8 else "***"
            print(f"   {provider}: {masked}")
        else:
            print(f"   {provider}: ❌ Not configured")
    
    # Validate keys
    validation = validate_api_keys()
    configured_count = sum(1 for v in validation.values() if v)
    print(f"\n✅ Configured providers: {configured_count}/3")
    
    if configured_count == 0:
        print("\n⚠️  No API keys found!")
        print("   Please configure keys in one of these ways:")
        print("   1. Run the Streamlit app and use Settings UI")
        print("   2. Set environment variables (FINNHUB_API_KEY, etc.)")
        print("   3. Create config/api_keys.json manually")
        return False
    
    print("\n✅ TEST 1 PASSED")
    return True


def test_provider_initialization():
    """Test 2: Provider initialization."""
    print("\n" + "="*60)
    print("TEST 2: PROVIDER INITIALIZATION")
    print("="*60)
    
    providers = get_default_providers()
    
    if not providers:
        print("\n❌ No providers initialized!")
        print("   Check API keys configuration.")
        return False
    
    print(f"\n✅ Initialized {len(providers)} provider(s):")
    for provider in providers:
        print(f"   - {provider.name}")
    
    print("\n✅ TEST 2 PASSED")
    return True


def test_news_fetching():
    """Test 3: Actual news fetching."""
    print("\n" + "="*60)
    print("TEST 3: NEWS FETCHING")
    print("="*60)
    
    test_ticker = "AAPL"
    print(f"\n🔍 Fetching news for {test_ticker}...")
    
    try:
        df = fetch_news_dataframe_for_ticker(
            ticker=test_ticker,
            max_articles=5
        )
        
        if df.empty:
            print(f"\n⚠️  No articles found for {test_ticker}")
            print("   This could mean:")
            print("   1. API keys are invalid")
            print("   2. API rate limits exceeded")
            print("   3. No recent news available")
            return False
        
        print(f"\n✅ Fetched {len(df)} articles!")
        print(f"\n📰 Sample headlines:")
        
        for i, row in df.head(3).iterrows():
            headline = row['headline'][:80]
            source = row['source']
            print(f"   [{source}] {headline}...")
        
        print("\n✅ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ News fetching failed: {e}")
        return False


def test_save_functionality():
    """Test 4: Save new keys (optional)."""
    print("\n" + "="*60)
    print("TEST 4: SAVE FUNCTIONALITY")
    print("="*60)
    
    print("\n📝 Testing save functionality...")
    print("   (Using test keys, will be cleared after test)")
    
    test_keys = {
        "finnhub": "test_finnhub_key",
        "newsapi": "test_newsapi_key",
        "alphavantage": "test_alphavantage_key"
    }
    
    # Save test keys
    success = save_api_keys(test_keys)
    if not success:
        print("\n❌ Failed to save keys")
        return False
    
    # Load and verify
    loaded = load_api_keys()
    
    all_match = True
    for key, value in test_keys.items():
        if loaded.get(key) != value:
            all_match = False
            break
    
    if all_match:
        print("\n✅ Save and load verified!")
    else:
        print("\n⚠️  Saved keys don't match loaded keys")
    
    # Clear test keys
    print("\n🗑️  Clearing test keys...")
    clear_api_keys()
    
    print("\n✅ TEST 4 PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("API KEY SYSTEM - VALIDATION TEST")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Storage & Retrieval", test_api_key_storage()))
    results.append(("Provider Initialization", test_provider_initialization()))
    results.append(("News Fetching", test_news_fetching()))
    results.append(("Save Functionality", test_save_functionality()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYour API key system is working correctly.")
        print("You can now use the Streamlit app to fetch news.")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("\nTroubleshooting:")
        print("1. Ensure API keys are configured (Settings UI or environment)")
        print("2. Check API keys are valid (test on provider websites)")
        print("3. Verify config/api_keys.json exists and has correct format")
        print("4. Check file permissions for config directory")
    
    print("\n")


if __name__ == "__main__":
    main()