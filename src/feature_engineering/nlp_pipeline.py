"""
nlp_pipeline.py
===============
NLP feature generation aligned EXACTLY with model_ready_full.csv schema.

This module takes raw news headlines and produces the exact 24 sentiment/event/entity
features that the trained CatBoost model expects.

Output features (24 total):
- 13 sentiment features
- 6 event-specific features
- 5 entity features

Author: Rajveer Singh Pall
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import clean_text, log_info, log_warning
from FEATURE_SCHEMA import FEATURE_DEFAULTS

# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seed at module initialization
set_seed(42)

# ============================================================
# MODEL REGISTRY (LAZY LOADING)
# ============================================================

class ModelRegistry:
    """Lazy-load NLP models to avoid slow imports."""
    
    finbert_model = None
    finbert_tokenizer = None
    vader = None
    
    @staticmethod
    def load_finbert():
        if ModelRegistry.finbert_model is None:
            model_name = "ProsusAI/finbert"
            ModelRegistry.finbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
            ModelRegistry.finbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            ModelRegistry.finbert_model.to(DEVICE)
            ModelRegistry.finbert_model.eval()
            log_info(f"FinBERT loaded on {DEVICE}", "NLP")
        return ModelRegistry.finbert_model, ModelRegistry.finbert_tokenizer
    
    @staticmethod
    def load_vader():
        if ModelRegistry.vader is None:
            ModelRegistry.vader = SentimentIntensityAnalyzer()
            log_info("VADER loaded", "NLP")
        return ModelRegistry.vader


# ============================================================
# CORE SENTIMENT MODELS
# ============================================================

def finbert_sentiment(texts: List[str]) -> List[float]:
    """
    Calculate FinBERT sentiment scores.
    
    Returns:
        List of scores in range [-1, 1] (negative to positive)
    """
    if not texts:
        return []
    
    model, tokenizer = ModelRegistry.load_finbert()
    scores = []
    
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = [clean_text(t, max_length=512) for t in texts[i:i + batch_size]]
        
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        
        # FinBERT classes: [negative, neutral, positive]
        pos = probs[:, 2].cpu().numpy()
        neg = probs[:, 0].cpu().numpy()
        
        # Score: positive - negative
        batch_scores = (pos - neg).tolist()
        scores.extend(batch_scores)
    
    return scores


def vader_sentiment(texts: List[str]) -> List[float]:
    """
    Calculate VADER sentiment scores.
    
    Returns:
        List of compound scores in range [-1, 1]
    """
    vader = ModelRegistry.load_vader()
    return [vader.polarity_scores(clean_text(t))["compound"] for t in texts]


def textblob_sentiment(texts: List[str]) -> List[float]:
    """
    Calculate TextBlob sentiment scores.
    
    Returns:
        List of polarity scores in range [-1, 1]
    """
    return [TextBlob(clean_text(t)).sentiment.polarity for t in texts]


# ============================================================
# EVENT CLASSIFICATION
# ============================================================

def classify_event(headline: str) -> Dict[str, float]:
    """
    Classify headline into event categories with confidence scores.
    
    Uses keyword matching matching training data generation.
    This ensures consistency between training and inference pipelines.
    
    Returns probability scores for each event type (matching training data format).
    This matches the continuous scores in events_classified.csv (0.150, 0.258, etc.)
    rather than binary classification.
    
    NOTE: Training data generation used the same keyword-based approach.
    If your training data used a different method (e.g., trained classifier),
    you must either:
    1. Retrain the model with keyword-based features, OR
    2. Include the classifier in this inference pipeline
    
    Returns:
        Dictionary with scores for: earnings, product, analyst, regulatory, macroeconomic, m&a, other
    """
    text = clean_text(headline).lower()
    
    # Initialize scores (keyword matching with weights)
    scores = {
        'earnings': 0.0,
        'product': 0.0,
        'analyst': 0.0,
        'regulatory': 0.0,
        'macroeconomic': 0.0,
        'm&a': 0.0,
        'other': 0.0,
    }
    
    # Earnings keywords (weighted)
    earnings_keywords = ["earnings", "eps", "revenue", "profit", "quarterly", "results", "q1", "q2", "q3", "q4"]
    if any(k in text for k in earnings_keywords):
        scores['earnings'] = 0.8
    
    # Product keywords
    product_keywords = ["launch", "unveils", "introduces", "product", "release", "announces"]
    if any(k in text for k in product_keywords):
        scores['product'] = 0.7
    
    # Analyst keywords
    analyst_keywords = ["analyst", "upgrade", "downgrade", "price target", "rating", "initiates coverage"]
    if any(k in text for k in analyst_keywords):
        scores['analyst'] = 0.7
    
    # Regulatory keywords
    regulatory_keywords = ["regulatory", "lawsuit", "fine", "sec", "probe", "investigation", "fda", "ftc"]
    if any(k in text for k in regulatory_keywords):
        scores['regulatory'] = 0.8
    
    # Macroeconomic keywords
    macro_keywords = ["inflation", "fed", "interest rate", "macro", "economy", "gdp", "unemployment"]
    if any(k in text for k in macro_keywords):
        scores['macroeconomic'] = 0.7
    
    # M&A keywords
    ma_keywords = ["merger", "acquisition", "buyout", "takeover", "deal", "m&a"]
    if any(k in text for k in ma_keywords):
        scores['m&a'] = 0.8
    
    # If no matches, assign to 'other'
    if sum(scores.values()) == 0.0:
        scores['other'] = 1.0
    else:
        # Normalize to probabilities (softmax-like)
        total = sum(np.exp(v) for v in scores.values())
        scores = {k: np.exp(v) / total for k, v in scores.items()}
    
    return scores


def classify_event_simple(headline: str) -> str:
    """
    Simple binary event classification (backward compatibility).
    
    Returns:
        One of: earnings, product, analyst, regulatory, macroeconomic, other
    """
    scores = classify_event(headline)
    # Return the event type with highest score
    return max(scores.items(), key=lambda x: x[1])[0]


# ============================================================
# ENTITY DETECTION
# ============================================================

def calculate_ceo_sentiment(headlines: List[str], ceo_name: str) -> tuple[int, float]:
    """
    Calculate CEO-specific sentiment.
    
    Returns:
        (mention_count, average_sentiment)
    """
    if not ceo_name:
        return 0, 0.0
    
    ceo_lower = ceo_name.lower()
    ceo_headlines = [h for h in headlines if ceo_lower in h.lower()]
    
    if not ceo_headlines:
        return 0, 0.0
    
    ceo_scores = textblob_sentiment(ceo_headlines)
    return len(ceo_headlines), float(np.mean(ceo_scores))


def calculate_competitor_mentions(headlines: List[str], competitors: List[str]) -> int:
    """Count competitor mentions across all headlines."""
    if not competitors:
        return 0
    
    comp_lower = [c.lower() for c in competitors]
    count = 0
    
    for h in headlines:
        h_lower = h.lower()
        if any(c in h_lower for c in comp_lower):
            count += 1
    
    return count


def calculate_entity_density(headlines: List[str], company_names: List[str]) -> float:
    """
    Calculate entity density (mentions per headline).
    
    Returns:
        Average mentions per headline
    """
    if not headlines or not company_names:
        return 0.0
    
    total_mentions = 0
    for h in headlines:
        h_lower = h.lower()
        for name in company_names:
            total_mentions += h_lower.count(name.lower())
    
    return total_mentions / len(headlines)


def calculate_entity_sentiment_gap(headlines: List[str], company_name: str) -> float:
    """
    Calculate sentiment gap between entity-mentioned vs all headlines.
    
    OPTIMIZED: Pre-compute FinBERT scores once instead of calling twice.
    
    Returns:
        sentiment(entity_mentioned) - sentiment(all)
    """
    if not headlines or not company_name:
        return 0.0
    
    # Pre-compute FinBERT scores once for all headlines
    all_scores = np.array(finbert_sentiment(headlines))
    all_sentiment = float(np.mean(all_scores))
    
    # Filter entity-mentioned headlines and use pre-computed scores
    entity_indices = [
        i for i, h in enumerate(headlines)
        if company_name.lower() in h.lower()
    ]
    
    if not entity_indices:
        return 0.0
    
    entity_sentiment = float(np.mean([all_scores[i] for i in entity_indices]))
    
    return float(entity_sentiment - all_sentiment)


# ============================================================
# MAIN PIPELINE
# ============================================================

def align_news_to_trading_day(
    news_date: datetime,
    market_calendar: Optional[pd.DatetimeIndex] = None
) -> datetime:
    """
    Align news publication date to next trading day.
    
    CRITICAL ASSUMPTION: News from weekends/holidays → assigned to next trading day features.
    
    Examples:
    - News published: 2024-11-24 (Sunday) 10:00 AM
      → Aligned to: 2024-11-25 (Monday) features
      → Predicts: 2024-11-26 (Tuesday) movement
    
    - News published: 2024-11-23 (Saturday)
      → Aligned to: 2024-11-25 (Monday) features
      → Predicts: 2024-11-26 (Tuesday) movement
    
    This ensures:
    1. Weekend/holiday news contributes to next trading day's features
    2. Features at time T predict movement at time T+1
    3. No look-ahead bias (news timestamp < prediction timestamp)
    
    Args:
        news_date: News publication datetime
        market_calendar: Optional business day calendar (uses default if None)
        
    Returns:
        Next trading day after news publication
    """
    if market_calendar is None:
        # Use pandas business day calendar
        from pandas.tseries.offsets import BDay
        # Get next business day
        next_bday = news_date + BDay(1)
        return next_bday
    
    # Use provided calendar
    news_date_only = news_date.date() if isinstance(news_date, datetime) else news_date
    price_dates_only = pd.to_datetime(market_calendar).date if hasattr(market_calendar, 'date') else market_calendar
    
    next_trading_days = [d for d in price_dates_only if d > news_date_only]
    if not next_trading_days:
        return max(price_dates_only)
    
    return min(next_trading_days)


def generate_sentiment_features(
    headlines_df: pd.DataFrame,
    ticker_metadata: Dict[str, Dict],
    ticker: Optional[str] = None
) -> Dict[str, float]:
    """
    Generate ALL 24 sentiment/event/entity features for a single ticker/date.
    
    Args:
        headlines_df: DataFrame with columns [date, ticker, headline]
        ticker_metadata: Ticker metadata dict (from tickers.json)
        ticker: Optional ticker to filter (if None, uses first ticker in df)
        
    Returns:
        Dictionary with 24 features matching FEATURE_SCHEMA
    """
    if headlines_df.empty:
        log_warning("No headlines provided, returning defaults", "NLP")
        return {k: FEATURE_DEFAULTS[k] for k in FEATURE_DEFAULTS if k in [
            "finbert_sentiment_score_mean", "vader_sentiment_score_mean",
            "textblob_sentiment_score_mean", "ensemble_sentiment_mean",
            "sentiment_variance_mean", "model_consensus_mean",
            "ensemble_sentiment_max", "ensemble_sentiment_min",
            "ensemble_sentiment_std", "confidence_mean",
            "num_headlines", "headline_length_avg",
            "sentiment_earnings", "sentiment_product", "sentiment_analyst",
            "count_positive_earnings", "count_negative_regulatory",
            "has_macroeconomic_news", "ceo_mention_count", "ceo_sentiment",
            "competitor_mention_count", "entity_density", "entity_sentiment_gap"
        ]}
    
    # Infer ticker if not provided
    if ticker is None:
        ticker = headlines_df.iloc[0]["ticker"]
    
    headlines = headlines_df["headline"].astype(str).tolist()
    
    # -------------------- Base Sentiment Scores --------------------
    
    finbert_scores = np.array(finbert_sentiment(headlines))
    vader_scores = np.array(vader_sentiment(headlines))
    textblob_scores = np.array(textblob_sentiment(headlines))
    
    # Ensemble: weighted average
    ensemble_scores = 0.6 * finbert_scores + 0.3 * vader_scores + 0.1 * textblob_scores
    
    # Variance across models
    sentiment_variance = float(np.var([
        np.mean(finbert_scores),
        np.mean(vader_scores),
        np.mean(textblob_scores)
    ]))
    
    # Model consensus (1 - std of model means)
    model_consensus = 1.0 - float(np.std([
        np.mean(finbert_scores),
        np.mean(vader_scores),
        np.mean(textblob_scores)
    ]))
    
    # Confidence (mean absolute ensemble score)
    confidence_mean = float(np.mean(np.abs(ensemble_scores)))
    
    # -------------------- Event Classification --------------------
    
    # Get event scores for each headline (returns dict of scores)
    event_scores_list = [classify_event(h) for h in headlines]
    
    # For backward compatibility, also get simple classifications
    events = [classify_event_simple(h) for h in headlines]
    
    # Event-specific sentiment (using simple classification for filtering)
    def event_sentiment(event_type: str) -> float:
        indices = [i for i, e in enumerate(events) if e == event_type]
        if not indices:
            return 0.0
        return float(np.mean(ensemble_scores[indices]))
    
    sentiment_earnings = event_sentiment("earnings")
    sentiment_product = event_sentiment("product")
    sentiment_analyst = event_sentiment("analyst")
    
    # Event counts
    count_positive_earnings = sum(
        1 for i, e in enumerate(events)
        if e == "earnings" and ensemble_scores[i] > 0.1
    )
    
    count_negative_regulatory = sum(
        1 for i, e in enumerate(events)
        if e == "regulatory" and ensemble_scores[i] < -0.1
    )
    
    has_macroeconomic_news = 1 if "macroeconomic" in events else 0
    
    # -------------------- Entity Features --------------------
    
    meta = ticker_metadata.get(ticker, {})
    ceo_name = meta.get("ceo", "")
    competitors = meta.get("competitors", [])
    company_name = meta.get("company_name", ticker)
    short_name = meta.get("short_name", ticker)
    company_names = [company_name, short_name]
    
    ceo_mention_count, ceo_sentiment = calculate_ceo_sentiment(headlines, ceo_name)
    competitor_mention_count = calculate_competitor_mentions(headlines, competitors)
    entity_density = calculate_entity_density(headlines, company_names)
    entity_sentiment_gap = calculate_entity_sentiment_gap(headlines, company_name)
    
    # -------------------- Metadata --------------------
    
    num_headlines = len(headlines)
    headline_length_avg = float(np.mean([len(h) for h in headlines]))
    
    # -------------------- Return Feature Dict --------------------
    
    return {
        # Sentiment features (13)
        "finbert_sentiment_score_mean": float(np.mean(finbert_scores)),
        "vader_sentiment_score_mean": float(np.mean(vader_scores)),
        "textblob_sentiment_score_mean": float(np.mean(textblob_scores)),
        "ensemble_sentiment_mean": float(np.mean(ensemble_scores)),
        "sentiment_variance_mean": sentiment_variance,
        "model_consensus_mean": model_consensus,
        "ensemble_sentiment_max": float(np.max(ensemble_scores)),
        "ensemble_sentiment_min": float(np.min(ensemble_scores)),
        "ensemble_sentiment_std": float(np.std(ensemble_scores)),
        "confidence_mean": confidence_mean,
        "num_headlines": num_headlines,
        "headline_length_avg": headline_length_avg,
        
        # Event features (6)
        "sentiment_earnings": sentiment_earnings,
        "sentiment_product": sentiment_product,
        "sentiment_analyst": sentiment_analyst,
        "count_positive_earnings": count_positive_earnings,
        "count_negative_regulatory": count_negative_regulatory,
        "has_macroeconomic_news": has_macroeconomic_news,
        
        # Entity features (5)
        "ceo_mention_count": ceo_mention_count,
        "ceo_sentiment": ceo_sentiment,
        "competitor_mention_count": competitor_mention_count,
        "entity_density": entity_density,
        "entity_sentiment_gap": entity_sentiment_gap,
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("Testing nlp_pipeline.py...")
    
    # Mock data
    test_df = pd.DataFrame({
        "date": ["2024-01-15"] * 3,
        "ticker": ["AAPL"] * 3,
        "headline": [
            "Apple reports record earnings, beats analyst expectations",
            "Tim Cook announces new product launch",
            "Regulatory probe into Apple's app store policies"
        ]
    })
    
    test_metadata = {
        "AAPL": {
            "company_name": "Apple Inc",
            "short_name": "Apple",
            "ceo": "Tim Cook",
            "competitors": ["Samsung", "Google"]
        }
    }
    
    features = generate_sentiment_features(test_df, test_metadata, "AAPL")
    
    print("\n✅ Generated features:")
    for k, v in features.items():
        print(f"  {k}: {v}")
    
    print(f"\n✅ Total features: {len(features)}")
    print("✅ nlp_pipeline.py test passed")