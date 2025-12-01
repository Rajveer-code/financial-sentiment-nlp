"""
app_main.py (FIXED)
===================
Main Streamlit application with unified feature pipeline.

Key fixes:
- Uses nlp_pipeline.generate_sentiment_features() correctly
- No more broken imports or circular dependencies
- Proper ticker metadata loading with utils.safe_json_load()
- Aligned with FEATURE_SCHEMA

Author: Rajveer Singh Pall
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# IMPORTS (Fixed)
# ============================================================

from src.utils.utils import safe_json_load, log_info, log_error
from src.api_clients.news_api import fetch_news_dataframe_for_ticker
from src.api_clients.settings_ui import render_api_settings
from src.feature_engineering.nlp_pipeline import generate_sentiment_features
from src.modeling.models_backtest import PredictionEngine, quick_predict, compare_strategies
try:
    from src.reporting.reporting import generate_pdf_report
except ImportError:
    # Fallback if reporting module doesn't exist
    def generate_pdf_report(*args, **kwargs):
        return b"PDF generation not available"

# ============================================================
# LOAD TICKERS
# ============================================================

TICKER_FILE = PROJECT_ROOT / "config" / "tickers.json"
TICKER_METADATA = safe_json_load(TICKER_FILE)

log_info(f"Loaded {len(TICKER_METADATA)} tickers from metadata", "APP")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS (Preserved from original)
# ============================================================

def load_custom_css() -> None:
    """Load custom CSS for modern fintech UI."""
    st.markdown(
        """
    <style>
    :root {
        --bg-primary: #0B0E11;
        --bg-secondary: #1A1D23;
        --accent-cyan: #00C2FF;
        --accent-green: #00FF88;
        --accent-red: #FF4757;
        --text-primary: #E6E6E6;
    }
    .stApp {
        background: linear-gradient(135deg, #0B0E11 0%, #1A1D23 100%);
    }
    h1 {
        background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

# ============================================================
# INITIALIZATION
# ============================================================

def initialize_session_state() -> None:
    """Initialize session state variables."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = True

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = "AAPL"

    if "latest_prediction" not in st.session_state:
        st.session_state.latest_prediction = None

    if "latest_articles" not in st.session_state:
        st.session_state.latest_articles = None

    if "sentiment_features" not in st.session_state:
        st.session_state.sentiment_features = None

    if "default_confidence" not in st.session_state:
        st.session_state.default_confidence = 0.55

    if "default_lookback" not in st.session_state:
        st.session_state.default_lookback = 7


@st.cache_resource
def load_prediction_engine() -> Optional[PredictionEngine]:
    """Load and cache prediction engine."""
    try:
        with st.spinner("Loading prediction engine..."):
            engine = PredictionEngine(
                confidence_threshold=st.session_state.default_confidence
            )
            st.success("✅ Prediction engine loaded successfully")
            return engine
    except Exception as e:
        st.error(f"❌ Failed to load prediction engine: {str(e)}")
        import traceback
        st.error(f"Error details:\n{traceback.format_exc()}")
        return None

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_sentiment_badge(score: float) -> str:
    """Create colored sentiment badge HTML."""
    if score > 0.1:
        badge_class = "sentiment-positive"
        label = "BULLISH"
    elif score < -0.1:
        badge_class = "sentiment-negative"
        label = "BEARISH"
    else:
        badge_class = "sentiment-neutral"
        label = "NEUTRAL"

    return f'<span class="sentiment-badge {badge_class}">{label} ({score:.2f})</span>'


def get_fallback_realistic_data(ticker: str, days: int = 90):
    """Generate realistic fallback data for major tickers using approximate recent prices."""
    import numpy as np
    # Approximate recent prices (as of Dec 1, 2024) - realistic starting points
    ticker_prices = {
        "AAPL": 230,
        "MSFT": 415,
        "GOOGL": 172,
        "AMZN": 195,
        "NVDA": 135,
        "TSLA": 268,
        "META": 560,
        "NFLX": 285,
        "IBM": 215,
    }
    base_price = ticker_prices.get(ticker.upper(), np.random.uniform(100, 300))
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    # More realistic random walk with smaller steps
    close_prices = base_price + np.cumsum(np.random.randn(days) * 1.5)
    close_prices = np.maximum(close_prices, base_price * 0.5)  # Don't go below 50% of base
    
    # Generate realistic OHLC data with mixed red/green candles
    opens = close_prices + np.random.randn(days) * 1.2  # Open can be above or below close
    highs = np.maximum(opens, close_prices) + np.abs(np.random.randn(days) * 1.5)  # High is max of O/C + some
    lows = np.minimum(opens, close_prices) - np.abs(np.random.randn(days) * 1.5)  # Low is min of O/C - some
    
    df = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": close_prices,
        "Volume": np.random.randint(50000000, 150000000, days),
    }, index=dates)
    return df.sort_index()


def create_price_chart(ticker: str, days: int = 90):
    """Create interactive price chart with Plotly, return (fig, source_used, error_message)."""
    import yfinance as yf
    import time
    import requests

    # Simple in-memory cache per session to avoid repeated network calls
    cache_key = f"price_cache_{ticker}_{days}"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        df = cached
        source_used = st.session_state.get(f"price_cache_source_{ticker}_{days}", None)
        error_message = st.session_state.get(f"price_cache_error_{ticker}_{days}", None)
        if df is None or df.empty:
            return None, source_used, error_message, df
        try:
            fig = create_professional_candlestick(df, ticker)
            return fig, source_used, error_message, df
        except Exception as e:
            return None, source_used, f"Plotly error: {str(e)}", df
    else:
        max_retries = 3
        retry_delay = 1  # seconds

        df = pd.DataFrame()
        source_used = None
        error_message = None

        # Track yfinance failure counts per ticker and skip if repeatedly failing
        yf_fail_key = f"yf_fail_count_{ticker}"
        yf_fail_count = st.session_state.get(yf_fail_key, 0)

        if yf_fail_count < 2:
            for attempt in range(max_retries):
                try:
                    stock = yf.Ticker(ticker)
                    df = stock.history(period=f"{days}d")
                    if df is None or df.empty:
                        yf_fail_count += 1
                        st.session_state[yf_fail_key] = yf_fail_count
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            continue
                        else:
                            df = pd.DataFrame()
                            break
                    # success — reset fail count
                    st.session_state[yf_fail_key] = 0
                    break
                except Exception as e:
                    # increment and possibly backoff
                    yf_fail_count += 1
                    st.session_state[yf_fail_key] = yf_fail_count
                    log_error(f"yfinance error for {ticker}: {str(e)}", "APP")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    else:
                        df = pd.DataFrame()
                        break
        else:
            # Skip yfinance because of repeated failures
            log_info(f"Skipping yfinance for {ticker} due to repeated failures", "APP")

        # If yfinance failed or returned empty, try Finnhub (daily candles, free)
        if df.empty:
            try:
                from src.utils.api_key_manager import get_api_key
                finnhub_key = get_api_key("finnhub")
                if finnhub_key:
                    finnhub_url = f"https://finnhub.io/api/v1/stock/candle"
                    import datetime
                    end_time = int(datetime.datetime.now().timestamp())
                    start_time = end_time - days * 24 * 60 * 60
                    params = {
                        "symbol": ticker,
                        "resolution": "D",
                        "from": start_time,
                        "to": end_time,
                        "token": finnhub_key,
                    }
                    resp = requests.get(finnhub_url, params=params, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    if data.get("s") == "ok":
                        df = pd.DataFrame({
                            "Open": data["o"],
                            "High": data["h"],
                            "Low": data["l"],
                            "Close": data["c"],
                            "Volume": data["v"],
                        }, index=pd.to_datetime(data["t"], unit="s"))
                        df = df.sort_index()
                        source_used = "finnhub"
                        error_message = None
                    else:
                        df = pd.DataFrame()
                        source_used = "finnhub"
                        error_message = f"Finnhub: {data.get('error', 'No data returned')}"
                else:
                    df = pd.DataFrame()
                    source_used = "finnhub"
                    error_message = "No Finnhub API key found."
            except Exception as e:
                df = pd.DataFrame()
                source_used = "finnhub"
                error_message = f"Finnhub exception: {str(e)}"

        # If Finnhub also fails, try AlphaVantage intraday (minute-level, free)
        if df.empty:
            try:
                from src.utils.api_key_manager import get_api_key
                av_key = get_api_key("alphavantage")
                # respect temporary block for AlphaVantage if rate-limited
                av_block_key = f"av_block_until_{ticker}"
                av_block_until = st.session_state.get(av_block_key, 0)
                now_ts = time.time()
                if av_block_until and av_block_until > now_ts:
                    df = pd.DataFrame()
                    source_used = "alphavantage_intraday"
                    error_message = f"AlphaVantage temporarily blocked until {time.ctime(av_block_until)} due to rate limits."
                elif av_key:
                    BASE = "https://www.alphavantage.co/query"
                    params = {
                        "function": "TIME_SERIES_INTRADAY",
                        "symbol": ticker,
                        "interval": "5min",
                        "outputsize": "compact",
                        "apikey": av_key,
                    }
                    resp = requests.get(BASE, params=params, timeout=10)
                    resp.raise_for_status()
                    data = resp.json()
                    # Detect rate-limit / premium / note responses
                    if isinstance(data, dict) and ("Information" in data or "Note" in data or "Error Message" in data):
                        # block further AlphaVantage attempts for some time
                        block_secs = 60 * 60  # 1 hour
                        st.session_state[av_block_key] = now_ts + block_secs
                        df = pd.DataFrame()
                        source_used = "alphavantage_intraday"
                        error_message = f"AlphaVantage: {data.get('Information') or data.get('Note') or data.get('Error Message')}"
                        log_error(f"AlphaVantage blocked for {ticker}: {error_message}", "APP")
                    else:
                        ts_key = None
                        for k in data.keys():
                            if "Time Series" in k:
                                ts_key = k
                                break
                        if ts_key:
                            records = data[ts_key]
                            df = pd.DataFrame.from_dict(records, orient="index")
                            if '1. open' in df.columns:
                                df['Open'] = pd.to_numeric(df['1. open'], errors='coerce')
                            if '2. high' in df.columns:
                                df['High'] = pd.to_numeric(df['2. high'], errors='coerce')
                            if '3. low' in df.columns:
                                df['Low'] = pd.to_numeric(df['3. low'], errors='coerce')
                            if '4. close' in df.columns:
                                df['Close'] = pd.to_numeric(df['4. close'], errors='coerce')
                            if '5. volume' in df.columns:
                                df['Volume'] = pd.to_numeric(df['5. volume'], errors='coerce')
                            df.index = pd.to_datetime(df.index)
                            df = df.sort_index()
                            # Only keep the last N rows (up to days*78 for 5min bars)
                            if len(df) > days * 78:
                                df = df.iloc[-days*78:]
                            source_used = "alphavantage_intraday"
                            error_message = None
                        else:
                            df = pd.DataFrame()
                            source_used = "alphavantage_intraday"
                            error_message = f"AlphaVantage Intraday: No time series key found. Response: {data}"
                else:
                    df = pd.DataFrame()
                    source_used = "alphavantage_intraday"
                    error_message = "No AlphaVantage API key found."
            except Exception as e:
                df = pd.DataFrame()
                source_used = "alphavantage_intraday"
                error_message = f"AlphaVantage Intraday exception: {str(e)}"
        
        # Last resort: if all real sources fail, generate realistic fallback data for UI testing
        if df.empty:
            try:
                log_info(f"All real sources failed for {ticker}; generating realistic fallback data", "APP")
                df = get_fallback_realistic_data(ticker, days)
                source_used = "fallback_data"
                error_message = "All live data sources unavailable; displaying fallback data (realistic prices for testing only)."
                log_info(f"Generated fallback data for {ticker}: {len(df)} days", "APP")
            except Exception as e:
                log_error(f"Fallback data generation failed for {ticker}: {str(e)}", "APP")
                df = pd.DataFrame()
                source_used = "fallback_data"
                error_message = f"Fallback data generation failed: {str(e)}"
        
        # If yfinance worked
        if not df.empty and source_used is None:
            source_used = "yfinance"
            error_message = None

        # Cache result in session state
        st.session_state[cache_key] = df
        st.session_state[f"price_cache_source_{ticker}_{days}"] = source_used
        st.session_state[f"price_cache_error_{ticker}_{days}"] = error_message

    # If still empty, return None and error info
    if df is None or df.empty:
        return None, source_used, error_message, df

    try:
        fig = create_professional_candlestick(df, ticker)
        return fig, source_used, error_message, df
    except Exception as e:
        return None, source_used, f"Plotly error: {str(e)}", df


def create_feature_importance_chart(features: Dict[str, float], top_n: int = 10) -> go.Figure:
    """Create horizontal bar chart of top features."""
    
    # Sort by absolute value
    sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_features[:top_n]

    feature_names = [f[0] for f in top_features]
    feature_values = [f[1] for f in top_features]

    colors = ["#00FF88" if v > 0 else "#FF4757" for v in feature_values]

    fig = go.Figure(
        go.Bar(
            x=feature_values,
            y=feature_names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3f}" for v in feature_values],
            textposition="outside",
        )
    )

    fig.update_layout(
        title=f"Top {top_n} Feature Values",
        xaxis_title="Feature Value",
        template="plotly_dark",
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26, 29, 35, 0.8)",
    )

    return fig

# --- Professional candlestick chart helper ---
def create_professional_candlestick(df, ticker):
    import plotly.graph_objects as go
    # Use green for up, red for down
    inc = df['Close'] >= df['Open']
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='#00FF88',
        decreasing_line_color='#FF4757',
        increasing_fillcolor='rgba(0,255,136,0.3)',
        decreasing_fillcolor='rgba(255,71,87,0.3)',
        line_width=1.5,
        opacity=0.95,
        name='Price',
        showlegend=False
    ))
    fig.update_layout(
        title=f"{ticker} Price Chart (Free Tier)",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        template="plotly_dark",
        height=520,
        margin=dict(l=30, r=30, t=60, b=30),
        font=dict(family="Inter, Roboto, Arial", size=15, color="#E6E6E6"),
        plot_bgcolor="#181B20",
        paper_bgcolor="#181B20",
        xaxis=dict(
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor="#23272f",
            tickformat="%b %d",
            tickfont=dict(size=13),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#23272f",
            tickfont=dict(size=13),
        ),
        hovermode="x unified",
    )
    fig.update_traces(
        increasing_line_color='#00FF88',
        decreasing_line_color='#FF4757',
        selector=dict(type='candlestick'),
    )
    return fig

# ============================================================
# DASHBOARD TAB
# ============================================================

def render_dashboard_tab() -> None:
    """Main dashboard with predictions and news."""
    st.markdown("## 📊 Live Market Dashboard")
    
    # Model info sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Info")
    st.sidebar.markdown(f"""
    - **Version**: v1.2.3
    - **Trained**: 2025-01-15
    - **Features**: 42
    - **Confidence Threshold**: {st.session_state.default_confidence:.2f}
    """)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        ticker_options = list(TICKER_METADATA.keys())
        # Safely get index, default to 0 if ticker not found
        try:
            default_index = ticker_options.index(st.session_state.selected_ticker)
        except ValueError:
            default_index = 0
            st.session_state.selected_ticker = ticker_options[0]
        
        ticker = st.selectbox(
            "Select Ticker",
            options=ticker_options,
            index=default_index,
        )
        # Update selected ticker and clear stale results so UI fully refreshes
        previous = st.session_state.get("selected_ticker")
        st.session_state.selected_ticker = ticker
        if previous != ticker:
            st.session_state.latest_prediction = None
            st.session_state.latest_articles = None
            st.session_state.sentiment_features = None

    with col2:
        news_days = st.number_input(
            "News Lookback (days)", 1, 30, st.session_state.default_lookback
        )
        # Persist news lookback to session state for immediate effect
        st.session_state.default_lookback = int(news_days)

    with col3:
        confidence_threshold = st.slider(
            "Confidence Threshold", 0.5, 0.9, st.session_state.default_confidence, 0.05
        )
        # Persist confidence change immediately so sidebar and model reflect it
        st.session_state.default_confidence = float(confidence_threshold)

    # ============================================================
    # LIVE MARKET OVERVIEW - Show ONLY selected ticker
    # ============================================================
    st.markdown("---")
    st.markdown(f"### 📈 Live Market Overview - {ticker}")
    
        # Show a single, professional price chart for the selected ticker
    with st.spinner(f"Loading {ticker} price data..."):
        try:
            fig, source_used, error_message, df = create_price_chart(ticker, days=90)
            # Calculate available days from df
            available_days = 0
            date_range_str = ""
            if df is not None and not df.empty:
                available_days = (df.index.max() - df.index.min()).days + 1
                date_range_str = f"{df.index.min().strftime('%b %d, %Y')} to {df.index.max().strftime('%b %d, %Y')}"
            # Show free tier notice
            if available_days > 0:
                st.markdown(f"<div style='background: #23272f; color: #00C2FF; padding: 0.75em 1em; border-radius: 8px; margin-bottom: 0.5em; font-size: 1.1em; font-weight: 500;'>"
                            f"<b>Free data source:</b> Only <span style='color:#00FF88'>{available_days} days</span> available ({date_range_str}) due to free tier limitations."
                            f"</div>", unsafe_allow_html=True)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                # Show data source badge with appropriate styling
                if source_used == "fallback_data":
                    st.warning(f"⚠️ **FALLBACK DATA**: Live price APIs unavailable. Showing realistic placeholder data (based on approximate recent prices) for demo only.")
                elif source_used == "demo_data":
                    st.warning(f"⚠️ **DEMO DATA**: All live data sources failed. This chart uses synthetic test data for UI demonstration only.")
                else:
                    st.info(f"✅ Live data source: **{source_used}** | Date range: {date_range_str if date_range_str else 'See chart'}")
                if error_message:
                    st.caption(f"ℹ️ {error_message}")
            else:
                st.warning(f"⚠️ No price data available for {ticker}. Try a different date range or check your internet connection.")
                if source_used:
                    st.error(f"Data source attempted: {source_used}")
                if error_message:
                    st.error(f"Error: {error_message}")
        except Exception as e:
            st.error(f"⚠️ Failed to load chart for {ticker}: {str(e)}")

    st.markdown("---")

    # Fetch news + predict
    if st.button("🔄 Fetch Latest News & Predict", type="primary"):
        with st.spinner("Fetching news and analyzing sentiment..."):
            
            try:
                # Log for debugging
                log_info(f"Fetching news for {ticker} from last {int(news_days)} days", "APP")
                
                # Fetch news
                df_news = fetch_news_dataframe_for_ticker(
                    ticker=ticker,
                    ticker_metadata=TICKER_METADATA,
                    from_date=datetime.utcnow() - timedelta(days=int(news_days)),
                    to_date=datetime.utcnow(),
                    max_articles=40
                )
                
                log_info(f"Fetched {len(df_news)} articles for {ticker}", "APP")

                if df_news.empty:
                    st.warning("No articles found. Check API keys in Settings.")
                    log_info("NewsAPI returned empty result", "APP")
                    return
                
                st.success(f"✅ Fetched {len(df_news)} articles")
                
                # Store articles
                st.session_state.latest_articles = df_news.to_dict(orient="records")

                # Generate sentiment features
                try:
                    sentiment_features = generate_sentiment_features(
                        headlines_df=df_news,
                        ticker_metadata=TICKER_METADATA,
                        ticker=ticker
                    )
                except Exception as e:
                    st.error(f"❌ Sentiment generation failed: {str(e)}")
                    import traceback
                    st.error(f"Details:\n{traceback.format_exc()}")
                    return

                if not sentiment_features:
                    st.warning("Sentiment feature generation returned empty dict.")
                    return

                st.session_state.sentiment_features = sentiment_features
                st.success(f"✅ Generated sentiment features ({len(sentiment_features)} features)")

                # Load prediction engine
                try:
                    engine = load_prediction_engine()
                    if engine is None:
                        st.error("❌ Prediction engine failed to load.")
                        return
                except Exception as e:
                    st.error(f"❌ Error loading prediction engine: {str(e)}")
                    import traceback
                    st.error(f"Details:\n{traceback.format_exc()}")
                    return

                # Update engine confidence threshold
                engine.confidence_threshold = confidence_threshold

                # Make prediction
                try:
                    prediction = engine.predict(ticker, sentiment_features)
                    st.session_state.latest_prediction = prediction
                    st.success(f"✅ Prediction: {prediction.signal}")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    import traceback
                    st.error(f"Details:\n{traceback.format_exc()}")
                    return
                    
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
                import traceback
                st.error(f"Full traceback:\n{traceback.format_exc()}")
                return

    # Display prediction if available
    prediction = st.session_state.get("latest_prediction", None)
    if prediction is not None:
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if prediction.signal == "BUY":
                signal_color = "#00FF88"
            elif prediction.signal == "SELL":
                signal_color = "#FF4757"
            else:
                signal_color = "#A0A0A0"

            st.markdown(
                f"### Signal: <span style='color:{signal_color}'>{prediction.signal}</span>",
                unsafe_allow_html=True,
            )

        with col2:
            st.metric("Probability (UP)", f"{prediction.probability:.1%}")

        with col3:
            st.metric("Confidence", f"{prediction.confidence:.1%}")

        with col4:
            direction = "📈 UP" if prediction.prediction == 1 else "📉 DOWN"
            st.metric("Direction", direction)

        st.markdown("### 🎯 Key Features")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_features = create_feature_importance_chart(prediction.features)
            st.plotly_chart(fig_features, use_container_width=True)

        with col2:
            st.markdown("#### Sentiment Breakdown")
            features = st.session_state.get("sentiment_features", {}) or {}

            ensemble_score = float(features.get("ensemble_sentiment_mean", 0.0))
            st.markdown(
                create_sentiment_badge(ensemble_score),
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
            - **FinBERT**: {features.get('finbert_sentiment_score_mean', 0.0):.3f}
            - **VADER**: {features.get('vader_sentiment_score_mean', 0.0):.3f}
            - **TextBlob**: {features.get('textblob_sentiment_score_mean', 0.0):.3f}
            - **Headlines**: {features.get('num_headlines', 0)}
            - **CEO Sentiment**: {features.get('ceo_sentiment', 0.0):.3f}
            """
            )

    # News display
    articles = st.session_state.get("latest_articles", None)
    if articles:
        st.markdown("---")
        st.markdown("### 📰 Recent News")

        for i, article in enumerate(articles[:10]):
            title = article.get("headline") or "Untitled"
            with st.expander(f"📄 {title}", expanded=(i == 0)):
                st.markdown(f"**Source**: {article.get('source', 'Unknown')}")
                st.markdown(f"**Published**: {article.get('published_at', 'N/A')}")
                if article.get("url"):
                    st.markdown(f"[Read Full Article]({article['url']})")

    # PDF export
    if (
        st.session_state.get("latest_prediction") is not None
        and st.session_state.get("sentiment_features") is not None
        and st.session_state.get("latest_articles") is not None
    ):
        st.markdown("---")
        st.markdown("### 📄 Export Report")

        if st.button("Generate PDF Report", type="primary"):
            pdf_bytes = generate_pdf_report(
                ticker=st.session_state.selected_ticker,
                prediction=st.session_state.latest_prediction,
                sentiment_features=st.session_state.sentiment_features,
                articles=st.session_state.latest_articles,
                backtest_results=None,
            )

            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"{st.session_state.selected_ticker}_sentiment_report.pdf",
                mime="application/pdf"
            )

    # Single price chart is rendered above in Live Market Overview


# ============================================================
# SETTINGS TAB
# ============================================================

def render_settings_tab() -> None:
    """Application settings and configuration."""
    st.markdown("## ⚙️ Settings")

    render_api_settings()

    st.markdown("---")
    st.markdown("### 🤖 Model Settings")

    col1, col2 = st.columns(2)

    with col1:
        default_confidence = st.slider(
            "Default Confidence Threshold",
            0.50,
            0.90,
            st.session_state.default_confidence,
            0.05,
        )
        st.session_state.default_confidence = default_confidence

    with col2:
        default_lookback = st.number_input(
            "Default News Lookback (days)",
            1,
            30,
            st.session_state.default_lookback,
        )
        st.session_state.default_lookback = default_lookback


# ============================================================
# MAIN APP
# ============================================================

def main() -> None:
    """Main application entry point."""
    initialize_session_state()
    load_custom_css()

    # Sidebar
    with st.sidebar:
        st.markdown("## 📈 Sentiment Dashboard")
        st.markdown(
            "A research-grade app that fuses **news sentiment**, "
            "**NLP**, and **quant models**."
        )

        st.markdown("---")

        nav = st.radio(
            "Navigation",
            ["Dashboard", "Settings"],
            index=0,
        )

        st.markdown("---")

        meta = TICKER_METADATA.get(st.session_state.selected_ticker, {})
        st.markdown("### Selected Ticker")
        st.markdown(
            f"""
        - **Ticker**: `{st.session_state.selected_ticker}`
        - **Company**: {meta.get('company_name', 'N/A')}
        - **Sector**: {meta.get('sector', 'N/A')}
        - **CEO**: {meta.get('ceo', 'N/A')}
        """
        )

        st.markdown("---")
        st.markdown(
            "<div class='footer'>Built by <b>Rajveer Singh Pall</b></div>",
            unsafe_allow_html=True,
        )

    # Main content routing
    if not st.session_state.authenticated:
        st.error("You are not authenticated.")
        return

    st.markdown("# Financial Sentiment Analysis")

    if nav == "Dashboard":
        render_dashboard_tab()
    elif nav == "Settings":
        render_settings_tab()


if __name__ == "__main__":
    main()