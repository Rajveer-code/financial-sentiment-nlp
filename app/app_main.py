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
        return PredictionEngine(confidence_threshold=st.session_state.default_confidence)
    except Exception as e:
        st.error(f"Failed to load prediction engine: {e}")
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


def create_price_chart(ticker: str, days: int = 90) -> Optional[go.Figure]:
    """Create interactive price chart with Plotly."""
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=f"{days}d")

        if df.empty:
            return None

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        )

        fig.update_layout(
            title=f"{ticker} Price History ({days} days)",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=500,
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(26, 29, 35, 0.8)",
        )

        return fig

    except Exception as e:
        st.error(f"Failed to load price data: {e}")
        return None


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
        st.session_state.selected_ticker = ticker

    with col2:
        news_days = st.number_input(
            "News Lookback (days)", 1, 30, st.session_state.default_lookback
        )

    with col3:
        confidence_threshold = st.slider(
            "Confidence Threshold", 0.5, 0.9, st.session_state.default_confidence, 0.05
        )

    # Fetch news + predict
    if st.button("🔄 Fetch Latest News & Predict", type="primary"):
        with st.spinner("Fetching news and analyzing sentiment..."):
            
            # Fetch news
            df_news = fetch_news_dataframe_for_ticker(
                ticker=ticker,
                ticker_metadata=TICKER_METADATA,
                from_date=datetime.utcnow() - timedelta(days=int(news_days)),
                to_date=datetime.utcnow(),
                max_articles=40
            )

            if df_news.empty:
                st.warning("No articles found. Check API keys in Settings.")
                return
            
            st.success(f"✅ Fetched {len(df_news)} articles")
            
            # Store articles
            st.session_state.latest_articles = df_news.to_dict(orient="records")

            # Generate sentiment features
            sentiment_features = generate_sentiment_features(
                headlines_df=df_news,
                ticker_metadata=TICKER_METADATA,
                ticker=ticker
            )

            if not sentiment_features:
                st.warning("Sentiment feature generation failed.")
                return

            st.session_state.sentiment_features = sentiment_features

            # Load prediction engine
            engine = load_prediction_engine()
            if engine is None:
                st.error("Prediction engine not available.")
                return

            # Update engine confidence threshold
            engine.confidence_threshold = confidence_threshold

            # Predict
            prediction = engine.predict(ticker, sentiment_features)
            st.session_state.latest_prediction = prediction

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

    # Price chart
    st.markdown("---")
    fig_price = create_price_chart(st.session_state.selected_ticker, days=90)
    if fig_price:
        st.plotly_chart(fig_price, use_container_width=True)


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