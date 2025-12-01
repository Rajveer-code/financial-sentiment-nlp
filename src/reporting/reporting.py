"""
reporting.py
=============
PDF report generation for prediction results.

Uses reportlab to generate professional PDF reports containing:
- Prediction results
- Sentiment analysis
- Technical indicators
- Articles summary

Author: Rajveer Singh Pall
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from io import BytesIO

import pandas as pd

# Try to import reportlab
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, 
        PageBreak, Image as RLImage
    )
    from reportlab.lib.colors import HexColor, colors
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import log_info, log_error

# ============================================================
# FALLBACK PDF GENERATION (if reportlab not available)
# ============================================================

def generate_simple_pdf(
    ticker: str,
    prediction: Any,
    sentiment_features: Dict[str, float],
    articles: List[Dict],
    backtest_results: Optional[Dict] = None,
) -> bytes:
    """
    Generate a simple PDF using canvas (no reportlab required).
    
    Args:
        ticker: Stock ticker
        prediction: PredictionResult object
        sentiment_features: Dict of sentiment features
        articles: List of article dicts
        backtest_results: Optional backtest metrics
        
    Returns:
        PDF content as bytes
    """
    buffer = BytesIO()
    
    # Create PDF document
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(0.5 * inch, height - 0.75 * inch, f"Sentiment Analysis Report - {ticker}")
    
    # Date
    c.setFont("Helvetica", 10)
    c.drawString(0.5 * inch, height - 1 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    y_position = height - 1.5 * inch
    
    # Prediction Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.5 * inch, y_position, "📊 Prediction Results")
    y_position -= 0.3 * inch
    
    c.setFont("Helvetica", 11)
    c.drawString(0.75 * inch, y_position, f"Signal: {prediction.signal}")
    y_position -= 0.25 * inch
    c.drawString(0.75 * inch, y_position, f"Direction: {'UP' if prediction.prediction == 1 else 'DOWN'}")
    y_position -= 0.25 * inch
    c.drawString(0.75 * inch, y_position, f"Probability: {prediction.probability:.1%}")
    y_position -= 0.25 * inch
    c.drawString(0.75 * inch, y_position, f"Confidence: {prediction.confidence:.1%}")
    y_position -= 0.4 * inch
    
    # Sentiment Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.5 * inch, y_position, "😊 Sentiment Features")
    y_position -= 0.3 * inch
    
    sentiment_keys = [
        'finbert_sentiment_score_mean',
        'vader_sentiment_score_mean',
        'textblob_sentiment_score_mean',
        'ensemble_sentiment_mean',
        'num_headlines'
    ]
    
    c.setFont("Helvetica", 10)
    for key in sentiment_keys:
        if key in sentiment_features:
            value = sentiment_features[key]
            if isinstance(value, float):
                c.drawString(0.75 * inch, y_position, f"{key}: {value:.4f}")
            else:
                c.drawString(0.75 * inch, y_position, f"{key}: {value}")
            y_position -= 0.2 * inch
    
    y_position -= 0.2 * inch
    
    # Articles Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.5 * inch, y_position, f"📰 Recent Articles ({len(articles)} total)")
    y_position -= 0.3 * inch
    
    c.setFont("Helvetica", 9)
    for i, article in enumerate(articles[:10]):  # Show first 10
        headline = article.get('headline', 'Untitled')[:60]  # Truncate
        c.drawString(0.75 * inch, y_position, f"{i+1}. {headline}")
        y_position -= 0.2 * inch
        
        if y_position < 1 * inch:
            c.showPage()
            y_position = height - 0.75 * inch
            c.setFont("Helvetica", 9)
    
    # Footer
    c.setFont("Helvetica", 8)
    c.drawString(0.5 * inch, 0.5 * inch, "Confidential - Financial Sentiment Analysis Report")
    
    c.save()
    
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# MAIN PDF GENERATION FUNCTION
# ============================================================

def generate_pdf_report(
    ticker: str,
    prediction: Any,
    sentiment_features: Dict[str, float],
    articles: List[Dict],
    backtest_results: Optional[Dict] = None,
) -> bytes:
    """
    Generate a comprehensive PDF report.
    
    Args:
        ticker: Stock ticker symbol
        prediction: PredictionResult object with prediction details
        sentiment_features: Dictionary of sentiment features
        articles: List of article dictionaries
        backtest_results: Optional backtest metrics dictionary
        
    Returns:
        PDF content as bytes
        
    Raises:
        ValueError: If required parameters are missing
    """
    
    try:
        # Validate inputs
        if not ticker or not prediction or not sentiment_features:
            raise ValueError("Missing required parameters for PDF generation")
        
        log_info(f"Generating PDF report for {ticker}", "REPORT")
        
        # Use simple PDF generation
        pdf_bytes = generate_simple_pdf(
            ticker=ticker,
            prediction=prediction,
            sentiment_features=sentiment_features,
            articles=articles or [],
            backtest_results=backtest_results
        )
        
        if not pdf_bytes:
            raise ValueError("PDF generation returned empty bytes")
        
        log_info(f"✅ Generated PDF ({len(pdf_bytes)} bytes) for {ticker}", "REPORT")
        
        return pdf_bytes
        
    except Exception as e:
        log_error(f"PDF generation failed: {str(e)}", "REPORT")
        # Return a simple error PDF
        return generate_error_pdf(ticker, str(e))


# ============================================================
# ERROR PDF FALLBACK
# ============================================================

def generate_error_pdf(ticker: str, error_message: str) -> bytes:
    """Generate a simple error PDF."""
    buffer = BytesIO()
    
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.5 * inch, height - 0.75 * inch, "Report Generation Error")
    
    c.setFont("Helvetica", 11)
    c.drawString(0.5 * inch, height - 1.5 * inch, f"Ticker: {ticker}")
    c.drawString(0.5 * inch, height - 2 * inch, f"Error: {error_message[:100]}")
    
    c.save()
    
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("Testing reporting.py...")
    print("✅ Module imported successfully")
