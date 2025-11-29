"""
evaluation.py
=============
Advanced evaluation metrics and plots for financial sentiment model.

Includes:
- Calibration plots
- Per-ticker performance breakdown
- Regime analysis
- Precision-Recall curves
- Feature stability analysis

Author: Rajveer Singh Pall
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    roc_curve,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.utils import log_info, log_warning


# ============================================================
# CALIBRATION ANALYSIS
# ============================================================

def evaluate_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, any]:
    """
    Evaluate model calibration (probability reliability).
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with calibration metrics and curve data
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    return {
        'prob_true': prob_true,
        'prob_pred': prob_pred,
        'brier_score': brier,
        'n_bins': n_bins,
    }


def plot_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[Path] = None,
    n_bins: int = 10
) -> None:
    """
    Plot calibration curve.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        save_path: Optional path to save figure
        n_bins: Number of bins
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Model', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration', color='gray')
    plt.xlabel('Predicted Probability', fontsize=12)
    plt.ylabel('True Frequency', fontsize=12)
    plt.title(f'Calibration Plot (Brier Score: {brier:.4f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        log_info(f"Saved calibration plot to {save_path}", "EVAL")
    
    plt.close()


# ============================================================
# PER-TICKER PERFORMANCE
# ============================================================

def evaluate_per_ticker(
    df: pd.DataFrame,
    y_true_col: str = 'movement',
    y_pred_col: str = 'prediction',
    ticker_col: str = 'ticker'
) -> pd.DataFrame:
    """
    Calculate performance metrics per ticker.
    
    Args:
        df: DataFrame with predictions and true labels
        y_true_col: Column name for true labels
        y_pred_col: Column name for predictions
        ticker_col: Column name for ticker
        
    Returns:
        DataFrame with metrics per ticker
    """
    ticker_metrics = []
    
    for ticker in df[ticker_col].unique():
        ticker_df = df[df[ticker_col] == ticker]
        
        if len(ticker_df) < 5:  # Skip tickers with too few samples
            continue
        
        y_true = ticker_df[y_true_col]
        y_pred = ticker_df[y_pred_col]
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate additional metrics if probabilities available
        metrics = {
            'ticker': ticker,
            'n_samples': len(ticker_df),
            'accuracy': accuracy,
        }
        
        if 'probability' in ticker_df.columns:
            y_proba = ticker_df['probability']
            metrics['brier_score'] = brier_score_loss(y_true, y_proba)
            metrics['mean_probability'] = float(y_proba.mean())
        
        ticker_metrics.append(metrics)
    
    return pd.DataFrame(ticker_metrics)


# ============================================================
# REGIME ANALYSIS
# ============================================================

def evaluate_by_regime(
    df: pd.DataFrame,
    y_true_col: str = 'movement',
    y_pred_col: str = 'prediction',
    volatility_col: str = 'volatility_lag1'
) -> pd.DataFrame:
    """
    Evaluate model performance across different volatility regimes.
    
    Args:
        df: DataFrame with predictions
        y_true_col: Column name for true labels
        y_pred_col: Column name for predictions
        volatility_col: Column name for volatility feature
        
    Returns:
        DataFrame with metrics per regime
    """
    # Define regimes: low, medium, high volatility
    df = df.copy()
    df['regime'] = pd.cut(
        df[volatility_col],
        bins=3,
        labels=['low_vol', 'medium_vol', 'high_vol']
    )
    
    regime_metrics = []
    
    for regime in ['low_vol', 'medium_vol', 'high_vol']:
        regime_df = df[df['regime'] == regime]
        
        if len(regime_df) < 10:
            continue
        
        y_true = regime_df[y_true_col]
        y_pred = regime_df[y_pred_col]
        
        accuracy = accuracy_score(y_true, y_pred)
        
        metrics = {
            'regime': regime,
            'n_samples': len(regime_df),
            'accuracy': accuracy,
            'mean_volatility': float(regime_df[volatility_col].mean()),
        }
        
        if 'probability' in regime_df.columns:
            y_proba = regime_df['probability']
            metrics['brier_score'] = brier_score_loss(y_true, y_proba)
        
        regime_metrics.append(metrics)
    
    return pd.DataFrame(regime_metrics)


# ============================================================
# COMPREHENSIVE EVALUATION
# ============================================================

def comprehensive_evaluation(
    df: pd.DataFrame,
    y_true_col: str = 'movement',
    y_pred_col: str = 'prediction',
    y_proba_col: Optional[str] = 'probability',
    ticker_col: str = 'ticker',
    output_dir: Optional[Path] = None
) -> Dict[str, any]:
    """
    Run comprehensive evaluation suite.
    
    Args:
        df: DataFrame with predictions and true labels
        y_true_col: Column name for true labels
        y_pred_col: Column name for predictions
        y_proba_col: Column name for probabilities (optional)
        ticker_col: Column name for ticker
        output_dir: Optional directory to save plots
        
    Returns:
        Dictionary with all evaluation results
    """
    results = {}
    
    y_true = df[y_true_col]
    y_pred = df[y_pred_col]
    
    # Basic metrics
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['classification_report'] = classification_report(y_true, y_pred)
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # Calibration (if probabilities available)
    if y_proba_col and y_proba_col in df.columns:
        y_proba = df[y_proba_col]
        calibration_data = evaluate_calibration(y_true, y_proba)
        results['calibration'] = calibration_data
        
        if output_dir:
            plot_calibration(y_true, y_proba, output_dir / 'calibration_plot.png')
    
    # Per-ticker breakdown
    ticker_metrics = evaluate_per_ticker(df, y_true_col, y_pred_col, ticker_col)
    results['per_ticker'] = ticker_metrics
    
    # Regime analysis
    if 'volatility_lag1' in df.columns:
        regime_metrics = evaluate_by_regime(df, y_true_col, y_pred_col)
        results['by_regime'] = regime_metrics
    
    # Precision-Recall curve
    if y_proba_col and y_proba_col in df.columns:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        results['pr_auc'] = pr_auc
        results['pr_curve'] = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'thresholds': thresholds.tolist(),
        }
    
    log_info("Comprehensive evaluation complete", "EVAL")
    
    return results


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":
    print("Testing evaluation.py...")
    
    # Create mock data
    np.random.seed(42)
    n_samples = 1000
    
    mock_df = pd.DataFrame({
        'ticker': np.random.choice(['AAPL', 'TSLA', 'NVDA'], n_samples),
        'movement': np.random.choice([0, 1], n_samples),
        'prediction': np.random.choice([0, 1], n_samples),
        'probability': np.random.rand(n_samples),
        'volatility_lag1': np.random.uniform(0.1, 0.5, n_samples),
    })
    
    # Test per-ticker evaluation
    ticker_metrics = evaluate_per_ticker(mock_df)
    print(f"✅ Per-ticker evaluation: {len(ticker_metrics)} tickers")
    
    # Test regime analysis
    regime_metrics = evaluate_by_regime(mock_df)
    print(f"✅ Regime analysis: {len(regime_metrics)} regimes")
    
    # Test calibration
    calibration_data = evaluate_calibration(
        mock_df['movement'].values,
        mock_df['probability'].values
    )
    print(f"✅ Calibration: Brier score = {calibration_data['brier_score']:.4f}")
    
    print("\n✅ evaluation.py test passed")

