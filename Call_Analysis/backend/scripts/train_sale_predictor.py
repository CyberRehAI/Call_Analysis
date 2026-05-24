"""
Training script for SalePredictor (XGBoost) using CSV with fused feature vectors.

This script:
1. Loads CSV file with pre-extracted fused feature vectors
2. Validates feature columns match expected format
3. Trains XGBoost classifier for sale prediction
4. Saves trained model to backend/models/sale_model.pkl
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
import joblib
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path to import call_analysis modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.call_analysis.models import SalePredictor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Expected feature names (11 features total)
EXPECTED_FEATURES = [
    'sentiment_mean',
    'sentiment_variance',
    'emotion_neutral',
    'emotion_happiness',
    'emotion_anger',
    'emotion_sadness',
    'emotion_frustration',
    'silence_ratio',
    'interruption_frequency',
    'talk_listen_ratio',
    'turn_taking_frequency'
]

LABEL_COLUMN = 'sale_outcome'


def validate_csv_format(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate CSV format matches expected feature columns.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, missing_columns)
    """
    missing_cols = []
    
    # Check for required feature columns
    for feature in EXPECTED_FEATURES:
        if feature not in df.columns:
            missing_cols.append(feature)
    
    # Check for label column
    if LABEL_COLUMN not in df.columns:
        missing_cols.append(LABEL_COLUMN)
    
    is_valid = len(missing_cols) == 0
    return is_valid, missing_cols


def load_and_preprocess_data(csv_path: str, test_split: float = 0.15, 
                             validation_split: float = 0.15, 
                             use_scaling: bool = False) -> Tuple[np.ndarray, np.ndarray, 
                                                                  np.ndarray, np.ndarray,
                                                                  np.ndarray, np.ndarray,
                                                                  Optional[SimpleImputer],
                                                                  Optional[StandardScaler]]:
    """
    Load CSV and preprocess data for training.
    
    FIXED: No data leakage - imputation and scaling happen AFTER split.
    FIXED: Scaling is optional (XGBoost doesn't need it).
    
    Args:
        csv_path: Path to CSV file
        test_split: Test split ratio (default: 0.15)
        validation_split: Validation split ratio (default: 0.15)
        use_scaling: Whether to apply StandardScaler (default: False, XGBoost doesn't need it)
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, imputer, scaler)
        imputer and scaler are None if not used
    """
    logger.info(f"Loading CSV from {csv_path}")
    df = pd.read_csv(csv_path)
    
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Validate format
    is_valid, missing_cols = validate_csv_format(df)
    if not is_valid:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Extract features and labels BEFORE any preprocessing
    X = df[EXPECTED_FEATURES].values
    y = df[LABEL_COLUMN].values
    
    # Check label distribution (before split)
    unique_labels, counts = np.unique(y, return_counts=True)
    logger.info("Label distribution (full dataset):")
    for label, count in zip(unique_labels, counts):
        logger.info(f"  {LABEL_COLUMN}={label}: {count} ({100*count/len(y):.1f}%)")
    
    # Split data FIRST (before any preprocessing to prevent leakage)
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )
    
    # Second split: train vs val
    val_size_adjusted = validation_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Test samples: {len(X_test)}")
    
    # Handle missing values AFTER split (fit on train only)
    imputer = None
    if np.isnan(X_train).any() or np.isnan(X_val).any() or np.isnan(X_test).any():
        logger.warning("Found missing values. Fitting imputer on training data only.")
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)
        X_test = imputer.transform(X_test)
    
    # Optional feature scaling (XGBoost doesn't need it, but kept for compatibility)
    scaler = None
    if use_scaling:
        logger.info("Applying StandardScaler (XGBoost doesn't require scaling)")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        logger.info("Skipping feature scaling (XGBoost is scale-invariant)")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, imputer, scaler


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray, metric: str = 'f1') -> float:
    """
    Find optimal classification threshold on validation data.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'f1_weighted', 'recall')
        
    Returns:
        Optimal threshold value
    """
    from sklearn.metrics import f1_score, precision_recall_curve
    
    if metric == 'f1':
        # Find threshold that maximizes F1
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    elif metric == 'recall':
        # Find threshold that maximizes recall (for sales, we want high recall)
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_recall = 0.0
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            recall = recall_score(y_true, y_pred_thresh, zero_division=0)
            if recall > best_recall:
                best_recall = recall
                best_threshold = threshold
    else:
        best_threshold = 0.5
    
    return best_threshold


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None, 
                   threshold: float = 0.5) -> Dict:
    """
    Calculate evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (or None to compute from y_proba)
        y_proba: Predicted probabilities (optional, for ROC-AUC)
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Dictionary of metrics
    """
    # Compute predictions from probabilities if y_pred not provided
    if y_pred is None and y_proba is not None:
        y_pred = (y_proba >= threshold).astype(int)
    elif y_pred is None:
        raise ValueError("Either y_pred or y_proba must be provided")
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'threshold': threshold
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            metrics['roc_auc'] = 0.0
    
    return metrics


def plot_feature_importance(model: SalePredictor, output_path: Path):
    """Plot and save feature importance visualization."""
    try:
        importance_dict = model.get_feature_importance()
        top_features = importance_dict.get('top_features', [])
        
        if not top_features:
            logger.warning("No feature importance data available")
            return
        
        features = [f['feature'] for f in top_features]
        importances = [f['importance'] for f in top_features]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title('XGBoost Feature Importance')
        plt.tight_layout()
        
        importance_plot_path = output_path / 'sale_model_feature_importance.png'
        plt.savefig(importance_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance plot saved to {importance_plot_path}")
    except Exception as e:
        logger.warning(f"Could not create feature importance plot: {e}")


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, output_path: Path):
    """Plot and save ROC curve."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Sale Prediction')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        roc_plot_path = output_path / 'sale_model_roc_curve.png'
        plt.savefig(roc_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curve plot saved to {roc_plot_path}")
    except Exception as e:
        logger.warning(f"Could not create ROC curve plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Train SalePredictor (XGBoost) on CSV data')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with fused feature vectors (required)')
    parser.add_argument('--output_dir', type=str, default='backend/models/',
                        help='Output directory for model (default: backend/models/)')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Test split ratio (default: 0.15)')
    parser.add_argument('--validation_split', type=float, default=0.15,
                        help='Validation split ratio (default: 0.15)')
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='XGBoost n_estimators (default: 100)')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='XGBoost max_depth (default: 6)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='XGBoost learning_rate (default: 0.1)')
    parser.add_argument('--use_scaling', action='store_true',
                        help='Apply StandardScaler (XGBoost doesn\'t need it, default: False)')
    parser.add_argument('--early_stopping_rounds', type=int, default=20,
                        help='Early stopping rounds (default: 20)')
    parser.add_argument('--optimize_threshold', action='store_true',
                        help='Optimize classification threshold on validation set (default: False)')
    
    args = parser.parse_args()
    
    # Validate CSV path
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data (FIXED: no data leakage)
    X_train, y_train, X_val, y_val, X_test, y_test, imputer, scaler = load_and_preprocess_data(
        str(csv_path), args.test_split, args.validation_split, args.use_scaling
    )
    
    # Calculate class imbalance for scale_pos_weight
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    logger.info(f"Class imbalance: {neg_count} negatives, {pos_count} positives")
    logger.info(f"Using scale_pos_weight={scale_pos_weight:.2f} to handle imbalance")
    
    # Initialize model
    sale_predictor = SalePredictor()
    
    # Train model
    logger.info("Training XGBoost model...")
    logger.info(f"Hyperparameters: n_estimators={args.n_estimators}, "
               f"max_depth={args.max_depth}, learning_rate={args.learning_rate}, "
               f"scale_pos_weight={scale_pos_weight:.2f}")
    
    # Create XGBoost model with specified hyperparameters
    # FIXED: Added scale_pos_weight for class imbalance
    # FIXED: Added early_stopping_rounds
    xgb_model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        scale_pos_weight=scale_pos_weight,  # FIXED: Handle class imbalance
        random_state=42,
        eval_metric='logloss'
    )
    
    # Train model with early stopping (FIXED)
    logger.info(f"Training with early stopping (patience={args.early_stopping_rounds})")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=args.early_stopping_rounds,  # FIXED: Early stopping
        verbose=False
    )
    
    # Apply probability calibration (Platt scaling) for better calibrated probabilities
    logger.info("Applying probability calibration (Platt scaling)...")
    calibrated_model = CalibratedClassifierCV(xgb_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_val, y_val)  # Calibrate on validation set
    
    # FIXED: Properly initialize SalePredictor with all required attributes
    sale_predictor.model = calibrated_model  # Use calibrated model
    sale_predictor.feature_importance = xgb_model.feature_importances_.copy()  # Copy to avoid reference issues
    sale_predictor.feature_names = EXPECTED_FEATURES.copy()
    sale_predictor.is_trained = True
    sale_predictor.scaler = scaler  # Store scaler if used
    logger.info("Probability calibration applied (Platt scaling)")
    
    # Evaluate on validation set (using calibrated probabilities)
    logger.info("Evaluating on validation set...")
    y_val_proba = calibrated_model.predict_proba(X_val)[:, 1]
    
    # FIXED: Optimize threshold on validation set if requested
    optimal_threshold = 0.5
    if args.optimize_threshold:
        optimal_threshold = find_optimal_threshold(y_val, y_val_proba, metric='f1')
        logger.info(f"Optimal threshold on validation set: {optimal_threshold:.3f}")
    else:
        logger.info("Using default threshold: 0.5")
    
    y_val_pred = (y_val_proba >= optimal_threshold).astype(int)
    val_metrics = evaluate_model(y_val, y_val_pred, y_val_proba, threshold=optimal_threshold)
    
    logger.info("Validation Metrics:")
    for metric, value in val_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")
        else:
            logger.info(f"  {metric}: {value}")
    
    # Evaluate on test set (using same threshold and calibrated probabilities)
    logger.info("Evaluating on test set...")
    y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_proba, threshold=optimal_threshold)
    
    logger.info("Test Metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save model
    model_path = output_path / 'sale_model.pkl'
    sale_predictor.save_model(str(model_path))
    sale_predictor.threshold = optimal_threshold  # FIXED: Store threshold in model
    logger.info(f"Model saved to {model_path}")
    
    # Save imputer and scaler if used (needed for inference)
    if imputer is not None:
        imputer_path = output_path / 'sale_model_imputer.pkl'
        joblib.dump(imputer, imputer_path)
        logger.info(f"Imputer saved to {imputer_path}")
    
    if scaler is not None:
        scaler_path = output_path / 'sale_model_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    else:
        logger.info("No scaler used (XGBoost doesn't require scaling)")
    
    # Generate visualizations
    plot_feature_importance(sale_predictor, output_path)
    plot_roc_curve(y_test, y_test_proba, output_path)
    
    # Save evaluation results
    results = {
        'validation_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                               for k, v in val_metrics.items()},
        'test_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                        for k, v in test_metrics.items()},
        'hyperparameters': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'scale_pos_weight': float(scale_pos_weight),
            'early_stopping_rounds': args.early_stopping_rounds,
            'use_scaling': args.use_scaling,
            'optimal_threshold': float(optimal_threshold)
        },
        'feature_names': EXPECTED_FEATURES,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'class_distribution': {
            'train_positives': int(pos_count),
            'train_negatives': int(neg_count),
            'train_imbalance_ratio': float(scale_pos_weight)
        }
    }
    
    results_path = output_path / 'sale_training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Training results saved to {results_path}")
    
    logger.info("Training completed!")


if __name__ == '__main__':
    main()

