"""
Production-Level Synthetic Dataset Generator for Sale Predictor Model

This script generates realistic training data for the XGBoost-based SalePredictor.
It is designed for FYP/demo use, but follows production-style practices:

- Overlapping feature distributions for sale / no-sale calls
- Probabilistic labels (no hard rules)
- Built-in noise and contradictory cases
- Emotions kept as valid probability distributions
- Dataset validation and statistics
- Optional visualizations for your report

Output CSV is compatible with `train_sale_predictor.py` and `SalePredictor`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Ensure relative imports resolve when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("generate_sale_training_dataset")


FEATURE_NAMES: List[str] = [
    "sentiment_mean",
    "sentiment_variance",
    "emotion_neutral",
    "emotion_happiness",
    "emotion_anger",
    "emotion_sadness",
    "emotion_frustration",
    "silence_ratio",
    "interruption_frequency",
    "talk_listen_ratio",
    "turn_taking_frequency",
    "filler_word_frequency",
]

LABEL_COLUMN = "sale_outcome"


class RealisticSaleDatasetGenerator:
    """
    More realistic synthetic generator:

    - All calls are drawn from the same overlapping feature distributions.
    - Sale outcome is sampled from Bernoulli(p), where
        p = sigmoid(w^T x + bias + noise)
    - This naturally creates:
        * happy customers who don't buy
        * frustrated customers who still buy
        * messy calls that convert
        * perfect calls that fail
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = np.random.default_rng(random_seed)

        # Weights for logistic sale probability (signs reflect tendencies, not rules)
        # Values are moderate so noise can flip outcomes.
        self.w = np.array(
            [
                +2.0,  # sentiment_mean
                -0.5,  # sentiment_variance
                +0.5,  # emotion_neutral
                +1.5,  # emotion_happiness
                -1.0,  # emotion_anger
                -0.7,  # emotion_sadness
                -1.0,  # emotion_frustration
                -1.2,  # silence_ratio
                -0.8,  # interruption_frequency
                +0.6,  # talk_listen_ratio
                +0.8,  # turn_taking_frequency
                -0.5,  # filler_word_frequency
            ],
            dtype=float,
        )

        # Bias term roughly controls global conversion rate
        self.bias = -0.5

    # ------------------------------------------------------------------
    # Feature sampling
    # ------------------------------------------------------------------

    def _sample_base_features(self, n_samples: int) -> np.ndarray:
        """
        Sample raw features from overlapping distributions.

        No label is used here; *all* calls share the same feature space.
        """
        F = np.zeros((n_samples, len(FEATURE_NAMES)), dtype=float)

        # sentiment_mean: mostly between -0.5 and 0.8
        F[:, 0] = np.clip(
            self.rng.normal(loc=0.1, scale=0.35, size=n_samples), -1.0, 1.0
        )

        # sentiment_variance: low to moderate volatility
        F[:, 1] = np.clip(
            self.rng.uniform(0.01, 0.25, size=n_samples), 0.0, 1.0
        )

        # Emotions: draw from a Dirichlet centered on mild polarity
        # [neutral, happiness, anger, sadness, frustration]
        alpha = np.array([2.0, 1.8, 1.0, 1.0, 1.2], dtype=float)
        E = self.rng.dirichlet(alpha, size=n_samples)
        F[:, 2:7] = E

        # silence_ratio: 0.05–0.6 (broad)
        F[:, 7] = np.clip(
            self.rng.beta(a=2.0, b=3.0, size=n_samples) * 0.6, 0.0, 1.0
        )

        # interruption_frequency: 0–0.5 (broad)
        F[:, 8] = np.clip(
            self.rng.beta(a=1.5, b=4.0, size=n_samples) * 0.5, 0.0, 1.0
        )

        # talk_listen_ratio: 0.2–2.5 (can be very unbalanced)
        F[:, 9] = np.clip(
            self.rng.normal(loc=1.0, scale=0.5, size=n_samples), 0.2, 2.5
        )

        # turn_taking_frequency: 0–0.8 (broad)
        F[:, 10] = np.clip(
            self.rng.beta(a=2.0, b=3.0, size=n_samples) * 0.8, 0.0, 1.0
        )

        # filler_word_frequency: 0–6 per minute (wide range)
        F[:, 11] = np.clip(
            self.rng.gamma(shape=1.5, scale=1.5, size=n_samples), 0.0, 6.0
        )

        return F

    def _add_correlated_noise(self, F: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        Add moderate Gaussian noise then fix constraints:
        - Emotions re-normalized to sum to 1
        - Clamp ranges for bounded features
        """
        noisy = F.copy()
        noisy += self.rng.normal(loc=0.0, scale=noise_level, size=noisy.shape)

        # Re-normalize emotion probabilities row-wise
        E = noisy[:, 2:7]
        E = np.clip(E, 0.0, None)
        row_sums = E.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        E = E / row_sums
        noisy[:, 2:7] = E

        # Clamp key ranges again
        noisy[:, 0] = np.clip(noisy[:, 0], -1.0, 1.0)   # sentiment_mean
        noisy[:, 1] = np.clip(noisy[:, 1], 0.0, 1.0)    # sentiment_variance
        noisy[:, 7] = np.clip(noisy[:, 7], 0.0, 1.0)    # silence_ratio
        noisy[:, 8] = np.clip(noisy[:, 8], 0.0, 1.0)    # interruption_frequency
        noisy[:, 9] = np.clip(noisy[:, 9], 0.0, 3.0)    # talk_listen_ratio
        noisy[:, 10] = np.clip(noisy[:, 10], 0.0, 1.0)  # turn_taking_frequency
        noisy[:, 11] = np.clip(noisy[:, 11], 0.0, 10.0) # filler_word_frequency

        return noisy

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    @staticmethod
    def _logistic(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_sale_probabilities(self, F: np.ndarray, label_noise: float) -> np.ndarray:
        """
        Compute p(sale) = sigmoid(w^T x + bias + eps), where eps ~ N(0, label_noise).

        label_noise controls how often \"bad\" calls still convert and vice versa.
        Higher label_noise → more overlap and contradictory cases.
        """
        linear = F @ self.w + self.bias
        eps = self.rng.normal(loc=0.0, scale=label_noise, size=linear.shape)
        logits = linear + eps
        return self._logistic(logits)

    def _calibrate_bias(
        self,
        F_probe: np.ndarray,
        target_sale_ratio: float,
        label_noise: float,
    ) -> None:
        """Roughly adjust bias so mean p(sale) ≈ target_sale_ratio."""
        p = self._compute_sale_probabilities(F_probe, label_noise=label_noise)
        current_ratio = float(p.mean())
        if not (0.0 < current_ratio < 1.0 and 0.0 < target_sale_ratio < 1.0):
            return

        # Shift bias in logit space
        desired_logit = np.log(target_sale_ratio / (1 - target_sale_ratio))
        current_logit = np.log(current_ratio / (1 - current_ratio))
        delta = desired_logit - current_logit
        self.bias += float(delta)
        logger.info(
            "Calibrated bias from current_ratio=%.3f to target=%.3f (delta=%.3f)",
            current_ratio,
            target_sale_ratio,
            delta,
        )

    def generate_dataset(
        self,
        n_samples: int,
        target_sale_ratio: float = 0.3,
        feature_noise: float = 0.05,
        label_noise: float = 0.8,
    ) -> pd.DataFrame:
        """
        Generate synthetic dataset with realistic overlap between classes.

        Args:
            n_samples: total number of examples
            target_sale_ratio: desired approximate fraction of sales (0–1).
            feature_noise: noise added to features
            label_noise: noise added before logistic (more noise → more contradictory cases)
        """
        # 1) Sample base features
        F = self._sample_base_features(n_samples)

        # 2) Add feature noise + renormalize emotions
        F = self._add_correlated_noise(F, noise_level=feature_noise)

        # 3) Calibrate bias using a probe subset
        probe = F[: min(2000, n_samples)]
        self._calibrate_bias(probe, target_sale_ratio=target_sale_ratio, label_noise=label_noise)

        # 4) Compute probabilities and sample labels
        p_sale = self._compute_sale_probabilities(F, label_noise=label_noise)
        y = self.rng.binomial(n=1, p=p_sale, size=n_samples)

        # 5) Build DataFrame
        data: Dict[str, List[float]] = {}
        for i, name in enumerate(FEATURE_NAMES):
            data[name] = F[:, i]
        data[LABEL_COLUMN] = y

        df = pd.DataFrame(data)

        # Shuffle
        df = df.sample(
            frac=1.0,
            random_state=int(self.rng.integers(1, 1_000_000)),
        ).reset_index(drop=True)

        return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, object]:
    """
    Validate dataset quality and return statistics.
    """
    logger.info("Validating dataset...")

    results: Dict[str, object] = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "statistics": {},
    }

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        results["errors"].append(
            f"Missing values: {missing[missing > 0].to_dict()}"
        )
        results["valid"] = False

    # Check required features
    for feature in FEATURE_NAMES:
        if feature not in df.columns:
            results["errors"].append(f"Missing feature column: {feature}")
            results["valid"] = False

    if not results["valid"]:
        return results

    # Label distribution
    label_counts = df[LABEL_COLUMN].value_counts(dropna=False)
    sale_ratio = label_counts.get(1, 0) / max(len(df), 1)
    results["statistics"]["label_distribution"] = label_counts.to_dict()
    results["statistics"]["sale_ratio"] = sale_ratio

    # Feature correlations with label
    correlations: Dict[str, float] = {}
    for feature in FEATURE_NAMES:
        correlations[feature] = float(df[feature].corr(df[LABEL_COLUMN]))
    results["statistics"]["feature_correlations"] = correlations

    # Mean differences
    sale_mean = df[df[LABEL_COLUMN] == 1][FEATURE_NAMES].mean()
    no_sale_mean = df[df[LABEL_COLUMN] == 0][FEATURE_NAMES].mean()
    mean_diff = {
        f: float(sale_mean[f] - no_sale_mean[f]) for f in FEATURE_NAMES
    }
    results["statistics"]["mean_differences"] = mean_diff

    logger.info("Validation finished")
    return results


def generate_statistics_report(df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate JSON stats + PNG plots for the synthetic dataset.
    """
    os.makedirs(output_dir, exist_ok=True)

    stats: Dict[str, object] = {
        "dataset_size": int(len(df)),
        "num_features": int(len(FEATURE_NAMES)),
        "label_distribution": df[LABEL_COLUMN].value_counts().to_dict(),
        "sale_ratio": float(df[LABEL_COLUMN].mean()),
        "feature_statistics": {},
        "feature_correlations": {},
    }

    for feature in FEATURE_NAMES:
        series = df[feature]
        stats["feature_statistics"][feature] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
        }

    for feature in FEATURE_NAMES:
        stats["feature_correlations"][feature] = float(
            df[feature].corr(df[LABEL_COLUMN])
        )

    stats_path = os.path.join(output_dir, "dataset_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved statistics JSON to %s", stats_path)

    # Visualizations
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # 1) Feature distributions by label
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, feature in enumerate(FEATURE_NAMES):
        ax = axes[i]
        df[df[LABEL_COLUMN] == 1][feature].hist(
            bins=40, alpha=0.5, label="Sale", ax=ax
        )
        df[df[LABEL_COLUMN] == 0][feature].hist(
            bins=40, alpha=0.5, label="No Sale", ax=ax
        )
        ax.set_title(feature)
        ax.legend()
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", dist_path)

    # 2) Correlation heatmap
    corr_matrix = df[FEATURE_NAMES + [LABEL_COLUMN]].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", heatmap_path)

    # 3) Feature importance by absolute correlation
    corrs = stats["feature_correlations"]
    sorted_feats = sorted(
        FEATURE_NAMES, key=lambda f: abs(corrs[f]), reverse=True
    )
    plt.figure(figsize=(10, 6))
    plt.barh(
        range(len(sorted_feats)),
        [corrs[f] for f in sorted_feats],
        color="steelblue",
    )
    plt.yticks(range(len(sorted_feats)), sorted_feats)
    plt.xlabel("Correlation with sale_outcome")
    plt.title("Feature Importance (Correlation with Label)")
    plt.tight_layout()
    importance_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved %s", importance_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a realistic synthetic dataset for the SalePredictor model "
            "with overlapping feature distributions and probabilistic labels."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 10,000 samples with realistic 30%% sale rate
  python generate_sale_training_dataset.py \\
    --n_samples 10000 \\
    --sale_ratio 0.3 \\
    --output data/sale_training_data.csv \\
    --stats_dir data/dataset_stats

  # Balanced dataset (50/50) for analysis
  python generate_sale_training_dataset.py \\
    --n_samples 5000 \\
    --sale_ratio 0.5 \\
    --output data/sale_training_balanced.csv
""",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Total number of samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--sale_ratio",
        type=float,
        default=0.3,
        help="Approximate ratio of sale=1 samples (0–1, default: 0.3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sale_training_data.csv",
        help="Output CSV path (default: data/sale_training_data.csv)",
    )
    parser.add_argument(
        "--stats_dir",
        type=str,
        default=None,
        help="Directory for dataset statistics and plots (default: alongside output)",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--feature_noise",
        type=float,
        default=0.05,
        help="Feature noise level (default: 0.05)",
    )
    parser.add_argument(
        "--label_noise",
        type=float,
        default=0.8,
        help="Label noise level controlling contradictory cases (default: 0.8)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (0.0 < args.sale_ratio < 1.0):
        logger.error("sale_ratio must be strictly between 0 and 1")
        sys.exit(1)

    if args.n_samples < 200:
        logger.warning(
            "n_samples < 200: dataset may be too small for stable training"
        )

    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        logger.error(
            "Output file %s already exists. Use --overwrite to replace it.",
            args.output,
        )
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_dir = (
        Path(args.stats_dir)
        if args.stats_dir
        else output_path.parent / "dataset_stats"
    )

    logger.info("=" * 72)
    logger.info("Realistic Synthetic Dataset Generator for Sale Predictor")
    logger.info("=" * 72)
    logger.info("Configuration:")
    logger.info("  n_samples     : %d", args.n_samples)
    logger.info("  sale_ratio    : %.3f", args.sale_ratio)
    logger.info("  random_seed   : %d", args.random_seed)
    logger.info("  feature_noise : %.3f", args.feature_noise)
    logger.info("  label_noise   : %.3f", args.label_noise)
    logger.info("  output        : %s", args.output)
    logger.info("  stats_dir     : %s", stats_dir)

    generator = RealisticSaleDatasetGenerator(random_seed=args.random_seed)

    try:
        df = generator.generate_dataset(
            n_samples=args.n_samples,
            target_sale_ratio=args.sale_ratio,
            feature_noise=args.feature_noise,
            label_noise=args.label_noise,
        )

        validation = validate_dataset(df)
        if validation["errors"]:
            logger.error("Dataset validation failed with errors:")
            for err in validation["errors"]:
                logger.error("  - %s", err)
            sys.exit(1)

        if validation["warnings"]:
            logger.warning("Dataset validation warnings:")
            for w in validation["warnings"]:
                logger.warning("  - %s", w)

        # Save CSV
        df.to_csv(output_path, index=False)
        logger.info("Saved %d samples to %s", len(df), args.output)

        # Save statistics + plots
        generate_statistics_report(df, str(stats_dir))

        # Summary for console
        stats = validation["statistics"]
        logger.info("\nSummary:")
        logger.info("  Dataset size   : %d", len(df))
        logger.info(
            "  Label dist     : %s", stats.get("label_distribution", {})
        )
        logger.info("  Achieved ratio : %.3f", stats.get("sale_ratio", 0.0))

        logger.info("\nTop feature correlations with sale_outcome:")
        corrs: Dict[str, float] = stats.get("feature_correlations", {})
        for feat, corr in sorted(
            corrs.items(), key=lambda x: abs(x[1]), reverse=True
        )[:5]:
            logger.info("  %-25s: %+0.3f", feat, corr)

        logger.info("\nGeneration completed successfully.")
        logger.info("You can now train the model with:")
        logger.info(
            "  python backend/scripts/train_sale_predictor.py "
            f"--csv_path {args.output} --output_dir backend/models/ "
            "--early_stopping_rounds 20 --optimize_threshold"
        )

    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Dataset generation failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


