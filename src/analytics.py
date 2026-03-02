"""
Analytics Engine for FMEA Multi-Model Benchmarking
===================================================
Handles variance analysis, consensus scoring, disagreement matrices,
confidence indicators, normalization, and action-trigger logic
across multiple LLM model outputs.

Mathematical basis:
  Mean RPN:   μ = Σ RPN_i / n
  Variance:   s² = Σ (RPN_i - μ)² / (n-1)
  StdDev:     σ  = √s²
  Confidence:  1 - (σ / max_possible_range)   (0..1 scale)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────
SCORE_MIN = 1
SCORE_MAX = 10
RPN_MIN = 1
RPN_MAX = 1000


# ────────────────────────────────────────────
# 1. Normalization helpers
# ────────────────────────────────────────────

def normalize_score(value: float, min_val: float = SCORE_MIN,
                    max_val: float = SCORE_MAX) -> float:
    """Clamp and normalize a score to the 1-10 scale."""
    return max(min_val, min(max_val, float(value)))


def normalize_rpn(value: float) -> float:
    """Clamp RPN into the 1-1000 range."""
    return max(RPN_MIN, min(RPN_MAX, float(value)))


def normalize_model_results(multi_model_results: Dict[str, pd.DataFrame]
                            ) -> Dict[str, pd.DataFrame]:
    """
    Ensure all model DataFrames have S, O, D on a 1-10 scale and
    RPN re-calculated as S*O*D.  Returns new (copied) DataFrames.
    """
    normalised = {}
    for model_name, df in multi_model_results.items():
        ndf = df.copy()
        for col in ["Severity", "Occurrence", "Detection"]:
            if col in ndf.columns:
                ndf[col] = ndf[col].apply(lambda v: normalize_score(v) if pd.notna(v) else v)
        # Re-derive RPN after normalization
        if all(c in ndf.columns for c in ["Severity", "Occurrence", "Detection"]):
            ndf["RPN"] = ndf.apply(
                lambda r: normalize_rpn(r["Severity"] * r["Occurrence"] * r["Detection"])
                if pd.notna(r["Severity"]) else np.nan, axis=1)
        normalised[model_name] = ndf
    return normalised


# ────────────────────────────────────────────
# 2. Core variance & disagreement functions
# ────────────────────────────────────────────

def calculate_fmea_variance(multi_model_results: Dict[str, pd.DataFrame],
                            metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculates mean, variance, std-dev, range for each failure mode across models.
    """
    if metrics is None:
        metrics = ["Severity", "Occurrence", "Detection", "RPN"]

    model_names = list(multi_model_results.keys())
    if len(model_names) < 2:
        return pd.DataFrame()

    ref_df = multi_model_results[model_names[0]]
    records: List[Dict[str, Any]] = []

    for idx in range(len(ref_df)):
        record: Dict[str, Any] = {
            "failure_mode": ref_df.iloc[idx].get(
                "Failure Mode", ref_df.iloc[idx].get("failure_mode", f"FM-{idx}")),
        }

        for metric in metrics:
            scores = []
            for mname in model_names:
                mdf = multi_model_results[mname]
                if idx < len(mdf):
                    val = mdf.iloc[idx].get(metric, None)
                    if val is not None and pd.notna(val):
                        try:
                            scores.append(float(val))
                        except (ValueError, TypeError):
                            pass

            if len(scores) >= 2:
                record[f"{metric}_mean"] = np.mean(scores)
                record[f"{metric}_variance"] = np.var(scores, ddof=1)
                record[f"{metric}_std"] = np.std(scores, ddof=1)
                record[f"{metric}_range"] = max(scores) - min(scores)
                record[f"{metric}_min"] = min(scores)
                record[f"{metric}_max"] = max(scores)
            else:
                val = scores[0] if scores else 0
                record[f"{metric}_mean"] = val
                record[f"{metric}_variance"] = 0
                record[f"{metric}_std"] = 0
                record[f"{metric}_range"] = 0
                record[f"{metric}_min"] = val
                record[f"{metric}_max"] = val

        records.append(record)

    return pd.DataFrame(records)


def generate_disagreement_matrix(multi_model_results: Dict[str, pd.DataFrame],
                                  metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Matrix: rows = failure modes, cols = metrics, values = std-dev across models.
    Directly usable with plotly.express.imshow for the disagreement heatmap.
    """
    if metrics is None:
        metrics = ["Severity", "Occurrence", "Detection", "RPN"]

    model_names = list(multi_model_results.keys())
    if len(model_names) < 2:
        return pd.DataFrame()

    ref_df = multi_model_results[model_names[0]]
    rows, labels = [], []

    for idx in range(len(ref_df)):
        fm = ref_df.iloc[idx].get(
            "Failure Mode", ref_df.iloc[idx].get("failure_mode", f"FM-{idx}"))
        labels.append(fm)

        row_stds = []
        for metric in metrics:
            scores = []
            for mname in model_names:
                mdf = multi_model_results[mname]
                if idx < len(mdf):
                    val = mdf.iloc[idx].get(metric, None)
                    if val is not None and pd.notna(val):
                        try:
                            scores.append(float(val))
                        except (ValueError, TypeError):
                            pass
            row_stds.append(np.std(scores, ddof=1) if len(scores) >= 2 else 0.0)
        rows.append(row_stds)

    df = pd.DataFrame(rows, columns=metrics, index=labels)
    df.index.name = "Failure Mode"
    return df


def generate_model_score_matrix(multi_model_results: Dict[str, pd.DataFrame],
                                 metric: str = "RPN") -> pd.DataFrame:
    """
    Matrix: rows = failure modes, columns = model names, values = scores.
    """
    model_names = list(multi_model_results.keys())
    ref_df = multi_model_results[model_names[0]]

    labels = []
    data = {m: [] for m in model_names}

    for idx in range(len(ref_df)):
        fm = ref_df.iloc[idx].get(
            "Failure Mode", ref_df.iloc[idx].get("failure_mode", f"FM-{idx}"))
        labels.append(fm)

        for mname in model_names:
            mdf = multi_model_results[mname]
            if idx < len(mdf):
                val = mdf.iloc[idx].get(metric, None)
                try:
                    data[mname].append(float(val) if val is not None and pd.notna(val) else 0)
                except (ValueError, TypeError):
                    data[mname].append(0)
            else:
                data[mname].append(0)

    df = pd.DataFrame(data, index=labels)
    df.index.name = "Failure Mode"
    return df


# ────────────────────────────────────────────
# 3. Consensus & Confidence scoring
# ────────────────────────────────────────────

def calculate_consensus_scores(multi_model_results: Dict[str, pd.DataFrame],
                                metrics: Optional[List[str]] = None
                                ) -> pd.DataFrame:
    """
    Produce a per-failure-mode confidence score (0..1) for each metric.

    Confidence = 1 - (σ / max_possible_range)
      where max_possible_range = 9  for S/O/D  and  999 for RPN.

    A value near 1 means all models agree; near 0 means extreme disagreement.
    """
    if metrics is None:
        metrics = ["Severity", "Occurrence", "Detection", "RPN"]

    max_ranges = {
        "Severity": SCORE_MAX - SCORE_MIN,     # 9
        "Occurrence": SCORE_MAX - SCORE_MIN,
        "Detection": SCORE_MAX - SCORE_MIN,
        "RPN": RPN_MAX - RPN_MIN,              # 999
    }

    variance_df = calculate_fmea_variance(multi_model_results, metrics)
    if variance_df.empty:
        return pd.DataFrame()

    confidence_records = []
    for _, row in variance_df.iterrows():
        rec: Dict[str, Any] = {"failure_mode": row["failure_mode"]}
        overall_scores = []
        for metric in metrics:
            std_val = row.get(f"{metric}_std", 0)
            max_range = max_ranges.get(metric, 9)
            confidence = max(0.0, 1.0 - (std_val / max_range))
            rec[f"{metric}_confidence"] = round(confidence, 3)
            overall_scores.append(confidence)

        rec["overall_confidence"] = round(np.mean(overall_scores), 3) if overall_scores else 1.0
        # Human-readable label
        oc = rec["overall_confidence"]
        rec["confidence_label"] = (
            "High" if oc >= 0.85 else
            "Medium" if oc >= 0.6 else
            "Low"
        )
        confidence_records.append(rec)

    return pd.DataFrame(confidence_records)


def calculate_average_agreement(multi_model_results: Dict[str, pd.DataFrame],
                                 agreement_threshold: float = 0.8
                                 ) -> Dict[str, Any]:
    """
    Overall benchmark summary:
      - average_confidence across all failure modes
      - pct_high_agreement: % of items at or above agreement_threshold
    """
    conf_df = calculate_consensus_scores(multi_model_results)
    if conf_df.empty:
        return {"average_confidence": 1.0, "pct_high_agreement": 100.0, "total_items": 0}

    avg = conf_df["overall_confidence"].mean()
    above = (conf_df["overall_confidence"] >= agreement_threshold).sum()
    total = len(conf_df)

    return {
        "average_confidence": round(float(avg), 3),
        "pct_high_agreement": round(float(above / total * 100), 1) if total else 100.0,
        "total_items": total,
        "high_count": int(above),
        "low_count": int(total - above),
    }


# ────────────────────────────────────────────
# 4. High-variance identification & action trigger
# ────────────────────────────────────────────

def identify_high_variance_items(variance_df: pd.DataFrame,
                                  threshold_std: float = 2.0,
                                  metric: str = "RPN") -> pd.DataFrame:
    """Filter failure modes where model disagreement exceeds a threshold."""
    std_col = f"{metric}_std"
    if std_col not in variance_df.columns:
        return pd.DataFrame()

    return variance_df[variance_df[std_col] >= threshold_std].sort_values(
        by=std_col, ascending=False
    )


def flag_for_expert_review(variance_df: pd.DataFrame,
                            variance_threshold: float = 2.5,
                            metric: str = "RPN") -> pd.DataFrame:
    """
    Action Trigger: if s² > threshold, add an 'Expert Review' flag.
    Returns the full DataFrame with an added 'expert_review_flag' column.
    """
    var_col = f"{metric}_variance"
    if var_col not in variance_df.columns:
        return variance_df

    df = variance_df.copy()
    df["expert_review_flag"] = df[var_col].apply(
        lambda v: "Manual Expert Review" if v > (variance_threshold ** 2) else "Auto-Approved"
    )
    return df


# ────────────────────────────────────────────
# 5. Field-level outlier detection
# ────────────────────────────────────────────

def identify_field_level_disagreements(
        multi_model_results: Dict[str, pd.DataFrame],
        severity_threshold: int = 3
) -> List[Dict[str, Any]]:
    """
    Identify failure modes where one model says 'Critical' (S>=8) while
    another says 'Minor' (S<=3).  Same logic applied to Occurrence and Detection.

    Args:
        multi_model_results: model_name -> DataFrame
        severity_threshold: minimum gap to flag

    Returns:
        List of disagreement detail dicts.
    """
    model_names = list(multi_model_results.keys())
    if len(model_names) < 2:
        return []

    ref_df = multi_model_results[model_names[0]]
    disagreements: List[Dict[str, Any]] = []

    for idx in range(len(ref_df)):
        fm = ref_df.iloc[idx].get(
            "Failure Mode", ref_df.iloc[idx].get("failure_mode", f"FM-{idx}"))

        for metric in ["Severity", "Occurrence", "Detection"]:
            model_scores: Dict[str, float] = {}
            for mname in model_names:
                mdf = multi_model_results[mname]
                if idx < len(mdf):
                    val = mdf.iloc[idx].get(metric, None)
                    if val is not None and pd.notna(val):
                        try:
                            model_scores[mname] = float(val)
                        except (ValueError, TypeError):
                            pass

            if len(model_scores) < 2:
                continue

            max_model = max(model_scores, key=model_scores.get)
            min_model = min(model_scores, key=model_scores.get)
            gap = model_scores[max_model] - model_scores[min_model]

            if gap >= severity_threshold:
                disagreements.append({
                    "failure_mode": fm,
                    "metric": metric,
                    "outlier_high_model": max_model,
                    "outlier_high_score": model_scores[max_model],
                    "outlier_low_model": min_model,
                    "outlier_low_score": model_scores[min_model],
                    "gap": gap,
                    "all_scores": model_scores,
                })

    return disagreements


# ────────────────────────────────────────────
# 6. Master benchmark analysis entry point
# ────────────────────────────────────────────

def analyze_benchmark_variance(results_list: List[Dict[str, Any]]
                                ) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Convenience function matching Gemini's suggested API.

    Args:
        results_list: list of dicts with keys 'severity', 'occurrence',
                      'detection', 'rpn' (one dict per model run).

    Returns:
        (stats_dict, variances_dict)
    """
    df = pd.DataFrame(results_list)

    for col in ["severity", "occurrence", "detection", "rpn"]:
        if col not in df.columns:
            df[col] = 0

    rpn_std = float(df["rpn"].std()) if len(df) >= 2 else 0.0

    stats = {
        "mean_rpn": round(float(df["rpn"].mean()), 2),
        "std_dev_rpn": round(rpn_std, 2),
        "disagreement_level": "High" if rpn_std > 2.0 else "Low",
    }

    variances = {
        "S_var": round(float(df["severity"].std()), 2) if len(df) >= 2 else 0.0,
        "O_var": round(float(df["occurrence"].std()), 2) if len(df) >= 2 else 0.0,
        "D_var": round(float(df["detection"].std()), 2) if len(df) >= 2 else 0.0,
    }

    return stats, variances


# ────────────────────────────────────────────
# 7. RPN per-model box-plot data helper
# ────────────────────────────────────────────

def prepare_box_plot_data(multi_model_results: Dict[str, pd.DataFrame],
                           metric: str = "RPN") -> pd.DataFrame:
    """
    Return a long-form DataFrame suitable for plotly box / violin plots.
    Columns: ['Model', 'Failure Mode', <metric>]
    """
    rows = []
    for model_name, df in multi_model_results.items():
        for idx, row in df.iterrows():
            fm = row.get("Failure Mode", row.get("failure_mode", f"FM-{idx}"))
            val = row.get(metric, None)
            if val is not None and pd.notna(val):
                try:
                    rows.append({"Model": model_name, "Failure Mode": fm, metric: float(val)})
                except (ValueError, TypeError):
                    pass
    return pd.DataFrame(rows)
