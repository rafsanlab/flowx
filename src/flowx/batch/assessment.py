"""Automated Batch Effect Assessment and Recommendation Engine for Flow Cytometry.

Evaluates inter-batch technical variation, detects experimental confounding,
measures marker-level Earth Mover's Distance (EMD) and median drift, and provides
an objective, MIFlowCyt-compliant recommendation on whether batch normalization
is necessary, beneficial, or counterproductive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from flowx.batch.models import CohortSampleInfo
from flowx.io.logger import get_logger


BatchRecommendationType = Literal[
    "NOT_NEEDED",
    "RECOMMENDED_CYTONORM",
    "CAUTION_ANCHOR_FREE",
    "NOT_RECOMMENDED_CONFOUNDED",
]


@dataclass
class MarkerDriftMetric:
    """Quantitative drift metrics for an individual flow cytometry marker across batches."""

    marker: str
    mean_pairwise_emd: float
    max_pairwise_emd: float
    relative_emd: float
    max_median_ratio: float
    mean_ks_stat: float
    is_drifting: bool
    drift_severity: Literal["none", "low", "moderate", "severe"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return asdict(self)


@dataclass
class BatchNecessityAssessment:
    """Comprehensive diagnostic evaluation and recommendation on batch normalization necessity."""

    recommendation: BatchRecommendationType
    confidence: float
    drift_score: float
    rationale: List[str]
    batch_count: int
    unique_batches: List[str]
    has_control_anchors: bool
    control_batches: List[str]
    sample_count: int
    evaluated_markers: List[str]
    marker_metrics: List[MarkerDriftMetric]
    confounded_with_condition: bool = False
    condition_confounding_note: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def is_normalization_recommended(self) -> bool:
        """Return True if batch normalization is safely advised."""
        return self.recommendation in ("RECOMMENDED_CYTONORM", "CAUTION_ANCHOR_FREE")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert per-marker drift metrics to a structured DataFrame."""
        if not self.marker_metrics:
            return pd.DataFrame()
        return pd.DataFrame([m.to_dict() for m in self.marker_metrics])

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire assessment to dictionary."""
        return {
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 3),
            "drift_score": round(self.drift_score, 2),
            "batch_count": self.batch_count,
            "unique_batches": self.unique_batches,
            "has_control_anchors": self.has_control_anchors,
            "control_batches": self.control_batches,
            "sample_count": self.sample_count,
            "confounded_with_condition": self.confounded_with_condition,
            "condition_confounding_note": self.condition_confounding_note,
            "rationale": self.rationale,
            "marker_metrics": [m.to_dict() for m in self.marker_metrics],
            "created_at": self.created_at,
        }

    def to_markdown(self) -> str:
        """Render a publication-ready Markdown diagnostic report."""
        badge_map = {
            "NOT_NEEDED": "🟩 **NOT NEEDED (Skip Batch Normalization)**",
            "RECOMMENDED_CYTONORM": "🟦 **RECOMMENDED (Anchor-Based CytoNorm)**",
            "CAUTION_ANCHOR_FREE": "🟨 **PROCEED WITH CAUTION (Anchor-Free Quantile)**",
            "NOT_RECOMMENDED_CONFOUNDED": "🟥 **NOT RECOMMENDED (Confounded Biological Experimental Design)**",
        }

        md = [
            "# FlowX Step 3 Batch Effect Necessity Assessment\n",
            f"**Evaluation Status:** {badge_map.get(self.recommendation, self.recommendation)}  ",
            f"**Diagnostic Drift Score:** `{self.drift_score:.1f} / 100.0`  ",
            f"**Recommendation Confidence:** `{self.confidence * 100:.1f}%`  ",
            f"**Batches Evaluated:** `{self.batch_count}` ({', '.join(self.unique_batches) if self.unique_batches else 'None'})  ",
            f"**Control Anchors Present:** `{'Yes (' + str(len(self.control_batches)) + ' batches)' if self.has_control_anchors else 'No'}`\n",
            "## 1. Rationale and Biological Context\n",
        ]

        for r in self.rationale:
            md.append(f"- {r}\n")

        if self.confounded_with_condition and self.condition_confounding_note:
            md.append(f"\n> [!CAUTION]\n> **Confounding Warning:** {self.condition_confounding_note}\n")

        md.append("\n## 2. Marker-Level Drift Breakdown\n")
        df = self.to_dataframe()
        if not df.empty:
            md.append("| Marker | Mean Pairwise EMD | Max EMD | Max Median Ratio | KS Stat | Severity | Drifting? |\n")
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for _, row in df.iterrows():
                drift_tag = "⚠️ Yes" if row["is_drifting"] else "✅ No"
                md.append(
                    f"| **{row['marker']}** | {row['mean_pairwise_emd']:.4f} | {row['max_pairwise_emd']:.4f} | "
                    f"{row['max_median_ratio']:.2f}x | {row['mean_ks_stat']:.4f} | {row['drift_severity'].title()} | {drift_tag} |\n"
                )
        else:
            md.append("*No marker distributions evaluated (single batch or insufficient events).*\n")

        md.append("\n## 3. Recommended Pipeline Action\n")
        if self.recommendation == "NOT_NEEDED":
            md.append(
                "Proceed directly to **Cohort Downsampling & Balancing** without applying distribution warping. "
                "Retaining raw scaled marker distributions preserves natural biological heterogeneity.\n"
            )
        elif self.recommendation == "RECOMMENDED_CYTONORM":
            md.append(
                "Execute **CytoNorm** (`method='cytonorm'`). Reference control anchors will be clustered via FlowSOM "
                "to fit spline warping curves, eliminating instrument drift while safeguarding patient phenotypes.\n"
            )
        elif self.recommendation == "CAUTION_ANCHOR_FREE":
            md.append(
                "If non-biological batch drift must be aligned, use **Anchor-Free Quantile Alignment** (`method='quantile'`). "
                "Inspect post-normalization density plots carefully to confirm that true biological population differences "
                "have not been compressed.\n"
            )
        elif self.recommendation == "NOT_RECOMMENDED_CONFOUNDED":
            md.append(
                "**Do NOT normalize across batches.** Experimental batches are confounded with biological condition. "
                "Normalizing will remove true disease/treatment signals. Analyze cohorts using differential statistics (Phase 5) "
                "with batch included as a covariate if sample replication permits.\n"
            )

        return "".join(md)


def assess_batch_effect_need(
    cohort_dfs: Dict[str, pd.DataFrame],
    sample_info: Union[List[CohortSampleInfo], Dict[str, CohortSampleInfo]],
    markers: Optional[List[str]] = None,
    condition_key: Optional[str] = "condition",
    emd_drift_threshold: float = 0.12,
    median_ratio_threshold: float = 1.30,
) -> BatchNecessityAssessment:
    """Objectively assess whether batch normalization is necessary, beneficial, or counterproductive.

    Evaluates:
    1. Experimental design (batch count, control anchors, condition confounding).
    2. Inter-batch distribution drift (Earth Mover's Distance, median fold changes, KS stats).
    3. Multi-marker aggregate drift score.

    Args:
        cohort_dfs: Dictionary mapping sample_id to event DataFrame.
        sample_info: List or dictionary of CohortSampleInfo objects.
        markers: List of markers to evaluate (auto-detected if None).
        condition_key: Extra metadata key in CohortSampleInfo to inspect for batch confounding.
        emd_drift_threshold: Earth Mover's Distance threshold above which a marker is flagged as drifting.
        median_ratio_threshold: Max/min median ratio threshold across batches for flagging drift.

    Returns:
        BatchNecessityAssessment containing recommendation, drift score, and diagnostics.
    """
    logger = get_logger()

    # Convert sample_info to dictionary
    if isinstance(sample_info, list):
        s_map = {s.sample_id: s for s in sample_info}
    else:
        s_map = sample_info

    # Filter to samples present in cohort_dfs
    active_samples = [s_map[s_id] for s_id in cohort_dfs.keys() if s_id in s_map]
    if not active_samples:
        return BatchNecessityAssessment(
            recommendation="NOT_NEEDED",
            confidence=1.0,
            drift_score=0.0,
            rationale=["No matching sample metadata found for provided event DataFrames."],
            batch_count=0,
            unique_batches=[],
            has_control_anchors=False,
            control_batches=[],
            sample_count=0,
            evaluated_markers=[],
            marker_metrics=[],
        )

    # 1. Experimental Design Breakdown
    unique_batches = sorted({s.batch_id for s in active_samples if s.batch_id})
    batch_count = len(unique_batches)

    # Check for technical control anchors
    control_batches = sorted({s.batch_id for s in active_samples if s.is_control and s.batch_id})
    has_control_anchors = len(control_batches) >= 2

    # Check for confounding between Batch and Biological Condition
    confounded = False
    confounding_note = None
    conditions_by_batch: Dict[str, set] = {}
    batches_by_condition: Dict[str, set] = {}
    for s in active_samples:
        cond = None
        if s.extra_metadata:
            cond = s.extra_metadata.get(condition_key)
            if cond is None:
                for fallback_key in ("condition", "Condition", "group", "Group", "treatment", "Treatment"):
                    if fallback_key in s.extra_metadata:
                        cond = s.extra_metadata[fallback_key]
                        condition_key = fallback_key
                        break
        if cond is not None and str(cond).strip() != "":
            c_str = str(cond).strip()
            conditions_by_batch.setdefault(s.batch_id, set()).add(c_str)
            batches_by_condition.setdefault(c_str, set()).add(s.batch_id)

    if len(conditions_by_batch) >= 2 and len(batches_by_condition) >= 2:
        # Confounded if:
        # (a) every batch contains only a single condition (pure-condition batches), OR
        # (b) any condition's batch set has zero overlap with all other conditions' batches.
        pure_condition_batches = all(len(c_set) == 1 for c_set in conditions_by_batch.values())
        isolated_condition = any(
            all(len(b_set.intersection(other_b)) == 0 for k2, other_b in batches_by_condition.items() if k2 != k1)
            for k1, b_set in batches_by_condition.items()
        )
        if pure_condition_batches or isolated_condition:
            confounded = True
            confounding_note = (
                f"Batches are completely confounded and segregated by biological condition '{condition_key}'. "
                f"Batch assignments: { {b: sorted(c) for b, c in conditions_by_batch.items()} }. "
                "Normalizing across batches will eliminate biological condition differences."
            )

    # Scenario A: Single batch
    if batch_count <= 1:
        return BatchNecessityAssessment(
            recommendation="NOT_NEEDED",
            confidence=1.0,
            drift_score=0.0,
            rationale=[
                f"Single experimental batch detected ('{unique_batches[0] if unique_batches else 'Batch_1'}').",
                "Technical inter-batch drift cannot exist within a single run.",
                "Applying batch normalization to single-batch cohorts is counterproductive and risks compressing patient heterogeneity.",
                "Recommendation: Apply deterministic downsampling and event balancing only.",
            ],
            batch_count=batch_count,
            unique_batches=unique_batches,
            has_control_anchors=has_control_anchors,
            control_batches=control_batches,
            sample_count=len(active_samples),
            evaluated_markers=[],
            marker_metrics=[],
            confounded_with_condition=confounded,
            condition_confounding_note=confounding_note,
        )

    # Scenario B: Confounded design
    if confounded:
        return BatchNecessityAssessment(
            recommendation="NOT_RECOMMENDED_CONFOUNDED",
            confidence=0.95,
            drift_score=0.0,
            rationale=[
                f"Experimental batches are completely confounded with biological condition '{condition_key}'.",
                "Each batch represents an isolated experimental group (e.g. Healthy in Batch A vs Disease in Batch B).",
                "Normalizing will force diseased and healthy profiles to match, destroying downstream biological findings.",
                "Recommendation: Do NOT normalize. Model batch as a covariate in statistical tests if replicates allow.",
            ],
            batch_count=batch_count,
            unique_batches=unique_batches,
            has_control_anchors=has_control_anchors,
            control_batches=control_batches,
            sample_count=len(active_samples),
            evaluated_markers=[],
            marker_metrics=[],
            confounded_with_condition=True,
            condition_confounding_note=confounding_note,
        )

    # 2. Inter-Batch Distribution Drift Assessment
    # Determine which samples to evaluate: if control anchors exist in >=2 batches, evaluate on anchors!
    eval_samples = [s for s in active_samples if s.is_control] if has_control_anchors else active_samples

    # Auto-detect markers if not provided
    if markers is None:
        first_df = cohort_dfs[active_samples[0].sample_id]
        markers = [
            c for c in first_df.columns
            if not any(
                c.lower().startswith(ex)
                for ex in ["time", "event", "orig", "sample", "batch", "fsc", "ssc", "is_"]
            )
        ]

    # Aggregate events by batch for evaluation samples
    batch_dfs: Dict[str, List[pd.DataFrame]] = {b: [] for b in unique_batches}
    for s in eval_samples:
        if s.sample_id in cohort_dfs:
            batch_dfs[s.batch_id].append(cohort_dfs[s.sample_id])

    active_batch_data: Dict[str, pd.DataFrame] = {}
    for b, dfs in batch_dfs.items():
        if dfs:
            active_batch_data[b] = pd.concat(dfs, ignore_index=True)

    eval_batches = sorted(active_batch_data.keys())
    if len(eval_batches) < 2:
        return BatchNecessityAssessment(
            recommendation="NOT_NEEDED",
            confidence=0.90,
            drift_score=0.0,
            rationale=[
                "Insufficient multi-batch evaluation data available to calculate pairwise drift.",
                "Recommendation: Downsampling only.",
            ],
            batch_count=batch_count,
            unique_batches=unique_batches,
            has_control_anchors=has_control_anchors,
            control_batches=control_batches,
            sample_count=len(active_samples),
            evaluated_markers=[],
            marker_metrics=[],
        )

    marker_metrics: List[MarkerDriftMetric] = []
    total_emd_sum = 0.0
    pair_count = 0

    for m in markers:
        pairwise_emds: List[float] = []
        medians: List[float] = []
        ks_stats: List[float] = []
        pooled_vals: List[np.ndarray] = []

        for b in eval_batches:
            df_b = active_batch_data[b]
            if m in df_b.columns and len(df_b) > 0:
                arr = df_b[m].to_numpy()
                medians.append(float(np.median(arr)))
                pooled_vals.append(arr)

        if not pooled_vals:
            continue

        all_vals = np.concatenate(pooled_vals)
        pooled_sd = float(np.std(all_vals)) if len(all_vals) > 1 else 1.0
        if pooled_sd < 1e-4:
            pooled_sd = 1.0

        for i in range(len(eval_batches)):
            for j in range(i + 1, len(eval_batches)):
                b1, b2 = eval_batches[i], eval_batches[j]
                df1, df2 = active_batch_data[b1], active_batch_data[b2]
                if m in df1.columns and m in df2.columns and len(df1) > 10 and len(df2) > 10:
                    v1 = df1[m].to_numpy()
                    v2 = df2[m].to_numpy()
                    emd = float(wasserstein_distance(v1, v2))
                    ks = float(ks_2samp(v1, v2, method="asymp").statistic)
                    pairwise_emds.append(emd)
                    ks_stats.append(ks)
                    total_emd_sum += emd
                    pair_count += 1

        if not pairwise_emds:
            continue

        mean_emd = float(np.mean(pairwise_emds))
        max_emd = float(np.max(pairwise_emds))
        mean_ks = float(np.mean(ks_stats))
        rel_emd = float(mean_emd / pooled_sd)

        # Calculate max median fold ratio across batches
        if medians and min(medians) > 1e-4:
            median_ratio = float(max(medians) / min(medians))
        elif medians:
            median_ratio = float(1.0 + (max(medians) - min(medians)))
        else:
            median_ratio = 1.0

        is_drifting = (rel_emd >= 0.20) or (median_ratio >= median_ratio_threshold)

        if rel_emd < 0.20 and median_ratio < median_ratio_threshold:
            severity = "none"
        elif rel_emd < 0.35 and median_ratio < (median_ratio_threshold * 1.25):
            severity = "low"
        elif rel_emd < 0.60 or median_ratio < (median_ratio_threshold * 1.75):
            severity = "moderate"
        else:
            severity = "severe"

        marker_metrics.append(
            MarkerDriftMetric(
                marker=m,
                mean_pairwise_emd=round(mean_emd, 4),
                max_pairwise_emd=round(max_emd, 4),
                relative_emd=round(rel_emd, 4),
                max_median_ratio=round(median_ratio, 3),
                mean_ks_stat=round(mean_ks, 4),
                is_drifting=is_drifting,
                drift_severity=severity,
            )
        )

    # 3. Calculate Overall Drift Score (0 to 100)
    if marker_metrics:
        drifting_frac = sum(1 for m in marker_metrics if m.is_drifting) / len(marker_metrics)
        mean_cohort_emd = total_emd_sum / max(pair_count, 1)
        mean_cohort_ks = float(np.mean([m.mean_ks_stat for m in marker_metrics]))
        mean_rel_emd = float(np.mean([m.relative_emd for m in marker_metrics]))

        # Drift score formula: combines fraction of drifting markers, mean relative EMD, and KS divergence
        drift_score = min(
            100.0,
            (drifting_frac * 50.0)
            + (min(mean_rel_emd / 0.50, 1.0) * 30.0)
            + (min(mean_cohort_ks / 0.25, 1.0) * 20.0),
        )
    else:
        drift_score = 0.0
        mean_cohort_emd = 0.0
        drifting_frac = 0.0

    # 4. Synthesize Final Recommendation
    rationale: List[str] = []
    rationale.append(f"Evaluated {len(active_samples)} samples across {batch_count} batches ({', '.join(eval_batches)}).")

    if has_control_anchors:
        rationale.append(
            f"Technical control anchors detected across {len(control_batches)} batches. "
            "Drift evaluation was isolated strictly to identical control aliquots to protect biological integrity."
        )
    else:
        rationale.append(
            "No technical control anchors found. Drift was evaluated across pooled sample distributions."
        )

    rationale.append(
        f"Diagnostic drift score is {drift_score:.1f}/100.0 with {int(drifting_frac * 100)}% of markers "
        f"exceeding drift thresholds (Mean EMD: {mean_cohort_emd:.4f})."
    )

    if drift_score < 18.0:
        recommendation: BatchRecommendationType = "NOT_NEEDED"
        confidence = 0.92
        rationale.append(
            "Baseline technical variation is minimal (<5-10% shift). Multi-batch distributions are already tightly aligned."
        )
        rationale.append(
            "Normalization is NOT needed. Applying spline or quantile warping would provide negligible gain and risks over-smoothing."
        )
    elif has_control_anchors:
        recommendation = "RECOMMENDED_CYTONORM"
        confidence = 0.95
        rationale.append(
            "Significant inter-batch technical drift detected, and reference control anchors are available."
        )
        rationale.append(
            "CytoNorm spline warping on control anchors will cleanly eliminate instrument drift without warping patient-specific phenotypes."
        )
    else:
        recommendation = "CAUTION_ANCHOR_FREE"
        confidence = 0.78
        rationale.append(
            "Noticeable inter-batch drift detected, but NO technical control anchors were found."
        )
        rationale.append(
            "Anchor-Free Quantile Alignment can align distributions, but proceed with caution: if biological proportions differ "
            "between batches, quantile matching may confound technical correction with biological signal."
        )

    assessment = BatchNecessityAssessment(
        recommendation=recommendation,
        confidence=confidence,
        drift_score=drift_score,
        rationale=rationale,
        batch_count=batch_count,
        unique_batches=unique_batches,
        has_control_anchors=has_control_anchors,
        control_batches=control_batches,
        sample_count=len(active_samples),
        evaluated_markers=[m.marker for m in marker_metrics],
        marker_metrics=marker_metrics,
        confounded_with_condition=False,
    )

    logger.info(
        f"Batch Necessity Assessment completed: {assessment.recommendation} "
        f"(Drift Score: {assessment.drift_score:.1f}/100, Confidence: {assessment.confidence*100:.1f}%)"
    )

    return assessment
