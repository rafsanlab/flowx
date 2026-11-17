"""Deterministic Downsampling and Cohort Balancing for Flow Cytometry.

Implements equal-quota sampling, minimum-cohort balancing, stratified subpopulation
sampling, and index preservation conforming to the MIFlowCyt data provenance standard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flowx.batch.models import DownsampleConfig
from flowx.io.logger import get_logger
from flowx.io.parser import FCSData


def downsample_dataframe(
    df: pd.DataFrame,
    config: Optional[DownsampleConfig] = None,
    target_count: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Downsample an event DataFrame deterministically based on configuration.

    Args:
        df: Input event DataFrame.
        config: DownsampleConfig settings. Defaults to target_count=50000 if None.
        target_count: Optional explicit event count override.
        random_seed: Optional PRNG seed override.

    Returns:
        Downsampled DataFrame with preserved indices and identical columns.
    """
    if config is None:
        config = DownsampleConfig()

    logger = get_logger()
    n_events = len(df)

    if n_events == 0:
        return df.copy()

    mode = config.mode
    seed = random_seed if random_seed is not None else config.random_seed
    rng = np.random.default_rng(seed)

    # 1. Bypass if mode is 'none' or downsampling disabled
    if mode == "none" or (not config.enabled and target_count is None):
        res = df.copy()
        if config.preserve_indices and "original_event_index" not in res.columns:
            res.insert(0, "original_event_index", res.index.to_numpy())
        return res

    # Record original event indices before sampling
    original_indices = df["original_event_index"].to_numpy() if "original_event_index" in df.columns else df.index.to_numpy()

    # 2. Resolve target size for single sample
    effective_target: int
    if mode == "proportional":
        effective_target = max(1, int(round(n_events * config.fraction)))
    elif mode == "stratified":
        return _downsample_stratified(df, config, rng)
    else:  # 'target_count' or 'min_sample'
        effective_target = target_count if target_count is not None else (config.target_events or n_events)

    # 3. Handle undersized samples
    if n_events <= effective_target:
        if config.min_event_policy == "error":
            raise ValueError(
                f"Sample contains only {n_events:,} events, fewer than target quota of {effective_target:,}."
            )
        elif config.min_event_policy == "exclude":
            logger.info(
                f"Sample with {n_events:,} events excluded (below target quota of {effective_target:,})."
            )
            empty_df = df.iloc[0:0].copy()
            if config.preserve_indices and "original_event_index" not in empty_df.columns:
                empty_df.insert(0, "original_event_index", np.array([], dtype=int))
            return empty_df
        else:  # 'cap'
            logger.debug(
                f"Sample has {n_events:,} events (<= target {effective_target:,}). Retaining all available events."
            )
            sampled_df = df.copy()
            if config.preserve_indices and "original_event_index" not in sampled_df.columns:
                sampled_df.insert(0, "original_event_index", original_indices)
            return sampled_df

    # 4. Deterministic uniform sampling without replacement
    chosen_locs = rng.choice(n_events, size=effective_target, replace=False)
    chosen_locs.sort()  # Maintain chronological order

    sampled_df = df.iloc[chosen_locs].copy().reset_index(drop=True)
    if config.preserve_indices:
        sampled_df.insert(0, "original_event_index", original_indices[chosen_locs])

    logger.debug(
        f"Downsampled from {n_events:,} to {len(sampled_df):,} events ({len(sampled_df) / n_events * 100:.1f}%)."
    )
    return sampled_df


def _downsample_stratified(
    df: pd.DataFrame,
    config: DownsampleConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Downsample events proportionally or with fixed quota per stratified population."""
    strat_col = config.stratified_column
    if not strat_col or strat_col not in df.columns:
        raise ValueError(
            f"Stratified downsampling requested but column '{strat_col}' not found in DataFrame columns."
        )

    groups = df.groupby(strat_col, observed=True)
    quota = config.stratified_quota
    sampled_indices: List[int] = []

    for name, group in groups:
        grp_len = len(group)
        if grp_len == 0:
            continue
        target = quota if quota is not None else max(1, config.target_events // max(1, len(groups)))
        if grp_len <= target:
            sampled_indices.extend(group.index.tolist())
        else:
            locs = rng.choice(grp_len, size=target, replace=False)
            sampled_indices.extend(group.index[locs].tolist())

    sampled_indices.sort()
    original_indices = df["original_event_index"].to_numpy() if "original_event_index" in df.columns else df.index.to_numpy()

    res = df.loc[sampled_indices].copy().reset_index(drop=True)
    if config.preserve_indices and "original_event_index" not in res.columns:
        res.insert(0, "original_event_index", original_indices[sampled_indices])

    return res


def downsample_cohort(
    cohort_dfs: Dict[str, pd.DataFrame],
    config: Optional[DownsampleConfig] = None,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Deterministically downsample an entire cohort of flow cytometry samples.

    Args:
        cohort_dfs: Dictionary mapping sample_id to event DataFrame.
        config: DownsampleConfig settings.

    Returns:
        Tuple of (downsampled_cohort_dfs, summary_statistics_df).
    """
    if config is None:
        config = DownsampleConfig()

    logger = get_logger()
    downsampled_dfs: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []

    if not cohort_dfs:
        return {}, pd.DataFrame()

    # Determine effective target size across cohort
    effective_target = config.target_events
    active_counts = [len(df) for df in cohort_dfs.values() if len(df) > 0]
    min_floor = max(0, int(getattr(config, "min_events_threshold", 2500)))
    valid_counts = [c for c in active_counts if c >= min_floor]

    if config.mode == "min_sample" or config.target_events is None:
        if active_counts:
            if valid_counts:
                min_c = min(valid_counts)
                if len(valid_counts) < len(active_counts):
                    n_low = len(active_counts) - len(valid_counts)
                    logger.warning(
                        f"Cohort Auto-Balancing: {n_low} low-event sample(s) (< {min_floor:,} events) "
                        f"bypassed during min_sample floor calculation to prevent cohort bottlenecking."
                    )
            else:
                min_c = min(active_counts)

            if config.target_events is not None and config.target_events > 0:
                effective_target = min(config.target_events, min_c)
            else:
                effective_target = min_c
            logger.info(
                f"Cohort Auto-Balancing (bottleneck min_sample): Target set to minimum cohort size ({effective_target:,} events)."
            )

    failed_tube_cutoff = min(500, min_floor) if min_floor > 0 else 0

    for idx, (sample_id, df) in enumerate(cohort_dfs.items()):
        raw_count = len(df)
        # Offset seed per sample deterministically to prevent identical sampling patterns
        sample_seed = config.random_seed + idx

        # Exclude failed acquisition tubes (< 500 viable singlets when other samples meet min_floor)
        if valid_counts and (
            raw_count < failed_tube_cutoff
            or (raw_count < min_floor and config.min_event_policy == "exclude")
        ):
            logger.warning(
                f"Sample '{sample_id}' has only {raw_count:,} clean events "
                f"(below minimum threshold {min_floor:,}); excluding failed/low-event tube from cohort."
            )
            summary_rows.append({
                "sample_id": sample_id,
                "raw_events": raw_count,
                "downsampled_events": 0,
                "retention_pct": 0.0,
                "status": "excluded_low_events",
            })
            continue

        sub_df = downsample_dataframe(
            df=df,
            config=config,
            target_count=effective_target,
            random_seed=sample_seed,
        )

        sampled_count = len(sub_df)
        retention = (sampled_count / raw_count * 100.0) if raw_count > 0 else 0.0

        if sampled_count > 0 or config.min_event_policy != "exclude":
            downsampled_dfs[sample_id] = sub_df

        status_label = "retained" if sampled_count > 0 else "excluded"
        if sampled_count > 0 and effective_target is not None and raw_count < effective_target:
            status_label = "retained_low_count"

        summary_rows.append({
            "sample_id": sample_id,
            "raw_events": raw_count,
            "downsampled_events": sampled_count,
            "retention_pct": round(retention, 2),
            "status": status_label,
        })

    summary_df = pd.DataFrame(summary_rows)
    total_pre = sum(r["raw_events"] for r in summary_rows)
    total_post = sum(r["downsampled_events"] for r in summary_rows)
    logger.info(
        f"Cohort Downsampling complete: {len(downsampled_dfs)}/{len(cohort_dfs)} samples prepared. "
        f"Events: {total_post:,} / {total_pre:,} ({total_post / total_pre * 100:.1f}% retained)."
    )

    return downsampled_dfs, summary_df


def export_downsampled_fcs(
    df: pd.DataFrame,
    fcs_source: FCSData,
    output_path: Union[str, Path],
    uncompensated: bool = False,
) -> Path:
    """Export downsampled events to a standardized valid FCS file using FlowIO.

    Args:
        df: Downsampled event DataFrame.
        fcs_source: Original FCSData instance providing channel metadata and scale ranges.
        output_path: Destination path for exported .fcs file.
        uncompensated: Whether to export uncompensated raw values (if available).

    Returns:
        Resolved Path of exported FCS file.
    """
    import flowio

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Exclude non-detector columns (like original_event_index, batch_id, sample_id)
    detector_cols = [c for c in fcs_source.pnn_labels if c in df.columns]

    if not detector_cols:
        raise ValueError("No matching FCS detector channels found in DataFrame for FCS export.")

    data_to_export = df[detector_cols].to_numpy().flatten().tolist()

    # Reconstruct channel metadata dict for FlowIO
    channel_names = [fcs_source.get_channel_label(c) or c for c in detector_cols]

    with open(dest, "wb+") as fh:
        flowio.create_fcs(
            fh,
            event_data=data_to_export,
            channel_names=channel_names,
            opt_channel_names=detector_cols,
        )

    get_logger().info(
        f"Exported {len(df):,} downsampled events across {len(detector_cols)} channels to '{dest.name}'."
    )
    return dest
