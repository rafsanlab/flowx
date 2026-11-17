"""Cohort Batch Preparation Orchestrator.

Coordinates inter-batch normalization (native CytoNorm, quantile alignment),
cohort event downsampling and balancing, diagnostic suite generation, and
MIFlowCyt audit persistence across multi-sample studies.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from flowx.batch.assessment import (
    BatchNecessityAssessment,
    assess_batch_effect_need,
)
from flowx.batch.downsampling import (
    downsample_cohort,
    export_downsampled_fcs,
)
from flowx.batch.models import (
    BatchCorrectionConfig,
    BatchPreparationConfig,
    CohortBatchResult,
    CohortSampleInfo,
    DownsampleConfig,
)
from flowx.batch.normalization import (
    CytoNormModel,
    QuantileNormalizer,
    calculate_batch_invariance_metrics,
    fit_anchor_free_quantile_normalizer,
    fit_cytonorm,
)
from flowx.io.logger import format_duration, get_logger
from flowx.io.parser import FCSData, parse_fcs
from flowx.visualization.batch_plots import (
    plot_batch_assessment_summary,
    plot_batch_density_comparison,
    plot_batch_metrics_summary,
    plot_cohort_downsampling_balance,
)


def load_cohort_manifest(manifest_path: Union[str, Path]) -> List[CohortSampleInfo]:
    """Parse sample manifest file (CSV or JSON) into structured CohortSampleInfo instances.

    Expected CSV columns:
        - 'filepath' or 'path': Absolute or relative path to .fcs file.
        - 'sample_id' (optional, stem inferred if omitted).
        - 'batch_id' or 'batch' (optional, 'Batch_1' if omitted).
        - 'is_control' or 'control' or 'reference' (optional, False if omitted).

    Args:
        manifest_path: Path to manifest CSV or JSON.

    Returns:
        List of CohortSampleInfo objects.
    """
    src = Path(manifest_path)
    if not src.is_file():
        raise FileNotFoundError(f"Cohort manifest not found: '{src}'")

    samples: List[CohortSampleInfo] = []

    manifest_dir = src.resolve().parent

    if src.suffix.lower() == ".json":
        data = json.loads(src.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("samples", [])
        for rec in records:
            fp = rec.get("filepath") or rec.get("path")
            if fp and not Path(fp).is_absolute():
                if (manifest_dir / fp).exists():
                    fp = str((manifest_dir / fp).resolve())
                elif Path(fp).exists():
                    fp = str(Path(fp).resolve())
            s_id = rec.get("sample_id") or (Path(fp).stem if fp else "Unknown")
            b_id = str(rec.get("batch_id") or rec.get("batch") or "Batch_1")
            is_ctrl = bool(rec.get("is_control") or rec.get("control") or rec.get("reference") or False)
            raw_cnt = int(rec.get("raw_event_count") or rec.get("raw_events") or rec.get("total_events") or 0)
            clean_cnt = int(rec.get("clean_event_count") or rec.get("clean_events") or 0)
            samples.append(
                CohortSampleInfo(
                    sample_id=s_id,
                    batch_id=b_id,
                    is_control=is_ctrl,
                    filepath=fp,
                    raw_event_count=raw_cnt,
                    clean_event_count=clean_cnt,
                    extra_metadata=rec,
                )
            )
        return samples

    df = pd.read_csv(src)
    # Normalize column names
    col_map = {c: c.strip().lower() for c in df.columns}
    inv_map = {v: k for k, v in col_map.items()}

    fp_col = inv_map.get("filepath") or inv_map.get("path") or inv_map.get("file")
    id_col = inv_map.get("sample_id") or inv_map.get("sample") or inv_map.get("name")
    batch_col = inv_map.get("batch_id") or inv_map.get("batch") or inv_map.get("run") or inv_map.get("day")
    ctrl_col = inv_map.get("is_control") or inv_map.get("control") or inv_map.get("ref") or inv_map.get("reference")
    raw_col = inv_map.get("raw_event_count") or inv_map.get("raw_events") or inv_map.get("total_events")
    clean_col = inv_map.get("clean_event_count") or inv_map.get("clean_events")

    for _, row in df.iterrows():
        fp = str(row[fp_col]) if fp_col and pd.notna(row[fp_col]) else None
        s_id = str(row[id_col]) if id_col and pd.notna(row[id_col]) else (Path(fp).stem if fp else f"Sample_{len(samples)+1}")
        b_id = str(row[batch_col]) if batch_col and pd.notna(row[batch_col]) else "Batch_1"
        is_ctrl = bool(row[ctrl_col]) if ctrl_col and pd.notna(row[ctrl_col]) else False

        raw_cnt = int(row[raw_col]) if raw_col and pd.notna(row[raw_col]) else 0
        clean_cnt = int(row[clean_col]) if clean_col and pd.notna(row[clean_col]) else 0

        if fp and not Path(fp).is_absolute():
            if Path(fp).exists():
                fp = str(Path(fp).resolve())
            elif (manifest_dir / fp).exists():
                fp = str((manifest_dir / fp).resolve())

        extra = {k: v for k, v in row.to_dict().items() if pd.notna(v)}
        samples.append(
            CohortSampleInfo(
                sample_id=s_id,
                batch_id=b_id,
                is_control=is_ctrl,
                filepath=fp,
                raw_event_count=raw_cnt,
                clean_event_count=clean_cnt,
                extra_metadata=extra,
            )
        )

    return samples


def discover_cohort_samples(
    targets: Union[str, Path, List[Union[str, Path]]],
    pattern: str = "*.fcs",
) -> List[CohortSampleInfo]:
    """Auto-discover FCS files from directory or paths and infer batch and control tags.

    Args:
        targets: Directory path, single FCS file, or list of filepaths.
        pattern: Glob pattern when searching directory. Default '*.fcs'.

    Returns:
        List of CohortSampleInfo instances with inferred metadata.
    """
    files: List[Path] = []
    if isinstance(targets, (str, Path)):
        p = Path(targets)
        if p.is_dir():
            files = sorted(p.rglob(pattern))
        elif p.is_file():
            files = [p]
    elif isinstance(targets, list):
        for item in targets:
            ip = Path(item)
            if ip.is_dir():
                files.extend(sorted(ip.rglob(pattern)))
            elif ip.is_file():
                files.append(ip)

    samples: List[CohortSampleInfo] = []
    logger = get_logger()

    for f in files:
        stem = f.stem
        # Default without guessing: all samples belong to Batch_1 and are not controls unless declared in a manifest
        samples.append(CohortSampleInfo(
            sample_id=stem,
            batch_id="Batch_1",
            is_control=False,
            filepath=str(f.resolve()),
        ))

    if files:
        logger.debug(
            "Discovered %d cohort FCS files without an explicit manifest. Defaulted batch_id='Batch_1', is_control=False. "
            "To assign batches or technical controls, provide a manifest CSV.",
            len(files),
        )

    return samples


def _auto_resolve_upstream_cleanup_counts(
    resolved_samples: List[CohortSampleInfo],
    out_base: Path,
    logger: Any,
) -> None:
    """Search for upstream cleanup gating audit summaries to recover true raw acquired events."""
    needs_lookup = any(
        s.raw_event_count == 0 or s.raw_event_count == s.clean_event_count
        for s in resolved_samples
    )
    if not needs_lookup:
        return

    candidate_dirs: List[Path] = [
        out_base.parent / "03_cleanup_gating",
        out_base.parent / "cleanup",
        out_base / "cleanup",
    ]
    for s in resolved_samples:
        if s.filepath:
            fp = Path(s.filepath)
            candidate_dirs.append(fp.parent)
            candidate_dirs.append(fp.parent.parent)

    summary_csv: Optional[Path] = None
    for d in candidate_dirs:
        if not d.exists():
            continue
        for fname in ["cohort_cleanup_summary.csv", "cleanup_summary.csv"]:
            p = d / fname
            if p.exists() and p.is_file():
                summary_csv = p
                break
        if summary_csv:
            break

    if not summary_csv:
        return

    try:
        df = pd.read_csv(summary_csv)
        name_col = next((c for c in ["sample_name", "sample_id", "sample"] if c in df.columns), None)
        tot_col = next((c for c in ["total_events", "raw_events", "raw_event_count"] if c in df.columns), None)
        cln_col = next((c for c in ["clean_events", "clean_event_count"] if c in df.columns), None)

        if not (name_col and tot_col and cln_col):
            return

        lookup: Dict[str, Tuple[int, int]] = {}
        for _, row in df.iterrows():
            s_name = str(row[name_col]).strip()
            lookup[s_name] = (int(row[tot_col]), int(row[cln_col]))
            lookup[Path(s_name).stem] = (int(row[tot_col]), int(row[cln_col]))

        matched_count = 0
        for s_info in resolved_samples:
            candidates = [
                s_info.sample_id,
                Path(s_info.filepath).stem if s_info.filepath else "",
                s_info.sample_id.replace("_prepared", ""),
                s_info.sample_id.replace("_exp", ""),
            ]
            for cand in candidates:
                if cand and cand in lookup:
                    raw_c, cln_c = lookup[cand]
                    s_info.raw_event_count = raw_c
                    if cln_c > 0:
                        s_info.clean_event_count = cln_c
                    matched_count += 1
                    break

        if matched_count > 0:
            logger.info(
                f"Auto-resolved upstream raw event counts for {matched_count}/{len(resolved_samples)} samples from '{summary_csv.name}'."
            )
    except Exception as exc:
        logger.debug(f"Could not auto-resolve cleanup summary counts: {exc}")


def run_batch_preparation(
    cohort_targets: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    cohort_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    sample_info: Optional[List[CohortSampleInfo]] = None,
    config: Optional[BatchPreparationConfig] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> CohortBatchResult:
    """Execute Step 3 Batch Preparation: Normalization, Cohort Downsampling, and Auditing.

    Supports both standalone file-based execution and in-memory chained handover from Step 2.

    Args:
        cohort_targets: Directory or list of .fcs files to process.
        manifest_path: Optional path to sample manifest CSV.
        cohort_dfs: Optional in-memory dictionary of event DataFrames (for pipeline chaining).
        sample_info: Optional list of CohortSampleInfo instances.
        config: BatchPreparationConfig settings.
        output_dir: Destination directory override for outputs and reports.

    Returns:
        CohortBatchResult holding prepared datasets, models, metrics, and plots.
    """
    t_start = time.perf_counter()
    logger = get_logger()

    if config is None:
        config = BatchPreparationConfig()

    out_base = Path(output_dir or config.output_dir)
    plots_dir = out_base / "plots"
    tables_dir = out_base / "tables"
    fcs_dir = out_base / "processed_fcs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    generated_plots: Dict[str, str] = {}
    generated_tables: Dict[str, str] = {}
    exported_fcs: Dict[str, str] = {}

    # 1. Resolve Sample Information
    resolved_samples: List[CohortSampleInfo] = []
    if sample_info is not None:
        resolved_samples = sample_info
    elif manifest_path is not None:
        resolved_samples = load_cohort_manifest(manifest_path)
    elif cohort_targets is not None:
        resolved_samples = discover_cohort_samples(cohort_targets)
    elif cohort_dfs is not None:
        # Construct default sample info for in-memory DataFrames: uniform single batch, no guessing
        for s_id in cohort_dfs.keys():
            resolved_samples.append(CohortSampleInfo(sample_id=s_id, batch_id="Batch_1", is_control=False))
    else:
        raise ValueError("Must provide either cohort_targets, manifest_path, or cohort_dfs.")

    if not resolved_samples:
        raise ValueError("No valid samples identified for cohort batch preparation.")

    sample_info_map = {s.sample_id: s for s in resolved_samples}

    # 2. Load or Resolve Event DataFrames
    working_dfs: Dict[str, pd.DataFrame] = {}
    parsed_fcs_sources: Dict[str, FCSData] = {}

    if cohort_dfs is not None:
        working_dfs = {k: v.copy() for k, v in cohort_dfs.items()}
    else:
        logger.info(f"Loading {len(resolved_samples)} cohort FCS files...")
        for s_info in resolved_samples:
            if not s_info.filepath:
                continue
            fcs = parse_fcs(s_info.filepath)
            parsed_fcs_sources[s_info.sample_id] = fcs
            working_dfs[s_info.sample_id] = fcs.events_df.copy()
            if s_info.clean_event_count == 0:
                s_info.clean_event_count = fcs.event_count
            if s_info.raw_event_count == 0:
                s_info.raw_event_count = s_info.clean_event_count

    # Update event counts for in-memory inputs
    for s_info in resolved_samples:
        if s_info.sample_id in working_dfs:
            cnt = len(working_dfs[s_info.sample_id])
            if s_info.clean_event_count == 0:
                s_info.clean_event_count = cnt
            if s_info.raw_event_count == 0:
                s_info.raw_event_count = cnt

    # Recover true raw acquired event counts from upstream cleanup audit if available
    _auto_resolve_upstream_cleanup_counts(resolved_samples, out_base, logger)

    raw_cohort_backup = {k: v.copy() for k, v in working_dfs.items()}
    norm_cfg = config.batch_normalization
    norm_model: Optional[Any] = None
    metrics_df: Optional[pd.DataFrame] = None
    batch_assessment: Optional[BatchNecessityAssessment] = None
    skip_norm_by_autocheck: bool = False

    # -----------------------------------------------------------------
    # Step 3.0: Automated Batch Effect & Necessity Assessment
    # -----------------------------------------------------------------
    if config.assess_batch_effect:
        logger.info("Step 3.0: Assessing inter-batch variation and normalization necessity...")
        try:
            batch_assessment = assess_batch_effect_need(
                cohort_dfs=working_dfs,
                sample_info=sample_info_map,
                markers=norm_cfg.markers,
            )
            logger.info(
                f"Batch Assessment: {batch_assessment.recommendation} "
                f"(Drift Score: {batch_assessment.drift_score:.1f}/100, Confidence: {batch_assessment.confidence * 100:.1f}%)"
            )

            if norm_cfg.enabled and norm_cfg.method != "none":
                if batch_assessment.recommendation == "NOT_RECOMMENDED_CONFOUNDED":
                    if config.overwrite_autocheck:
                        logger.warning(
                            "Batch assessment flagged CONFOUNDED experimental design (NOT_RECOMMENDED_CONFOUNDED), "
                            "but overwrite_autocheck=True — forcing batch normalization as requested."
                        )
                    else:
                        skip_norm_by_autocheck = True
                        logger.warning(
                            "Batch normalization was requested, but auto-check flagged NOT_RECOMMENDED_CONFOUNDED "
                            "(normalizing may erase true biological differences). Skipping normalization cleanly; "
                            "set overwrite_autocheck=True (or --overwrite-autocheck) to force normalization."
                        )
                elif batch_assessment.recommendation == "NOT_NEEDED":
                    if config.overwrite_autocheck:
                        logger.warning(
                            "Batch assessment returned NOT_NEEDED (negligible baseline drift or single batch), "
                            "but overwrite_autocheck=True — forcing batch normalization as requested."
                        )
                    else:
                        skip_norm_by_autocheck = True
                        logger.warning(
                            "Batch normalization was requested, but auto-check returned NOT_NEEDED "
                            "(single batch or negligible baseline drift). Skipping normalization cleanly; "
                            "set overwrite_autocheck=True (or --overwrite-autocheck) to force normalization."
                        )

            if config.generate_diagnostics:
                # Save assessment JSON
                assess_json = tables_dir / "batch_necessity_assessment.json"
                assess_json.write_text(json.dumps(batch_assessment.to_dict(), indent=2), encoding="utf-8")
                generated_tables["batch_necessity_assessment"] = str(assess_json)

                # Save marker metrics CSV if available
                m_df = batch_assessment.to_dataframe()
                if not m_df.empty:
                    m_csv = tables_dir / "batch_necessity_metrics.csv"
                    m_df.to_csv(m_csv, index=False)
                    generated_tables["batch_necessity_metrics"] = str(m_csv)

                # Save markdown diagnostic report
                report_md = out_base / "batch_necessity_report.md"
                report_md.write_text(batch_assessment.to_markdown(), encoding="utf-8")
                generated_tables["batch_necessity_report"] = str(report_md)

                # Plot diagnostic summary
                p_summary = plots_dir / "batch_necessity_summary.png"
                plot_batch_assessment_summary(batch_assessment, output_path=p_summary)
                generated_plots["batch_necessity_summary"] = str(p_summary)

        except Exception as exc:
            logger.warning(f"Batch effect assessment encountered an exception: {exc}")

    # -----------------------------------------------------------------
    # Step 3.1: Inter-Batch Normalization (CytoNorm / Quantile)
    # -----------------------------------------------------------------
    unique_batches = {s.batch_id for s in resolved_samples}
    if norm_cfg.enabled and norm_cfg.method != "none" and len(unique_batches) < 2:
        logger.info(
            f"Step 3.1: Skipping inter-batch normalization ({norm_cfg.method.upper()}) — "
            f"only {len(unique_batches)} unique batch detected."
        )
    elif norm_cfg.enabled and norm_cfg.method != "none" and skip_norm_by_autocheck:
        logger.info(
            f"Step 3.1: Skipping inter-batch normalization ({norm_cfg.method.upper()}) per automated batch assessment "
            f"({batch_assessment.recommendation if batch_assessment else 'NOT_NEEDED'}). "
            "Pass overwrite_autocheck=True (--overwrite-autocheck) to override."
        )
    elif norm_cfg.enabled and norm_cfg.method != "none":
        logger.info(f"Step 3.1: Executing inter-batch normalization ({norm_cfg.method.upper()})...")

        if norm_cfg.method == "cytonorm":
            has_controls = any(s.is_control for s in resolved_samples)
            if not has_controls:
                logger.warning(
                    "CytoNorm requested but no samples tagged as controls (is_control=True). "
                    "Falling back to Anchor-Free Quantile Normalization."
                )
                norm_cfg.method = "quantile"

        if norm_cfg.method == "cytonorm":
            model = fit_cytonorm(
                cohort_dfs=working_dfs,
                sample_info=sample_info_map,
                config=norm_cfg,
            )
            norm_model = model

            # Apply normalization to all samples
            for s_id, df in working_dfs.items():
                b_id = sample_info_map[s_id].batch_id
                working_dfs[s_id] = model.transform(df, b_id)

            if norm_cfg.save_model:
                model_dest = out_base / "cytonorm_model.json"
                model.save_json(model_dest)
                generated_tables["cytonorm_model"] = str(model_dest)

        elif norm_cfg.method == "quantile":
            q_norm = fit_anchor_free_quantile_normalizer(
                cohort_dfs=working_dfs,
                sample_info=sample_info_map,
                config=norm_cfg,
            )
            norm_model = q_norm

            for s_id, df in working_dfs.items():
                b_id = sample_info_map[s_id].batch_id
                working_dfs[s_id] = q_norm.transform(df, b_id)

        # Calculate Batch Invariance Metrics
        try:
            target_markers = norm_model.markers if norm_model and hasattr(norm_model, "markers") else None
            metrics_df = calculate_batch_invariance_metrics(
                raw_cohort=raw_cohort_backup,
                normalized_cohort=working_dfs,
                sample_info=sample_info_map,
                markers=target_markers,
                controls_only=norm_cfg.method == "cytonorm",
            )
            if not metrics_df.empty:
                metrics_csv = tables_dir / "batch_invariance_metrics.csv"
                metrics_df.to_csv(metrics_csv, index=False)
                generated_tables["batch_invariance_metrics"] = str(metrics_csv)

                avg_emd_red = metrics_df["emd_reduction_pct"].mean()
                logger.info(
                    f"Batch Normalization complete: Mean EMD reduction = {avg_emd_red:+.2f}% "
                    f"across {len(metrics_df)} marker/batch comparisons."
                )
        except Exception as met_err:
            logger.warning(f"Failed to calculate batch invariance metrics: {met_err}")

    # -----------------------------------------------------------------
    # Step 3.2: Cohort Downsampling & Balancing
    # -----------------------------------------------------------------
    down_cfg = config.downsampling
    if down_cfg.enabled and down_cfg.mode != "none":
        logger.info(f"Step 3.2: Executing deterministic cohort downsampling (mode='{down_cfg.mode}')...")
        working_dfs, down_summary = downsample_cohort(
            cohort_dfs=working_dfs,
            config=down_cfg,
        )
        down_summary_csv = tables_dir / "downsampling_summary.csv"
        down_summary.to_csv(down_summary_csv, index=False)
        generated_tables["downsampling_summary"] = str(down_summary_csv)

    # Update sample tracking yields
    for s_info in resolved_samples:
        if s_info.sample_id in working_dfs:
            final_cnt = len(working_dfs[s_info.sample_id])
            s_info.prepared_event_count = final_cnt
            s_info.retention_pct_of_raw = (
                (final_cnt / s_info.raw_event_count * 100.0) if s_info.raw_event_count > 0 else 0.0
            )
            s_info.retention_pct_of_clean = (
                (final_cnt / s_info.clean_event_count * 100.0) if s_info.clean_event_count > 0 else 0.0
            )

    # -----------------------------------------------------------------
    # Step 3.3: Diagnostics & Figure Generation
    # -----------------------------------------------------------------
    if config.generate_diagnostics:
        # A. Density Comparison (if normalized)
        if norm_cfg.enabled and norm_cfg.method != "none" and norm_model is not None:
            try:
                density_plot_path = plots_dir / "step3_batch_density_comparison.png"
                plot_batch_density_comparison(
                    raw_cohort=raw_cohort_backup,
                    normalized_cohort=working_dfs,
                    sample_info=sample_info_map,
                    markers=norm_model.markers[:4],
                    output_path=density_plot_path,
                    controls_only=norm_cfg.method == "cytonorm",
                )
                generated_plots["batch_density_comparison"] = str(density_plot_path)
            except Exception as d_err:
                logger.warning(f"Could not render density comparison plot: {d_err}")

        # B. EMD Reduction Bar Chart
        if metrics_df is not None and not metrics_df.empty:
            try:
                metrics_plot_path = plots_dir / "step3_batch_metrics_reduction.png"
                plot_batch_metrics_summary(metrics_df, output_path=metrics_plot_path)
                generated_plots["batch_metrics_reduction"] = str(metrics_plot_path)
            except Exception as m_err:
                logger.warning(f"Could not render metrics reduction plot: {m_err}")

        # C. Downsampling Balance Summary
        try:
            down_balance_path = plots_dir / "step3_cohort_downsampling_balance.png"
            plot_cohort_downsampling_balance(resolved_samples, output_path=down_balance_path)
            generated_plots["cohort_downsampling_balance"] = str(down_balance_path)
        except Exception as b_err:
            logger.warning(f"Could not render downsampling balance plot: {b_err}")

    # -----------------------------------------------------------------
    # Step 3.4: FCS & Pooled Cohort Data Exports
    # -----------------------------------------------------------------
    if config.export_individual_fcs or config.downsampling.export_fcs:
        fcs_dir.mkdir(parents=True, exist_ok=True)
        for s_id, df in working_dfs.items():
            if s_id in parsed_fcs_sources:
                out_fcs = fcs_dir / f"{s_id}_prepared.fcs"
                export_downsampled_fcs(df, parsed_fcs_sources[s_id], out_fcs)
                exported_fcs[s_id] = str(out_fcs)

    # Construct Pooled Cohort DataFrame (with sample_id & batch_id categorical metadata)
    pooled_df: Optional[pd.DataFrame] = None
    if config.export_pooled_cohort:
        pooled_dfs = []
        for s_id, df in working_dfs.items():
            df_labeled = df.copy()
            if "sample_id" in df_labeled.columns:
                df_labeled["sample_id"] = s_id
            else:
                df_labeled.insert(0, "sample_id", s_id)

            if "batch_id" in df_labeled.columns:
                df_labeled["batch_id"] = sample_info_map[s_id].batch_id
            else:
                df_labeled.insert(1, "batch_id", sample_info_map[s_id].batch_id)
            pooled_dfs.append(df_labeled)

        if pooled_dfs:
            pooled_df = pd.concat(pooled_dfs, ignore_index=True)
            pooled_csv = tables_dir / "cohort_prepared_events.csv"
            pooled_df.to_csv(pooled_csv, index=False)
            generated_tables["pooled_cohort_events"] = str(pooled_csv)

    # Generate Summary Table
    summary_rows = [s.to_dict() for s in resolved_samples]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = tables_dir / "cohort_batch_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    generated_tables["cohort_batch_summary"] = str(summary_csv)

    elapsed = time.perf_counter() - t_start
    dur_str = format_duration(elapsed)
    logger.info(
        f"Step 3 Batch Preparation completed in {dur_str} ({elapsed:.2f}s) for {len(resolved_samples)} samples. "
        f"Results saved to '{out_base}'."
    )

    return CohortBatchResult(
        samples=resolved_samples,
        prepared_dfs=working_dfs,
        summary_df=summary_df,
        pooled_df=pooled_df,
        normalization_model=norm_model,
        metrics_df=metrics_df,
        batch_assessment=batch_assessment,
        generated_plots=generated_plots,
        generated_tables=generated_tables,
        exported_fcs_paths=exported_fcs,
        output_dir=str(out_base),
        execution_time_seconds=round(elapsed, 2),
    )

