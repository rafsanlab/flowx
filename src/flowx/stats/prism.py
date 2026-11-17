"""GraphPad Prism table formatting and export utilities for flow cytometry cohorts.

Converts multi-sample population discovery feature sets (frequencies, event counts, MFIs)
into GraphPad Prism compatible Grouped Tables (Two-way comparison: Rows=Populations,
Columns=Conditions, Subcolumns=Sample Replicates) and Column Tables for immediate
statistical testing and publication-ready charting in Prism.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from flowx.stats.models import CohortFeatureSet


def format_prism_grouped_csv(
    matrix_df: pd.DataFrame,
    sample_metadata: Optional[pd.DataFrame] = None,
    condition_column: Optional[str] = "condition",
    row_title: str = "Population",
) -> str:
    """Format a sample-by-population matrix into a GraphPad Prism Grouped Table CSV string.

    Prism Grouped Table Structure:
      Line 1: <row_title>,<Condition_1>,<Condition_1>,...,<Condition_2>,<Condition_2>...
      Line 2: Sample_ID,<Sample_A>,<Sample_B>,...,<Sample_C>,<Sample_D>...
      Line 3+: <Population_Name>,<Val_A>,<Val_B>,...,<Val_C>,<Val_D>...

    When imported or pasted into GraphPad Prism as a "Grouped" table:
      - Main Column Headers = Experimental Conditions / Treatment Groups
      - Subcolumns = Individual Sample Replicates
      - Row Titles = Discovered Populations / Lineages

    If no condition column is present or metadata is missing, formats as a Prism
    Column Table (Columns=Populations, Rows=Samples).

    Args:
        matrix_df: DataFrame of shape [Samples x Populations].
        sample_metadata: Optional metadata DataFrame with sample annotations.
        condition_column: Column name in sample_metadata defining experimental groups.
        row_title: Header label for row title column (default: 'Population').

    Returns:
        Formatted CSV string ready to save or paste into GraphPad Prism.
    """
    has_condition = (
        condition_column is not None
        and sample_metadata is not None
        and condition_column in sample_metadata.columns
    )

    if not has_condition:
        # Standard Prism Column Table: Rows=Samples, Columns=Populations
        df_col = matrix_df.copy()
        df_col.index.name = "Sample_ID"
        return df_col.to_csv()

    # Determine condition for each sample
    sample_condition_map: Dict[str, str] = {}
    for s_id in matrix_df.index:
        if s_id in sample_metadata.index:
            val = sample_metadata.loc[s_id, condition_column]
            sample_condition_map[s_id] = str(val) if pd.notna(val) else "Unassigned"
        else:
            sample_condition_map[s_id] = "Unassigned"

    # Sort samples by condition, then sample ID to group replicates together
    sample_tuples: List[Tuple[str, str]] = [
        (sample_condition_map[s_id], str(s_id)) for s_id in matrix_df.index
    ]
    sample_tuples.sort(key=lambda t: (t[0], t[1]))

    lines: List[str] = []

    # Header Row 1: Condition Group Names
    h1 = [row_title] + [t[0] for t in sample_tuples]
    lines.append(",".join(h1))

    # Header Row 2: Individual Sample IDs (Replicates)
    h2 = ["Sample_ID"] + [t[1] for t in sample_tuples]
    lines.append(",".join(h2))

    # Data Rows: Population names and corresponding sample values
    for pop in matrix_df.columns:
        row_vals = [str(pop)]
        for _, s_id in sample_tuples:
            if s_id in matrix_df.index and pop in matrix_df.columns:
                val = matrix_df.loc[s_id, pop]
                if pd.notna(val):
                    # Format float cleanly without excessive scientific notation
                    if isinstance(val, (int, float)):
                        row_vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
                    else:
                        row_vals.append(str(val))
                else:
                    row_vals.append("")
            else:
                row_vals.append("")
        lines.append(",".join(row_vals))

    return "\n".join(lines) + "\n"


def export_prism_tables(
    feature_set: CohortFeatureSet,
    output_dir: Union[str, Path],
    condition_column: Optional[str] = "condition",
) -> Dict[str, Path]:
    """Export Prism-compatible CSV tables for viable frequencies, parent frequencies, counts, and MFIs.

    Generated Files:
      - prism_population_frequencies_viable.csv: Viable singlet percentages for 2-way ANOVA / multiple t-tests.
      - prism_population_frequencies_parent.csv: Parent percentages.
      - prism_population_counts.csv: Absolute event counts.
      - prism_marker_mfi_<marker>.csv: Optional per-marker MFI tables across populations and groups.

    Args:
        feature_set: CohortFeatureSet holding extracted cohort matrices.
        output_dir: Destination folder for exported CSV tables.
        condition_column: Metadata column holding experimental conditions.

    Returns:
        Dictionary mapping table identifier to exported Path.
    """
    dest = Path(output_dir)
    if dest.name != "prism":
        dest = dest / "prism"
    dest.mkdir(parents=True, exist_ok=True)
    exported: Dict[str, Path] = {}

    meta = feature_set.sample_metadata

    # 1. Viable frequencies
    if feature_set.viable_freq_matrix is not None and not feature_set.viable_freq_matrix.empty:
        csv_viable = format_prism_grouped_csv(
            feature_set.viable_freq_matrix,
            sample_metadata=meta,
            condition_column=condition_column,
            row_title="Population (% Viable Singlets)",
        )
        p_viable = dest / "prism_population_frequencies_viable.csv"
        p_viable.write_text(csv_viable, encoding="utf-8")
        exported["prism_frequencies_viable"] = p_viable

    # 2. Parent frequencies
    if feature_set.parent_freq_matrix is not None and not feature_set.parent_freq_matrix.empty:
        csv_parent = format_prism_grouped_csv(
            feature_set.parent_freq_matrix,
            sample_metadata=meta,
            condition_column=condition_column,
            row_title="Population (% Parent)",
        )
        p_parent = dest / "prism_population_frequencies_parent.csv"
        p_parent.write_text(csv_parent, encoding="utf-8")
        exported["prism_frequencies_parent"] = p_parent

    # 3. Population counts
    if feature_set.count_matrix is not None and not feature_set.count_matrix.empty:
        csv_counts = format_prism_grouped_csv(
            feature_set.count_matrix,
            sample_metadata=meta,
            condition_column=condition_column,
            row_title="Population (Event Count)",
        )
        p_counts = dest / "prism_population_counts.csv"
        p_counts.write_text(csv_counts, encoding="utf-8")
        exported["prism_counts"] = p_counts

    # 4. Marker MFI summary table (Consolidated MFI per population across conditions)
    if feature_set.mfi_tensor:
        for marker, mfi_df in feature_set.mfi_tensor.items():
            if not mfi_df.empty:
                safe_marker = marker.replace("/", "_").replace(" ", "_")
                csv_mfi = format_prism_grouped_csv(
                    mfi_df,
                    sample_metadata=meta,
                    condition_column=condition_column,
                    row_title=f"Population ({marker} MFI)",
                )
                p_mfi = dest / f"prism_marker_mfi_{safe_marker}.csv"
                p_mfi.write_text(csv_mfi, encoding="utf-8")
                exported[f"prism_mfi_{safe_marker}"] = p_mfi

    return exported

