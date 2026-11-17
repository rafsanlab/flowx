"""Shared pytest fixtures and deterministic synthetic FCS factory for FlowX tests.

All unit and integration tests use these self-contained fixtures so the test suite
runs with zero skips and zero dependency on external datasets in `data/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import flowio
import numpy as np
import pandas as pd
import pytest
import yaml


def create_fcs_file(
    filepath: Path,
    events_df: pd.DataFrame,
    pnn_labels: Optional[List[str]] = None,
    pns_labels: Optional[List[str]] = None,
    metadata: Optional[Dict[str, str]] = None,
    spillover_matrix: Optional[pd.DataFrame] = None,
) -> Path:
    """Create a valid binary FCS 3.1 file from a DataFrame and custom metadata."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    channels = list(pnn_labels) if pnn_labels is not None else list(events_df.columns)
    stains = list(pns_labels) if pns_labels is not None else ["" for _ in channels]
    if len(stains) < len(channels):
        stains.extend(["" for _ in range(len(channels) - len(stains))])

    meta_dict: Dict[str, str] = {}
    for i, _ in enumerate(channels, 1):
        meta_dict[f"P{i}R"] = "262144"
        meta_dict[f"P{i}G"] = "1.0"

    if metadata:
        for k, v in metadata.items():
            meta_dict[str(k).lstrip("$")] = str(v)

    if spillover_matrix is not None:
        dets = list(spillover_matrix.index)
        n_dets = len(dets)
        spill_vals = [str(n_dets)] + dets
        for i in range(n_dets):
            for j in range(n_dets):
                spill_vals.append(str(float(spillover_matrix.iloc[i, j])))
        spill_str = ",".join(spill_vals)
        meta_dict["SPILLOVER"] = spill_str
        meta_dict["SPILL"] = spill_str

    arr = events_df[channels].to_numpy(dtype=np.float32).flatten()
    with open(filepath, "wb") as fh:
        flowio.create_fcs(
            fh,
            arr,
            channels,
            opt_channel_names=stains,
            metadata_dict=meta_dict,
        )
    return filepath


@pytest.fixture(scope="session")
def cytotrol_fcs_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize a 12-channel CytoTrol PBMC FCS file with 7-channel spillover matrix."""
    base = tmp_path_factory.mktemp("cytotrol")
    fcs_path = base / "CytoTrol_CytoTrol_1.fcs"

    rng = np.random.default_rng(42)
    n = 1200
    pnn = [
        "FSC-A", "FSC-H", "FSC-W", "SSC-A",
        "B710-A", "R660-A", "R780-A", "V450-A", "V545-A", "G560-A", "G780-A",
        "Time",
    ]
    pns = [
        "", "", "", "",
        "CD4", "CD27", "CD8", "CD3", "CD45RA", "CCR7", "CD38",
        "",
    ]
    fsc_a = rng.normal(110000, 15000, n).clip(10000, 250000)
    df = pd.DataFrame({
        "FSC-A": fsc_a,
        "FSC-H": fsc_a * 0.88 + rng.normal(0, 1500, n),
        "FSC-W": rng.normal(72000, 4000, n),
        "SSC-A": rng.normal(45000, 10000, n).clip(5000, 250000),
        "B710-A": rng.normal(2500, 800, n),
        "R660-A": rng.normal(1800, 600, n),
        "R780-A": rng.normal(2200, 700, n),
        "V450-A": rng.normal(3500, 900, n),
        "V545-A": rng.normal(2000, 500, n),
        "G560-A": rng.normal(1600, 500, n),
        "G780-A": rng.normal(1400, 400, n),
        "Time": np.linspace(0.0, 60.0, n),
    })

    fluoro = ["B710-A", "R660-A", "R780-A", "V450-A", "V545-A", "G560-A", "G780-A"]
    spill_mat = np.eye(7, dtype=np.float64)
    spill_mat[0, 3] = 0.04
    spill_mat[1, 2] = 0.08
    spill_df = pd.DataFrame(spill_mat, index=fluoro, columns=fluoro)

    return create_fcs_file(
        fcs_path,
        df,
        pnn_labels=pnn,
        pns_labels=pns,
        metadata={"CYT": "LSRII", "DATE": "15-JAN-2026"},
        spillover_matrix=spill_df,
    )


@pytest.fixture(scope="session")
def tqc_sample_fcs_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize a 14-channel PBMC tutorial FCS file (`LD1_NS+NS_A01_exp.fcs`).

    Contains realistic planted morphology and fluorescence subsets:
    - 82% intact singlets, 10% low-FSC debris, 8% high-Area/low-Height doublets
    - 94% viable (low AARD-A) and 6% dead cells (high AARD-A) with bimodal valley ~0.42
    - Distinct CD3+ CD4+, CD3+ CD8+, CD3- CD19+, and CD3- CD14+ populations
    """
    base = tmp_path_factory.mktemp("tqc_tutorial")
    fcs_path = base / "LD1_NS+NS_A01_exp.fcs"

    rng = np.random.default_rng(101)
    n_singlets = 4920
    n_debris = 600
    n_doublets = 480
    n_total = n_singlets + n_debris + n_doublets  # 6000

    # 1. Scatter morphology
    fsc_singlets = rng.normal(115000.0, 14000.0, n_singlets).clip(65000.0, 170000.0)
    ssc_singlets = rng.normal(52000.0, 10000.0, n_singlets).clip(20000.0, 110000.0)
    fsc_h_singlets = fsc_singlets * 0.86 + rng.normal(0.0, 1800.0, n_singlets)
    ssc_h_singlets = ssc_singlets * 0.86 + rng.normal(0.0, 1200.0, n_singlets)

    fsc_debris = rng.normal(22000.0, 4500.0, n_debris).clip(5000.0, 40000.0)
    ssc_debris = rng.normal(12000.0, 3000.0, n_debris).clip(2000.0, 30000.0)
    fsc_h_debris = fsc_debris * 0.85 + rng.normal(0.0, 800.0, n_debris)
    ssc_h_debris = ssc_debris * 0.85 + rng.normal(0.0, 600.0, n_debris)

    fsc_doublets = rng.normal(165000.0, 12000.0, n_doublets).clip(130000.0, 220000.0)
    ssc_doublets = rng.normal(75000.0, 10000.0, n_doublets).clip(40000.0, 130000.0)
    fsc_h_doublets = fsc_doublets * 0.45 + rng.normal(0.0, 2000.0, n_doublets)
    ssc_h_doublets = ssc_doublets * 0.45 + rng.normal(0.0, 1500.0, n_doublets)

    fsc_a = np.concatenate([fsc_singlets, fsc_debris, fsc_doublets])
    fsc_h = np.concatenate([fsc_h_singlets, fsc_h_debris, fsc_h_doublets])
    ssc_a = np.concatenate([ssc_singlets, ssc_debris, ssc_doublets])
    ssc_h = np.concatenate([ssc_h_singlets, ssc_h_debris, ssc_h_doublets])
    ssc_w = (ssc_a / np.maximum(ssc_h, 1.0)) * 65000.0

    # 2. Fluorescence channels (calibrated for LogicleTransform T=262144, M=4.5, W=0.5)
    aard = rng.normal(120.0, 45.0, n_total)
    # 5% dead cells on singlets with high AARD-A (~48,000) so bimodal valley is ~0.42
    dead_idx = rng.choice(n_singlets, size=int(0.05 * n_singlets), replace=False)
    aard[dead_idx] = rng.normal(48000.0, 6000.0, len(dead_idx))

    # Lineage subsets across singlets (with positive non-zero baseline so bin medians are rock-steady):
    # 0..2200: CD3+ CD4+ T cells
    # 2200..3600: CD3+ CD8+ T cells
    # 3600..4300: CD3- CD19+ B cells
    # 4300..4920: CD3- CD14+ Monocytes
    cd3 = rng.normal(200.0, 40.0, n_total)
    cd4 = rng.normal(200.0, 40.0, n_total)
    cd8 = rng.normal(200.0, 40.0, n_total)
    cd14 = rng.normal(200.0, 40.0, n_total)
    cd19 = rng.normal(200.0, 40.0, n_total)
    cd56 = rng.normal(200.0, 40.0, n_total)
    blank = rng.normal(200.0, 30.0, n_total)

    cd3[:3600] = rng.normal(28000.0, 3500.0, 3600)
    cd3[n_singlets:] = rng.normal(28000.0, 3500.0, n_total - n_singlets)
    cd4[:2200] = rng.normal(32000.0, 3500.0, 2200)
    cd8[2200:3600] = rng.normal(30000.0, 3500.0, 1400)
    cd19[3600:4300] = rng.normal(24000.0, 3000.0, 700)
    cd14[4300:4920] = rng.normal(26000.0, 3000.0, 620)

    # Stratified shuffle across 20 time bins so every bin has identical population proportions (zero drift)
    n_bins_t = 20
    perm_blocks = [np.arange(i, n_total, n_bins_t) for i in range(n_bins_t)]
    for blk in perm_blocks:
        rng.shuffle(blk)
    perm = np.concatenate(perm_blocks)

    pnn = [
        "FSC-A", "FSC-H", "SSC-A", "SSC-H", "SSC-W",
        "Ax700-A", "PE-TxRed-A", "PacBlue-A", "Qdot605-A",
        "APC-Cy7-A", "PE-Cy7-A", "Ax647-A", "AARD-A", "Time",
    ]
    pns = [
        "", "", "", "", "",
        "CD3", "CD4", "CD8", "CD14",
        "CD19", "CD56", "Blank", "Dead", "",
    ]
    df = pd.DataFrame({
        "FSC-A": fsc_a[perm],
        "FSC-H": fsc_h[perm],
        "SSC-A": ssc_a[perm],
        "SSC-H": ssc_h[perm],
        "SSC-W": ssc_w[perm],
        "Ax700-A": cd3[perm],
        "PE-TxRed-A": cd4[perm],
        "PacBlue-A": cd8[perm],
        "Qdot605-A": cd14[perm],
        "APC-Cy7-A": cd19[perm],
        "PE-Cy7-A": cd56[perm],
        "Ax647-A": blank[perm],
        "AARD-A": aard[perm],
        "Time": np.linspace(0.0, 10.0, n_total),
    })

    fluoro = pnn[5:13]
    spill = pd.DataFrame(np.eye(len(fluoro), dtype=np.float64), index=fluoro, columns=fluoro)

    return create_fcs_file(
        fcs_path,
        df,
        pnn_labels=pnn,
        pns_labels=pns,
        metadata={"CYT": "LSRII", "DATE": "10-FEB-2026", "BTIM": "09:00:00", "ETIM": "09:01:00"},
        spillover_matrix=spill,
    )


@pytest.fixture(scope="session")
def navios_suite_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize clinical Beckman Coulter Navios / Kaluza and BD multi-vendor FCS files.

    Replaces legacy `bini_2024_flowcyt` and multi-vendor inspection files:
    - `Case1_A.fcs`, `Case1_O.fcs`, `Case1_P.fcs`, `Case1_N.fcs`, `Case1_G.fcs`, `Case1_K.fcs`, `Case1_B.fcs`
      (7 files matching `Case1_*.fcs` in Variant-1 Navios 10-color format with 20-bit range and gain=5.0 on scatter)
    - `Case8_A.fcs` (Variant-1 Navios file for gain sanitization testing)
    - `Case10_A.fcs` (Kaluza file with `FS INT`, `SS INT`, `SS PEAK` and no `FS PEAK` for SSC-singlet fallback)
    - `Case10_B.fcs` (Variant-2 with `FL1.INT_CD14.FITC` merged dot detector names and no spillover)
    - `Case30_A.fcs` (Variant-3 with `FS.INT`, `SS.INT`, `CD14.FITC`..`CD45.KO` and no spillover)
    - `BD_Fortessa_sample.fcs` and `BD_LSRII_sample.fcs` for multi-vendor recognition
    """
    base = tmp_path_factory.mktemp("navios_suite")
    rng = np.random.default_rng(77)
    n = 1200

    pnn_v1 = [
        "FS INT", "SS INT", "SS PEAK", "SS TOF",
        "FL1 INT", "FL2 INT", "FL3 INT", "FL4 INT", "FL5 INT",
        "FL6 INT", "FL7 INT", "FL8 INT", "FL9 INT", "FL10 INT",
        "TIME",
    ]
    pns_v1 = [
        "", "", "", "",
        "CD14-FITC", "CD19-PE", "CD13-ECD", "CD33-PC5.5", "CD34-PC7",
        "CD117-APC", "CD7-APC700", "CD16-APC750", "HLA-PB", "CD45-KO",
        "",
    ]

    fs_int = np.concatenate([
        rng.normal(95000.0, 15000.0, int(n * 0.85)).clip(40000.0, 180000.0),
        rng.normal(18000.0, 4000.0, int(n * 0.15)).clip(3000.0, 35000.0),
    ])
    ss_int = np.concatenate([
        rng.normal(45000.0, 10000.0, int(n * 0.85)).clip(15000.0, 120000.0),
        rng.normal(10000.0, 3000.0, int(n * 0.15)).clip(2000.0, 25000.0),
    ])
    ss_peak = ss_int * 0.88 + rng.normal(0.0, 1200.0, n)
    ss_tof = rng.normal(50000.0, 4000.0, n)

    df_v1 = pd.DataFrame({
        "FS INT": fs_int,
        "SS INT": ss_int,
        "SS PEAK": ss_peak,
        "SS TOF": ss_tof,
    })
    for fl in pnn_v1[4:14]:
        df_v1[fl] = rng.normal(5000.0, 1200.0, n).clip(10.0, 900000.0)
    df_v1["TIME"] = np.linspace(0.0, 360.0, n)

    fluoro_v1 = pnn_v1[4:14]
    spill_v1 = pd.DataFrame(np.eye(10, dtype=np.float64), index=fluoro_v1, columns=fluoro_v1)

    meta_v1 = {
        "CYT": "Navios",
        "DATE": "12-MAR-2026",
        "BTIM": "10:00:00",
        "ETIM": "10:06:00",
    }
    for i in range(1, 16):
        meta_v1[f"P{i}R"] = "1048576"
        meta_v1[f"P{i}G"] = "5.0" if i in (1, 2) else "1.0"

    for code in ("A", "O", "P", "N", "G", "K", "B"):
        create_fcs_file(
            base / f"Case1_{code}.fcs",
            df_v1,
            pnn_labels=pnn_v1,
            pns_labels=pns_v1,
            metadata=meta_v1,
            spillover_matrix=spill_v1,
        )

    create_fcs_file(
        base / "Case8_A.fcs",
        df_v1,
        pnn_labels=pnn_v1,
        pns_labels=pns_v1,
        metadata=meta_v1,
        spillover_matrix=spill_v1,
    )

    # Case10_A.fcs: Kaluza sample with FS INT, SS INT, SS PEAK (no FS PEAK)
    create_fcs_file(
        base / "Case10_A.fcs",
        df_v1,
        pnn_labels=pnn_v1,
        pns_labels=pns_v1,
        metadata=meta_v1,
        spillover_matrix=spill_v1,
    )

    # Case10_B.fcs: Variant-2 (dots & merged detectors, no spillover -> WARNING)
    pnn_v2 = [
        "FS.INT", "SS.INT", "SS.PEAK", "SS.TOF",
        "FL1.INT_CD14.FITC", "FL2.INT_CD19.PE", "FL3.INT_CD13.ECD",
        "FL4.INT_CD33.PC5.5", "FL5.INT_CD34.PC7", "FL6.INT_CD117.APC",
        "FL7.INT_CD7.APC700", "FL8.INT_CD16.APC750", "FL9.INT_HLA.PB",
        "FL10.INT_CD45.KO", "TIME",
    ]
    df_v2 = df_v1.copy()
    df_v2.columns = pnn_v2
    create_fcs_file(
        base / "Case10_B.fcs",
        df_v2,
        pnn_labels=pnn_v2,
        pns_labels=["" for _ in pnn_v2],
        metadata=meta_v1,
        spillover_matrix=None,
    )

    # Case30_A.fcs: Variant-3 (clean stain names as detectors, no FS.PEAK, 20-bit range, no spillover)
    pnn_v3 = [
        "FS.INT", "SS.INT",
        "CD14.FITC", "CD19.PE", "CD13.ECD", "CD33.PC5.5", "CD34.PC7",
        "CD117.APC", "CD7.APC700", "CD16.APC750", "HLA.PB", "CD45.KO",
        "TIME",
    ]
    df_v3 = pd.DataFrame({
        "FS.INT": fs_int * 6.0,  # >1,000,000 max to verify 20-bit detect_max_range()
        "SS.INT": ss_int,
    })
    for col in pnn_v3[2:12]:
        df_v3[col] = rng.normal(5000.0, 1200.0, n)
    df_v3["TIME"] = np.linspace(0.0, 180.0, n)
    meta_v3 = {"CYT": "Navios"}
    for i in range(1, len(pnn_v3) + 1):
        meta_v3[f"P{i}R"] = "1048576"
        meta_v3[f"P{i}G"] = "1.0"
    create_fcs_file(
        base / "Case30_A.fcs",
        df_v3,
        pnn_labels=pnn_v3,
        pns_labels=["" for _ in pnn_v3],
        metadata=meta_v3,
        spillover_matrix=None,
    )

    # BD Fortessa & BD LSRII files for multi-vendor recognition
    create_fcs_file(
        base / "BD_Fortessa_sample.fcs",
        df_v1[["FS INT", "SS INT", "FL1 INT", "TIME"]].rename(
            columns={"FS INT": "FSC-A", "SS INT": "SSC-A", "FL1 INT": "B515-A"}
        ),
        pnn_labels=["FSC-A", "SSC-A", "B515-A", "TIME"],
        pns_labels=["", "", "CD3", ""],
        metadata={"CYT": "BD LSRFortessa"},
    )
    create_fcs_file(
        base / "BD_LSRII_sample.fcs",
        df_v1[["FS INT", "SS INT", "FL1 INT", "TIME"]].rename(
            columns={"FS INT": "FSC-A", "SS INT": "SSC-A", "FL1 INT": "B515-A"}
        ),
        pnn_labels=["FSC-A", "SSC-A", "B515-A", "TIME"],
        pns_labels=["", "", "CD3", ""],
        metadata={"CYT": "LSRII"},
    )

    return base


@pytest.fixture(scope="session")
def synthetic_compensation_suite(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    """Synthesize a multi-color compensation dataset with bead and cell single-stained controls.

    Replaces legacy `somasundaram_2020_frfcmz2zh` 28-color dataset with a fast, exact
    ground-truth 6-color suite testing:
    - Exact spillover recovery (r > 0.999, MAE < 0.002) across `median_delta_mfi`,
      `autospill_regression`, and `hybrid` estimators
    - Carrier detection (`beads` vs `cells`)
    - Sub-cellular RBC debris removal (>55%) on cell viability controls
    - Monomer bead elliptical gating & doublet exclusion (>25%) on bead controls
    - Control manifest CSV loading and Time QC manifest ingestion
    """
    base = tmp_path_factory.mktemp("comp_suite")
    raw_dir = base / "raw"
    ctrl_dir = raw_dir / "controls"
    samples_dir = raw_dir / "samples"
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2026)
    channels = [
        "Ex 402: Em 586-A",
        "Ex 488: Em 530-A",
        "Ex 561: Em 586-A",
        "Ex 640: Em 670-A",
        "Ex 355: Em 379-A",
        "Ex 355: Em 450-A",
    ]
    stains = ["CD3", "CD4", "CD8", "CD19", "CD14", "LiveDead"]
    pnn = ["FSC-A", "FSC-H", "SSC-A", "SSC-H"] + channels + ["Time"]
    pns = ["", "", "", ""] + stains + [""]

    # Known ground-truth 6x6 spillover matrix
    true_spill = np.eye(6, dtype=np.float64)
    true_spill[0, 1] = 0.12
    true_spill[1, 2] = 0.18
    true_spill[2, 1] = 0.05
    true_spill[3, 4] = 0.09
    true_spill[4, 5] = 0.15
    true_spill[5, 4] = 0.03
    spill_df = pd.DataFrame(true_spill, index=channels, columns=channels)

    manifest_rows = []

    for idx, ch in enumerate(channels):
        if idx < 5:
            # Antibody-capture bead control: tight low-CV monomer cluster + 25% bead dimers
            # Keep FSC-A IQR tight (rcv_fsc <= 0.45) while dimers have higher SSC-A and low FSC-H/FSC-A ratio
            n_mono = 1520
            n_dimer = 480
            n_tot = n_mono + n_dimer
            fsc_mono = rng.normal(75000.0, 1800.0, n_mono)
            ssc_mono = rng.normal(45000.0, 1200.0, n_mono)
            fsc_h_mono = fsc_mono * 0.92 + rng.normal(0.0, 350.0, n_mono)
            ssc_h_mono = ssc_mono * 0.92 + rng.normal(0.0, 300.0, n_mono)

            fsc_dim = rng.normal(92000.0, 3000.0, n_dimer)
            ssc_dim = rng.normal(82000.0, 4000.0, n_dimer)
            fsc_h_dim = fsc_dim * 0.48 + rng.normal(0.0, 800.0, n_dimer)
            ssc_h_dim = ssc_dim * 0.48 + rng.normal(0.0, 800.0, n_dimer)

            fsc_a = np.concatenate([fsc_mono, fsc_dim])
            fsc_h = np.concatenate([fsc_h_mono, fsc_h_dim])
            ssc_a = np.concatenate([ssc_mono, ssc_dim])
            ssc_h = np.concatenate([ssc_h_mono, ssc_h_dim])

            # 50% negative beads (~100 MFI), 50% positive beads (~40,000 MFI)
            primary_sig = np.concatenate([
                rng.normal(100.0, 15.0, n_mono // 2),
                rng.normal(40000.0, 2500.0, n_mono - n_mono // 2),
                rng.normal(40000.0, 3000.0, n_dimer),
            ])
        else:
            # Cell viability control (idx == 5, Ex 355: Em 450-A):
            # 62% low-FSC RBC lysis debris (~44 MFI) + 38% intact cells (higher scatter, broad CV)
            n_cells = 5000
            n_debris = 8200
            n_tot = n_cells + n_debris

            fsc_cells = rng.normal(85000.0, 18000.0, n_cells).clip(35000.0, 200000.0)
            ssc_cells = rng.normal(45000.0, 14000.0, n_cells).clip(15000.0, 150000.0)
            fsc_h_cells = fsc_cells * 0.88 + rng.normal(0.0, 2000.0, n_cells)
            ssc_h_cells = ssc_cells * 0.88 + rng.normal(0.0, 1500.0, n_cells)

            fsc_deb = rng.normal(8000.0, 2500.0, n_debris).clip(1000.0, 18000.0)
            ssc_deb = rng.normal(5000.0, 1800.0, n_debris).clip(500.0, 14000.0)
            fsc_h_deb = fsc_deb * 0.85 + rng.normal(0.0, 400.0, n_debris)
            ssc_h_deb = ssc_deb * 0.85 + rng.normal(0.0, 300.0, n_debris)

            fsc_a = np.concatenate([fsc_cells, fsc_deb])
            fsc_h = np.concatenate([fsc_h_cells, fsc_h_deb])
            ssc_a = np.concatenate([ssc_cells, ssc_deb])
            ssc_h = np.concatenate([ssc_h_cells, ssc_h_deb])

            # Intact negative cells have ~350 autofluorescence, positive dead cells ~45,000, debris ~44
            primary_sig = np.concatenate([
                rng.normal(350.0, 35.0, n_cells // 2),
                rng.normal(45000.0, 3000.0, n_cells - n_cells // 2),
                rng.normal(44.0, 10.0, n_debris),
            ])

        perm_c = rng.permutation(len(primary_sig))
        fsc_a = fsc_a[perm_c]
        fsc_h = fsc_h[perm_c]
        ssc_a = ssc_a[perm_c]
        ssc_h = ssc_h[perm_c]
        primary_sig = primary_sig[perm_c]

        df_ctrl = pd.DataFrame({
            "FSC-A": fsc_a,
            "FSC-H": fsc_h,
            "SSC-A": ssc_a,
            "SSC-H": ssc_h,
        })
        for j, other_ch in enumerate(channels):
            if j == idx:
                df_ctrl[other_ch] = primary_sig
            else:
                df_ctrl[other_ch] = primary_sig * true_spill[idx, j] + rng.normal(100.0, 8.0, len(primary_sig))
        df_ctrl["Time"] = np.linspace(0.0, 30.0, len(primary_sig))

        safe_name = ch.replace(":", ",").replace("-A", "")
        ctrl_path = ctrl_dir / f"Compensation Controls_{safe_name} Stained Control_{idx+1:03d}.fcs"
        create_fcs_file(
            ctrl_path,
            df_ctrl,
            pnn_labels=pnn,
            pns_labels=pns,
            metadata={"CYT": "BD LSRFortessa"},
            spillover_matrix=spill_df,
        )
        manifest_rows.append({"channel": ch, "control_file": str(ctrl_path)})

    # Multicolor sample file
    n_samp = 1500
    fsc_s = rng.normal(95000.0, 15000.0, n_samp).clip(40000.0, 200000.0)
    df_samp = pd.DataFrame({
        "FSC-A": fsc_s,
        "FSC-H": fsc_s * 0.88 + rng.normal(0.0, 1500.0, n_samp),
        "SSC-A": rng.normal(48000.0, 10000.0, n_samp).clip(15000.0, 150000.0),
        "SSC-H": rng.normal(42000.0, 9000.0, n_samp).clip(12000.0, 130000.0),
    })
    for ch in channels:
        df_samp[ch] = rng.normal(2500.0, 800.0, n_samp)
    df_samp["Time"] = np.linspace(0.0, 45.0, n_samp)

    sample_file = samples_dir / "samples_WBC 9196_002.fcs"
    create_fcs_file(
        sample_file,
        df_samp,
        pnn_labels=pnn,
        pns_labels=pns,
        metadata={"CYT": "BD LSRFortessa"},
        spillover_matrix=spill_df,
    )

    manifest_csv = base / "wbc_controls_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)

    return {
        "dataset_dir": base,
        "ctrl_dir": ctrl_dir,
        "sample_file": sample_file,
        "manifest_csv": manifest_csv,
        "channels": channels,
        "bead_ctrl_file": ctrl_dir / "Compensation Controls_Ex 402, Em 586 Stained Control_001.fcs",
        "cell_ctrl_file": ctrl_dir / "Compensation Controls_Ex 355, Em 450 Stained Control_006.fcs",
    }


@pytest.fixture(scope="session")
def synthetic_fmo_suite(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize FMO control FCS files for automated FMO threshold calibration tests."""
    base = tmp_path_factory.mktemp("fmo_suite")
    rng = np.random.default_rng(303)
    n = 1000

    pnn = [
        "FSC-A", "FSC-H", "SSC-A", "SSC-H",
        "PE-CF594-A", "BV605-A", "PE-A", "Horizon Brilliant Blue 515-A", "Time",
    ]
    pns = ["", "", "", "", "CD80", "CD206", "CD33", "CD11b", ""]

    fmo_specs = [
        ("Specimen_001_01gx FMO80.fcs", "PE-CF594-A"),
        ("Specimen_001_01gx FMO206.fcs", "BV605-A"),
        ("Specimen_001_01gx FMO33.fcs", "PE-A"),
        ("Specimen_001_01gx FMO11b.fcs", "Horizon Brilliant Blue 515-A"),
    ]

    for fname, missing_ch in fmo_specs:
        fsc_a = rng.normal(100000.0, 15000.0, n).clip(40000.0, 200000.0)
        df = pd.DataFrame({
            "FSC-A": fsc_a,
            "FSC-H": fsc_a * 0.88 + rng.normal(0.0, 1200.0, n),
            "SSC-A": rng.normal(50000.0, 10000.0, n).clip(15000.0, 150000.0),
            "SSC-H": rng.normal(44000.0, 9000.0, n).clip(12000.0, 130000.0),
        })
        for ch in ["PE-CF594-A", "BV605-A", "PE-A", "Horizon Brilliant Blue 515-A"]:
            if ch == missing_ch:
                # FMO negative background with spillover spread (~500..4000 raw -> ~0.55..0.75 Logicle p99)
                df[ch] = rng.normal(1800.0, 900.0, n)
            else:
                df[ch] = rng.normal(25000.0, 5000.0, n)
        df["Time"] = np.linspace(0.0, 30.0, n)
        create_fcs_file(base / fname, df, pnn_labels=pnn, pns_labels=pns, metadata={"CYT": "LSRFortessa"})

    return base


@pytest.fixture(scope="session")
def synthetic_cytonorm_cohort_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize a 6-sample, 3-batch CytoNorm benchmark cohort with planted technical drift."""
    base = tmp_path_factory.mktemp("cytonorm_bench")
    rng = np.random.default_rng(404)
    n = 800

    markers = ["CD3", "CD4", "CD8", "CD19", "CD14", "HLA-DR"]
    pnn = markers + ["Time"]

    batches = [
        ("PTLG021", 0.0, 1.0),
        ("PTLG028", 12.0, 1.8),
        ("PTLG034", -4.0, 0.7),
    ]

    manifest_rows = []
    for batch_id, shift, scale in batches:
        for role in ("Unstim_Control_1", "Stim_Sample_2"):
            fname = f"Gates_{batch_id}_{role}.fcs"
            # Generate 2 phenotypic clusters in arcsinh-scale, then invert to raw counts
            c1 = rng.normal(loc=3.0, scale=0.4, size=(n // 2, len(markers)))
            c2 = rng.normal(loc=5.0, scale=0.4, size=(n - n // 2, len(markers)))
            asinh_vals = np.vstack([c1, c2]) * scale + (shift * 0.15)
            raw_vals = np.sinh(asinh_vals) * 5.0

            df = pd.DataFrame(raw_vals, columns=markers)
            df["Time"] = np.linspace(0.0, 40.0, n)
            fpath = create_fcs_file(base / fname, df, pnn_labels=pnn, metadata={"CYT": "CyTOF"})
            manifest_rows.append({
                "filepath": str(fpath),
                "sample_id": fpath.stem,
                "batch_id": batch_id,
                "is_control": "Control_1" in role,
            })

    manifest_path = base / "cohort_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    cfg_path = base / "batch_prep_cytonorm.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "manifest": str(manifest_path),
                "batch_normalization": {
                    "enabled": True,
                    "method": "cytonorm",
                    "goal": "median",
                    "n_clusters": 4,
                    "n_quantiles": 25,
                },
                "downsampling": {
                    "enabled": True,
                    "mode": "target_count",
                    "target_events": 200,
                    "random_seed": 42,
                },
            },
            fh,
        )

    return base


@pytest.fixture(scope="session")
def synthetic_conventional_dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize a complete schema-compliant conventional dataset directory.

    Includes `raw/samples/` (4 sample FCS files), `raw/controls/` (2 bead controls + 1 unstained),
    `metadata/samples.csv`, `metadata/markers.csv`, and `metadata/study_metadata.json`.
    """
    root = tmp_path_factory.mktemp("datasets_root")
    ds_dir = root / "data" / "conventional" / "synth_pbmc_cohort"
    samples_dir = ds_dir / "raw" / "samples"
    controls_dir = ds_dir / "raw" / "controls"
    meta_dir = ds_dir / "metadata"
    cfg_dir = root / "configs" / "config_run" / "conventional"
    samples_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(505)
    n = 600
    pnn = ["FSC-A", "FSC-H", "SSC-A", "SSC-H", "V450-A", "B710-A", "R780-A", "AARD-A", "Time"]
    pns = ["", "", "", "", "CD3", "CD4", "CD8", "Dead", ""]
    fluoro = ["V450-A", "B710-A", "R780-A", "AARD-A"]
    spill = pd.DataFrame(np.eye(4, dtype=np.float64), index=fluoro, columns=fluoro)

    sample_specs = [
        ("Sample_HC_01", "Healthy_Control", "2026-01-10_LSRFortessa", False),
        ("Sample_HC_02", "Healthy_Control", "2026-01-11_LSRFortessa", False),
        ("Sample_Tx_01", "Treated", "2026-01-10_LSRFortessa", False),
        ("Sample_Tx_02", "Treated", "2026-01-11_LSRFortessa", False),
    ]

    samples_rows = []
    for sid, cond, batch, is_ctrl in sample_specs:
        fsc_a = rng.normal(100000.0, 15000.0, n).clip(30000.0, 220000.0)
        df = pd.DataFrame({
            "FSC-A": fsc_a,
            "FSC-H": fsc_a * 0.88 + rng.normal(0.0, 1200.0, n),
            "SSC-A": rng.normal(48000.0, 10000.0, n).clip(15000.0, 150000.0),
            "SSC-H": rng.normal(42000.0, 9000.0, n).clip(12000.0, 130000.0),
            "V450-A": rng.normal(20000.0, 4000.0, n),
            "B710-A": rng.normal(15000.0 if cond == "Healthy_Control" else 28000.0, 4000.0, n),
            "R780-A": rng.normal(12000.0, 3000.0, n),
            "AARD-A": rng.normal(100.0, 40.0, n),
            "Time": np.linspace(0.0, 30.0, n),
        })
        fname = f"{sid}.fcs"
        create_fcs_file(
            samples_dir / fname,
            df,
            pnn_labels=pnn,
            pns_labels=pns,
            metadata={"CYT": "BD LSRFortessa"},
            spillover_matrix=spill,
        )
        samples_rows.append({
            "sample_id": sid,
            "filename": fname,
            "relative_path": f"raw/samples/{fname}",
            "condition": cond,
            "batch_id": batch,
            "is_control": str(is_ctrl).lower(),
            "cytometer": "BD LSRFortessa",
            "total_events": n,
            "provenance_source": "FCS $TEXT",
        })

    # Control bead files in raw/controls/ (must be excluded by resolve_cohort_samples)
    for cname in ("Compensation_Control_V450.fcs", "Unstained_Control.fcs"):
        create_fcs_file(
            controls_dir / cname,
            df,
            pnn_labels=pnn,
            pns_labels=pns,
            metadata={"CYT": "BD LSRFortessa"},
            spillover_matrix=spill,
        )

    pd.DataFrame(samples_rows).to_csv(meta_dir / "samples.csv", index=False)

    markers_rows = [
        ("FSC-A", "FSC-A", "Scatter", "Not specified in FCS or publication", "scatter", "All Events", "FCS $PnN"),
        ("FSC-H", "FSC-H", "Scatter", "Not specified in FCS or publication", "scatter", "Singlets", "FCS $PnN"),
        ("SSC-A", "SSC-A", "Scatter", "Not specified in FCS or publication", "scatter", "All Events", "FCS $PnN"),
        ("SSC-H", "SSC-H", "Scatter", "Not specified in FCS or publication", "scatter", "Singlets", "FCS $PnN"),
        ("V450-A", "CD3", "V450", "Not specified in FCS or publication", "lineage", "T Cells", "FCS $PnS"),
        ("B710-A", "CD4", "PerCP-Cy5.5", "Not specified in FCS or publication", "lineage;functional", "Helper T Cells", "FCS $PnS"),
        ("R780-A", "CD8", "APC-H7", "Not specified in FCS or publication", "lineage", "Cytotoxic T Cells", "FCS $PnS"),
        ("AARD-A", "Dead", "Amine Reactive Dye", "Not specified in FCS or publication", "viability", "Dead Cells", "FCS $PnS"),
        ("Time", "Time", "Time", "Not specified in FCS or publication", "time", "Acquisition Time", "FCS $PnN"),
    ]
    pd.DataFrame(
        markers_rows,
        columns=[
            "channel_name", "marker_name", "fluorophore", "clone_or_catalog",
            "marker_class", "cell_type_association", "provenance_source",
        ],
    ).to_csv(meta_dir / "markers.csv", index=False)

    (meta_dir / "study_metadata.json").write_text(
        json.dumps({"dataset_id": "synth_pbmc_cohort", "cytometer": "BD LSRFortessa"}, indent=2),
        encoding="utf-8",
    )

    cfg_payload = {
        "input": {
            "fcs_file": str(samples_dir),
            "metadata_file": str(meta_dir / "samples.csv"),
            "markers_file": str(meta_dir / "markers.csv"),
            "technology": "conventional",
        },
        "analysis": {
            "enable_qc": True,
            "enable_discovery": True,
        },
        "02_compensation": {
            "enabled": True,
            "source": "controls",
            "controls_dir": str(controls_dir),
        },
        "03_cleanup_gating": {
            "enabled": True,
            "enable_viability": True,
            "viability_channel": "AARD-A",
            "dump_enabled": False,
        },
        "06_statistical_analysis": {
            "enabled": True,
            "condition_column": "condition",
            "reference_group": "Healthy_Control",
            "target_group": "Treated",
        },
    }
    with open(cfg_dir / "synth_pbmc_cohort.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg_payload, fh)

    return ds_dir


@pytest.fixture(scope="session")
def population_map_json_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize a population class mapping JSON file in tmp_path."""
    base = tmp_path_factory.mktemp("pop_maps")
    json_path = base / "flowcyt_benchmark.json"
    payload = {
        "description": "Cell population class mapping for bone marrow benchmark dataset",
        "pattern": r"_(?P<code>[A-Z0-9]+)(?:\.fcs)?$",
        "mappings": {
            "A": "Full cell population (Total)",
            "O": "T lymphocytes",
            "N": "B lymphocytes",
            "G": "Monocytes",
            "P": "Mast cells",
            "K": "Hematopoietic Stem & Progenitor Cells (HSPCs)",
            "B": "Other / unclassified cells",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


@pytest.fixture(scope="session")
def panel_harmonization_yaml_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthesize a 10-color panel harmonization YAML file in tmp_path."""
    base = tmp_path_factory.mktemp("harm_configs")
    yaml_path = base / "panel_harmonization.yaml"
    payload = {
        "name": "10-Color Leukemia / Bone Marrow Consensus Panel",
        "version": "1.0",
        "settings": {
            "on_missing": "warn",
            "drop_unmapped": False,
            "preserve_raw_names": True,
        },
        "channels": {
            "FSC-A": {"aliases": ["FS INT", "FS.INT", "FSC-A"], "type": "scatter", "required": True},
            "SSC-A": {"aliases": ["SS INT", "SS.INT", "SSC-A"], "type": "scatter", "required": True},
            "FSC-H": {"aliases": ["FS PEAK", "FS.PEAK", "FSC-H"], "type": "scatter", "required": False},
            "SSC-H": {"aliases": ["SS PEAK", "SS.PEAK", "SSC-H"], "type": "scatter", "required": False},
            "Time": {"aliases": ["TIME", "Time"], "type": "time", "required": False},
            "CD14": {"aliases": ["FL1 INT", "CD14.FITC", "CD14-FITC"], "fluorophore": "FITC", "type": "fluorescent", "required": True},
            "CD19": {"aliases": ["FL2 INT", "CD19.PE", "CD19-PE"], "fluorophore": "PE", "type": "fluorescent", "required": True},
            "CD13": {"aliases": ["FL3 INT", "CD13.ECD", "CD13-ECD"], "fluorophore": "ECD / PE-TxRed", "type": "fluorescent", "required": True},
            "CD33": {"aliases": ["FL4 INT", "CD33.PC5.5", "CD33-PC5.5"], "fluorophore": "PerCP-Cy5.5 / PC5.5", "type": "fluorescent", "required": True},
            "CD34": {"aliases": ["FL5 INT", "CD34.PC7", "CD34-PC7"], "fluorophore": "PE-Cy7 / PC7", "type": "fluorescent", "required": True},
            "CD117": {"aliases": ["FL6 INT", "CD117.APC", "CD117-APC"], "fluorophore": "APC", "type": "fluorescent", "required": True},
            "CD7": {"aliases": ["FL7 INT", "CD7.APC700", "CD7-APC700"], "fluorophore": "APC-Alexa700", "type": "fluorescent", "required": True},
            "CD16": {"aliases": ["FL8 INT", "CD16.APC750", "CD16-APC750"], "fluorophore": "APC-Alexa750", "type": "fluorescent", "required": True},
            "HLA-DR": {"aliases": ["FL9 INT", "HLA.PB", "HLA-PB"], "fluorophore": "Pacific Blue", "type": "fluorescent", "required": True},
            "CD45": {"aliases": ["FL10 INT", "CD45.KO", "CD45-KO"], "fluorophore": "Krome Orange", "type": "fluorescent", "required": True},
        },
    }
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh)
    return yaml_path


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Automatically close all Matplotlib figures after each test to prevent memory accumulation."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")
