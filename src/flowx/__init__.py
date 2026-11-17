"""FlowX: Flow Cytometry Analysis and Visualization Framework.

An open-source, modular Python framework for parsing, compensating,
transforming, gating, clustering, and visualizing flow cytometry FCS files.
"""

from flowx.io.config import PipelineConfig
from flowx.io.parser import FCSData, parse_fcs
from flowx.pipeline import FlowCytometryPipeline
from flowx.workspace.session import WorkspaceSession

__version__ = "1.0.0"

__all__ = [
    "FlowCytometryPipeline",
    "PipelineConfig",
    "FCSData",
    "parse_fcs",
    "WorkspaceSession",
    "__version__",
]
