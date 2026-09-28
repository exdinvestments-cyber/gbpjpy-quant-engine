"""Phase 1C: H1 Setup Intelligence Engine (analysis only).

Consumes canonical H1 bars plus a completed H4 ``H4AnalysisResult`` and
classifies whether a high-quality setup is developing in a direction the H4
engine permits.  A QUALIFIED setup is not a trade: no entry, stop, target,
size or order exists anywhere in this package.
"""

from .alignment import AlignmentError, align_h4_to_h1
from .config import H1Config, h1_config_from_dict, load_h1_config
from .engine import H1SetupEngine, H1SetupResult
from .report import explain_h1_setup, explain_h1_bar

__all__ = ["AlignmentError", "H1Config", "H1SetupEngine", "H1SetupResult", "align_h4_to_h1", "explain_h1_bar",
           "explain_h1_setup", "h1_config_from_dict", "load_h1_config"]
