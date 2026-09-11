"""FoldGuard: decide whether a predicted protein structure is good enough for what you are about to do with it."""

from .core import Structure, Residue, Region, PaeSegment, parse_structure, attach_pae, parse_site, ParseError
from .verdict import assess, Report, Finding, Level, TASK_RULES

__version__ = "0.1.0"
__all__ = [
    "Structure", "Residue", "Region", "PaeSegment", "parse_structure", "attach_pae",
    "parse_site", "ParseError", "assess", "Report", "Finding", "Level", "TASK_RULES",
]
