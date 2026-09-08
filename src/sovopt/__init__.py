"""SOVOPT — a sovereign optimization engine.

LP / MILP / QP solver core built from mathematical foundations.
SIH 2026 · PS 26119 · MRPL.
"""

__version__ = "0.1.0"

from .core.problem import Problem, Solution, Status, ObjSense, VarKind
from .core.sparse import SparseMatrix
from .core.tolerances import Tolerances, INF

__all__ = ["Problem", "Solution", "Status", "ObjSense", "VarKind",
           "SparseMatrix", "Tolerances", "INF", "__version__"]
