"""Frame-level ball possession for More than a Hack 2027 -- the solution package of PLAN.md.

Layers (PLAN.md section 4): ingest -> players -> ball state -> features/candidates -> emission ->
semi-Markov decoder -> MBR/output policy. `possession.run` is the end-to-end entry point.

The organiser files (games.py, starter/, evaluation/) sit at the repository root and are reused,
not copied, so the root is put on sys.path here exactly the way the starter scripts do it.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
EVAL_DIR = os.path.join(ROOT, "evaluation")
