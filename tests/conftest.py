import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Commands that set up logging during the tests write their log files here, not into logs/.
import tempfile

os.environ.setdefault("MTAH_LOG_DIR", tempfile.mkdtemp(prefix="mtah_logs_"))
