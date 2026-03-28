"""Test configuration shared across the suite.

The project is intentionally lightweight and does not install itself as a
package in every workflow. Keep the repository root on ``sys.path`` so tests
can import project modules directly without repeating path setup in each file.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
