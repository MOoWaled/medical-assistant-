"""Local-only Streamlit entry point for the ML and deep-learning models."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


# app.py reads this flag before it builds its sidebar.  Keeping one shared UI
# prevents the friend edition from drifting away from the primary application.
os.environ["LOCAL_MODELS_ONLY"] = "1"
runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
