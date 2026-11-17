"""Streamlit Community Cloud entrypoint. Select Python 3.11 at deployment."""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
os.environ["TRAFFIC_CLOUD"] = "1"
os.environ.setdefault("MODEL_PATH", str(ROOT / "UI" / "models" / "best.pt"))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch

torch.set_num_threads(2)
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
