from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_cloud_entrypoint_uses_cloud_defaults(monkeypatch):
    # Preserve the outer environment: the wrapper sets these for deployment.
    for key in ["TRAFFIC_CLOUD", "MODEL_PATH", "OMP_NUM_THREADS"]:
        monkeypatch.setenv(key, __import__("os").environ.get(key, ""))
        monkeypatch.delenv(key)
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "cloud" / "streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert app.selectbox[0].value == "CPU"
    assert app.selectbox[0].options == ["CPU"]
    assert app.select_slider[0].value == 320
    assert app.slider[1].value == 10
    assert app.slider[1].max == 30


def test_cloud_dependencies_match_web():
    root = Path(__file__).resolve().parents[1]
    web = set((root / "requirements-web.txt").read_text().splitlines())
    cloud = set((root / "cloud" / "requirements.txt").read_text().splitlines())
    assert web <= cloud
    assert "torch==2.7.1+cpu" in cloud
    assert (root / "UI" / "models" / "best.pt").stat().st_size == 57045857
