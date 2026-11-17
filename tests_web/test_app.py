from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_initial_page_loads():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "🚦 Traffic Vision"
    assert app.slider[0].value == 0.5
    assert app.checkbox[0].value is False


def test_unavailable_gpu_shows_explanation(monkeypatch):
    import streamlit as st
    st.cache_data.clear()
    monkeypatch.setattr("traffic_web.devices.device_info", lambda: {
        "available": False, "name": "CPU", "torch": "test-cpu", "cuda": None})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    app.selectbox[0].select("GPU").run(timeout=30)
    assert not app.exception
    assert any("GPU chưa sẵn sàng" in item.value for item in app.warning)
    st.cache_data.clear()
