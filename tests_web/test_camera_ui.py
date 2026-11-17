from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_can_switch_to_camera_without_starting_capture():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=30)
    app.get("segmented_control")[0].set_value("Camera trực tiếp").run(timeout=30)
    assert not app.exception
    assert any(button.label == "Chuẩn bị camera" for button in app.button)
