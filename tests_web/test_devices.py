import pytest
from traffic_web.devices import select_device


def test_device_selection():
    assert select_device("Tự động", {"available": True}) == "0"
    assert select_device("CPU", {"available": True}) == "cpu"
    assert select_device("Tự động", {"available": False}) == "cpu"
    with pytest.raises(ValueError, match="GPU"):
        select_device("GPU", {"available": False})
