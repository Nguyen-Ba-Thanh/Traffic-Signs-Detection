"""Small runtime device probe; no downloads or model loading."""
def device_info():
    import torch
    import torch.version
    if torch.cuda.is_available():
        return {"available": True, "name": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "cuda": torch.version.cuda}
    return {"available": False, "name": "CPU", "torch": torch.__version__, "cuda": torch.version.cuda}


def select_device(choice, info):
    if choice == "GPU" and not info["available"]:
        raise ValueError("GPU chưa sẵn sàng. Hãy chạy môi trường CUDA hoặc chọn CPU.")
    return "0" if info["available"] and choice != "CPU" else "cpu"
