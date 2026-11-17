import os
import tempfile
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

from traffic_web.processor import Options, analyze_video
from traffic_web.devices import device_info, select_device

BASE = Path(__file__).resolve().parent
CLOUD = os.getenv("TRAFFIC_CLOUD") == "1"
UPLOAD_MB = 50 if CLOUD else 100
st.set_page_config(page_title="Traffic Vision · Phân tích biển báo", page_icon="🚦", layout="wide")


@st.cache_resource(max_entries=2)
def load_detector(path, modified, device):
    from ultralytics import YOLO
    return YOLO(path).to("cuda:0" if device == "0" else "cpu")


@st.cache_data(ttl=60)
def hardware():
    return device_info()


@st.cache_resource
def load_ocr():
    import easyocr
    return easyocr.Reader(["en"], gpu=False, model_storage_directory=os.getenv("OCR_MODEL_DIR", str(BASE / "easyocr_models")))


@st.cache_resource
def inference_lock():
    return threading.Lock()


st.title("🚦 Traffic Vision")
st.write("Nhận diện biển báo giao thông Việt Nam trong video của bạn.")
st.caption("YOLO · 32 nhóm biển báo · Đọc số tốc độ bằng OCR tùy chọn")
previous_camera = st.session_state.get("_live_ctx")
live_playing = bool(previous_camera and previous_camera.state.playing)
mode = st.segmented_control("Chế độ", ["Video tải lên", "Camera trực tiếp"], default="Video tải lên",
                            disabled=live_playing)
with st.sidebar:
    st.subheader("Cấu hình phân tích")
    device_choice = st.selectbox("Thiết bị nhận diện", ["CPU"] if CLOUD else ["Tự động", "GPU", "CPU"],
                                disabled=live_playing)
    try:
        info = hardware()
        st.caption(f"{info['name']} · Torch {info['torch']}")
        device = select_device(device_choice, info)
        device_error = ""
    except (ImportError, ValueError, RuntimeError) as exc:
        device, device_error = "cpu", str(exc)
        st.warning(device_error)
    confidence = st.slider("Ngưỡng tin cậy", 0.1, 0.9, 0.5, 0.05, disabled=live_playing)
    image_size = st.select_slider("Kích thước inference", options=[320, 480, 640], value=320 if CLOUD else 480, disabled=live_playing)
    max_seconds = st.slider("Số giây tối đa cần xử lý", 5, 30 if CLOUD else 120, 10 if CLOUD else 30, 5)
    use_ocr = st.checkbox("Đọc số giới hạn tốc độ", value=False, disabled=live_playing,
                            help="Chậm hơn trên CPU. Lần đầu cần Internet để tải model EasyOCR.")
    ocr_interval = st.slider("Đọc lại OCR sau mỗi N frame", 1, 30, 10, disabled=not use_ocr or live_playing,
                             help="Ghép vùng biển theo vị trí giữa hai frame; N=1 đọc mọi frame. OCR chạy CPU.")
    st.info(f"YOLO chạy {'GPU' if device == '0' else 'CPU'}. Bắt đầu với 10–30 giây để kiểm tra tốc độ.")
    st.caption("Video kết quả giữ tốc độ phát gốc, không có âm thanh. Ảnh được thu nhỏ tối đa 960 px mỗi chiều.")
    if CLOUD:
        st.caption("Bản demo Cloud: tối đa 50 MB/video, 30 giây đầu. Dùng CPU của máy hosting; OCR tùy chọn.")

model_path = Path(os.getenv("MODEL_PATH", str(BASE / "UI" / "models" / "best.pt")))
if not model_path.is_file():
    st.error("Chưa có model. Đặt best.pt vào UI/models hoặc cấu hình biến MODEL_PATH trên máy chủ.")

if mode == "Camera trực tiếp":
    from traffic_web.camera_ui import render_camera
    render_camera(model_path, device, device_error, confidence, image_size, use_ocr, ocr_interval,
                  load_detector, load_ocr, inference_lock)
    st.stop()

upload_col, detail_col = st.columns([2, 1])
with upload_col:
    st.subheader("1. Chọn video")
    uploaded = st.file_uploader("Video giao thông", type=["mp4", "mov", "avi", "mkv"],
                                max_upload_size=UPLOAD_MB, help=f"Tối đa {UPLOAD_MB} MB. Khuyến nghị MP4 H.264.")
    if uploaded:
        st.caption(f"{uploaded.name} · {uploaded.size / 1024 ** 2:.1f} MB")
        st.caption(f"Chỉ phân tích {max_seconds} giây đầu video. Nếu biển nằm sau đoạn này, hãy cắt đoạn có biển rồi tải lên.")
with detail_col:
    st.subheader("Bạn sẽ nhận được")
    st.write("Video có khung nhận diện, tên biển và confidence; bảng detection theo thời gian; file CSV để phân tích tiếp.")
    st.caption("Mỗi lần phát hiện trên một frame được tính là một detection. Đây chưa phải số biển riêng biệt.")


if uploaded is not None:
    identity = (uploaded.file_id, confidence, image_size, max_seconds, use_ocr, device, ocr_interval)
    if st.session_state.get("result_identity") != identity:
        st.session_state.pop("result", None)
    if st.button("Phân tích video", type="primary", disabled=not model_path.is_file() or bool(device_error)):
        if uploaded.size > UPLOAD_MB * 1024 ** 2:
            st.error(f"Video vượt quá {UPLOAD_MB} MB. Hãy chọn video nhỏ hơn.")
        elif not inference_lock().acquire(blocking=False):
            st.warning("Máy chủ đang phân tích video khác. Vui lòng thử lại sau.")
        else:
            st.session_state.pop("result", None)
            bar = st.progress(0.0, text="Đang tải model…")
            preview = st.empty()
            try:
                model = load_detector(str(model_path), model_path.stat().st_mtime_ns, device)
                reader = load_ocr() if use_ocr else None
                with tempfile.TemporaryDirectory(prefix="traffic-video-") as work:
                    source = Path(work) / ("input" + Path(uploaded.name).suffix.lower())
                    source.write_bytes(uploaded.getbuffer())

                    def update(fraction, frame, count):
                        bar.progress(float(fraction), text=f"Đã phân tích {count} frame…")
                        preview.image(frame, channels="BGR", width="stretch", alt="Frame video với các biển báo được nhận diện")

                    result = analyze_video(source, Path(work) / "output", model,
                                           Options(confidence=confidence, image_size=image_size, max_seconds=max_seconds,
                                                   device=device, ocr_interval=ocr_interval), reader, update)
                    result["model_classes"] = len(model.names)
                    result["confidence"] = confidence
                    result["image_size"] = image_size
                    result["video"] = result["video"].read_bytes()
                    result["csv"] = result["csv"].read_bytes()
                st.session_state.result = result
                st.session_state.result_identity = identity
                bar.progress(1.0, text="Hoàn thành")
                preview.empty()
            except Exception as exc:
                st.error(f"Không hoàn thành phân tích: {exc}")
                bar.empty()
                preview.empty()
            finally:
                inference_lock().release()
    result = st.session_state.get("result")
    if result:
        st.divider()
        st.subheader("2. Kết quả phân tích")
        if "model_classes" in result:
            st.success(f"Đã tải best.pt: {result['model_classes']} lớp; đã phân tích {result['frames']} frame. "
                       f"Ảnh inference {result['image_size']}, confidence {result['confidence']:.2f}.")
        metrics = st.columns(4)
        metrics[0].metric("Video đã phân tích", f"{result['seconds']:.1f} giây")
        metrics[1].metric("Detection trên các frame", result["detections"])
        metrics[2].metric("Nhóm biển xuất hiện", len(result["counts"]))
        metrics[3].metric("Thời gian xử lý", f"{result['elapsed']:.1f} giây")
        if "timings" in result:
            st.caption(f"Tốc độ toàn bộ pipeline: {result['processing_fps']:.1f} frame/giây · "
                       f"Video gốc: {result['source_fps']:.1f} FPS · "
                       f"Thiết bị: {'GPU' if result['device'] == '0' else 'CPU'} · "
                       f"Số lần gọi OCR: {result['ocr_calls']}")
            with st.expander("Chi tiết thời gian xử lý"):
                st.dataframe(pd.DataFrame([{"Công đoạn": stage, "Giây": round(seconds, 3)}
                                           for stage, seconds in result["timings"].items()]),
                             hide_index=True, alt="Thời gian YOLO, OCR, xem trước và xuất video")
                st.caption("Bao gồm lần inference đầu và xuất MP4; chưa gồm tải model, khởi tạo OCR và upload. "
                           "Đây là tốc độ xử lý file, chưa phải FPS camera trực tiếp.")
        if result["limit_reached"]:
            st.info("Đã đạt giới hạn thời lượng cấu hình. Nếu video dài hơn, phần còn lại chưa được phân tích.")
        st.video(result["video"], alt="Video kết quả nhận diện biển báo")
        left, right = st.columns(2)
        left.download_button("Tải video kết quả", result["video"], "traffic-result.mp4", "video/mp4")
        right.download_button("Tải bảng CSV", result["csv"], "traffic-detections.csv", "text/csv")
        if result["counts"]:
            st.bar_chart(pd.DataFrame.from_dict(result["counts"], orient="index", columns=["Detection"]), alt="Số detection trên các frame theo nhóm biển")
        else:
            st.info("Model đã chạy nhưng không có detection vượt ngưỡng trong đoạn được phân tích. "
                    "Thử ảnh inference 640, confidence 0,25 và một đoạn video có biển rõ, đủ lớn. "
                    "Model chỉ hỗ trợ 32 nhóm biển đã huấn luyện.")
        import io
        table = pd.read_csv(io.BytesIO(result["csv"]))
        st.dataframe(table.head(1000), hide_index=True, width="stretch", alt="Bảng detection và thời điểm xuất hiện")
        st.caption("Hiển thị tối đa 1.000 dòng; CSV chứa toàn bộ detection. Tọa độ theo kích thước video đầu ra.")
