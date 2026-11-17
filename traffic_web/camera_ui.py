"""Streamlit camera section, loaded only when selected."""
import streamlit as st

from traffic_web.processor import Options


def render_camera(model_path, device, device_error, confidence, image_size, use_ocr, ocr_interval,
                  load_detector, load_ocr, inference_lock):
    st.subheader("Camera trực tiếp")
    st.write("Chuẩn bị model, bấm START và cho phép trình duyệt sử dụng camera. Bấm STOP để dừng hoặc đổi cấu hình.")
    st.caption("Camera của người truy cập được truyền về máy chạy ứng dụng. Không thu âm, không lưu video camera.")
    st.info("Khởi đầu với kích thước 480 và tắt OCR. Khi hosting, model dùng CPU/GPU của máy hosting.")
    try:
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
        from traffic_web.live import LiveDetector
    except ImportError:
        st.error("Thiếu thư viện camera. Cài requirements-web.txt bằng Python của môi trường đang chạy.")
        return
    config = (str(model_path), model_path.stat().st_mtime_ns if model_path.is_file() else 0,
              device, confidence, image_size, use_ocr, ocr_interval)
    if st.session_state.get("_live_config") != config:
        st.session_state.pop("_live_worker", None)
    previous = st.session_state.get("_live_ctx")
    playing = bool(previous and previous.state.playing)
    if st.button("Chuẩn bị camera", type="primary", disabled=bool(device_error) or not model_path.is_file() or playing):
        lock = inference_lock()
        if not lock.acquire(blocking=False):
            st.warning("Máy chủ đang xử lý video khác. Vui lòng thử lại sau.")
        else:
            try:
                with st.spinner("Đang tải model camera…"):
                    model = load_detector(str(model_path), model_path.stat().st_mtime_ns, device)
                    reader = load_ocr() if use_ocr else None
                    st.session_state._live_worker = LiveDetector(
                        model, Options(confidence=confidence, image_size=image_size, device=device,
                                       ocr_interval=ocr_interval), lock, reader)
                    st.session_state._live_config = config
            except Exception as exc:
                st.error(f"Không chuẩn bị được camera: {exc}")
            finally:
                lock.release()
    worker = st.session_state.get("_live_worker")
    if worker is None:
        return
    st.success(f"Đã tải best.pt: {len(worker.model.names)} lớp. Thiết bị: {'GPU' if device == '0' else 'CPU'}.")
    try:
        ice_servers = st.secrets.get("webrtc", {}).get("ice_servers", [{"urls": ["stun:stun.l.google.com:19302"]}])
    except FileNotFoundError:
        ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    ctx = webrtc_streamer(
        key="traffic-camera", mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": ice_servers},
        media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                            "frameRate": {"ideal": 15, "max": 30}}, "audio": False},
        queued_video_frames_callback=worker.process_queued, on_video_ended=worker.reset,
        async_processing=True,
    )
    st.session_state._live_ctx = ctx

    @st.fragment(run_every=1)
    def metrics():
        stats = worker.snapshot()
        columns = st.columns(4)
        columns[0].metric("FPS xử lý gần đây", f"{stats['fps']:.1f}")
        columns[1].metric("Xử lý một frame", f"{stats['processing_ms']:.0f} ms")
        columns[2].metric("Frame bỏ qua", stats["dropped"])
        columns[3].metric("Frame đã nhận diện", stats["processed"])
        st.caption("Thời gian xử lý tính trên máy chủ, chưa gồm mạng và hiển thị trình duyệt. Frame cũ được bỏ để giảm trễ.")
        if stats["error"]:
            st.error(f"Lỗi nhận diện: {stats['error']}")
        if stats["busy"]:
            st.warning("Máy chủ đang bận; camera đang trả hình chưa nhận diện.")
        if not stats["processed"]:
            st.caption("Chưa có frame được nhận diện. Bấm START và cấp quyền camera. Nếu đang chờ kết nối, kiểm tra WebRTC/TURN.")
        elif not stats["rows"]:
            st.info("Đã nhận và xử lý hình, nhưng chưa phát hiện biển vượt ngưỡng. Bấm STOP, thử confidence 0,25 "
                    "và ảnh 640; đưa biển rõ, đủ lớn vào khung hình rồi START lại.")
        if stats["rows"]:
            st.dataframe(stats["rows"][-30:], hide_index=True, alt="Các detection camera gần đây")
    metrics()
    st.caption("Truy cập từ xa cần HTTPS và kết nối WebRTC. Nếu START bị kẹt khi hosting, cần cấu hình TURN trong secrets; xem WEB_GUIDE.md.")
