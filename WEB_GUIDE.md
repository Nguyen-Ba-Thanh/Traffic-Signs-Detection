# Traffic Vision — Streamlit

## Chạy bằng Docker (CPU)

Tại thư mục dự án, bảo đảm `UI/models/best.pt` có sẵn:

```powershell
docker compose -f compose.web.yml up --build
```

Mở http://localhost:8501. Không cần VcXsrv, CUDA hay cài Python trên máy host.
Dừng bằng Ctrl+C; xóa container bằng `docker compose -f compose.web.yml down`.
Ứng dụng desktop và cấu hình Docker cũ vẫn chạy riêng.

## Chạy bằng Python

Nếu đã có môi trường `.venv-web` với đủ thư viện, chạy `./run_web.bat` trong
PowerShell. Script luôn dùng Python của dự án, tránh nhầm với lệnh `streamlit`
đã cài trong Python khác trên máy.

Python 3.11:

```powershell
python -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv-web\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv-web\Scripts\python.exe -m streamlit run app.py
```

## Sử dụng

1. Chọn video MP4/MOV/AVI/MKV, tối đa 100 MB. Codec phải đọc được bằng OpenCV.
2. Chọn confidence, kích thước inference và giới hạn 5–120 giây đầu video.
3. Có thể bật OCR; lần đầu EasyOCR cần Internet để tải weights. Docker lưu weights trong volume `ocr-models`.
4. Bấm Phân tích, xem tiến độ và ảnh xem trước, sau đó phát/tải MP4 H.264 và tải CSV.

Xử lý lần lượt từng frame, giữ FPS gốc, không kèm âm thanh. Thu nhỏ ảnh tối đa
960 px mỗi chiều; biển nhỏ có thể khó nhận diện hơn. CSV có số frame (bắt đầu từ 0),
thời gian tính bằng giây, tên lớp từ model, confidence, số tốc độ OCR và bounding box
theo ảnh đầu ra. Mỗi detection thuộc một frame, chưa tracking hay đếm biển duy nhất.
OCR đọc crop khi biển mới xuất hiện hoặc đến chu kỳ đọc lại (mặc định 10 frame).
Cache chỉ ghép vùng theo IoU giữa hai frame liền nhau, ghép một-một; nếu nhiều ứng viên
thì đọc mới. Đây là heuristic theo vị trí, không bảo đảm nhận dạng cùng biển khi che khuất/
giao cắt. Chọn chu kỳ 1 để đọc mọi frame. Cột `speed_source` trong CSV ghi `ocr` hoặc
`cache`; số OCR có thể cũ đến N-1 frame. Biển mất detection sẽ bị loại khỏi cache.

Giao diện có chọn Tự động/GPU/CPU. Tự động chọn GPU nếu Torch khởi tạo CUDA được,
ngược lại chạy CPU; chọn GPU rõ ràng mà không có CUDA sẽ báo lỗi. OCR luôn dùng CPU
để tiết kiệm VRAM. Cập nhật preview tối đa khoảng 2 lần/giây. Bảng thời gian tách YOLO
(gồm chuyển box về CPU), OCR, preview, đọc/vẽ/ghi và mã hóa MP4. FPS toàn pipeline
không gồm tải model và upload, không đồng nghĩa FPS camera trực tiếp.

## Chạy GPU NVIDIA

Driver Windows đang dùng 566.07. Cấu hình được chuẩn bị dùng PyTorch 2.7.1,
torchvision 0.22.1 và CUDA 12.6 để tránh CUDA 13.2 của image `latest`.
Không cần cài riêng CUDA Toolkit cho các wheel PyTorch.

Docker (có NVIDIA GPU được cấp vào Docker/WSL2):

```powershell
docker compose -f compose.web.yml -f compose.web.gpu.yml up --build
```

Venv GPU riêng, không thay đổi môi trường CPU đang chạy:

```powershell
python -m venv .venv-web-gpu
.\.venv-web-gpu\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv-web-gpu\Scripts\python.exe -m pip install -r requirements-web.txt
.\run_web_gpu.bat
```

Wheel GPU tải vài GB và cần thêm dung lượng giải nén. Nếu ổ chứa TEMP không đủ,
đặt TEMP/TMP sang một thư mục còn trống trước khi cài; không tự xóa dữ liệu khác.

Đo trên cùng video, mặc định warmup 2 lượt, OCR tắt, không preview:

```powershell
.\.venv-web-gpu\Scripts\python.exe benchmark_web.py videos/test.mp4 --seconds 3 --imgsz 480
```

Kết quả ở `benchmark-results.json`. Thử imgsz 320/480/640 riêng và đánh giá chất
lượng biển nhỏ trước khi chọn. Benchmark file chưa kiểm chứng độ trễ camera.

Kết quả đo trên máy hiện tại (RTX 3050 Ti Laptop, driver 566.07, Torch 2.7.1+cu126):
3 giây đầu `test.mp4`, 90 frame, imgsz 480, OCR tắt, không preview, warmup 2 lượt
và lấy trung vị 3 lượt. CPU: 6,59 FPS toàn pipeline, YOLO 114,70 ms/frame;
GPU: 23,63 FPS toàn pipeline, YOLO 23,68 ms/frame. Mỗi lượt đều có 83 detection.
Đây là số đo cục bộ, có biến động theo tải máy; không bảo đảm tốc độ hosting hoặc
độ chính xác trên video khác. Chưa benchmark OCR thực tế trong lượt đo này.
Runtime Streamlit đã nhận GPU; cấu hình Compose GPU được kiểm tra hợp lệ nhưng
chưa build/chạy container GPU trong phiên kiểm thử này.

Không có detection vẫn tạo MP4 và CSV có header. Thay video hoặc cấu hình sẽ bỏ
kết quả cũ. Upload và file trung gian nằm trong thư mục tạm và được dọn sau khi
xử lý; kết quả giữ trong bộ nhớ phiên trình duyệt. Đóng/mất phiên có thể mất kết quả.
Một tiến trình phục vụ một lượt inference tại một thời điểm; người dùng khác nhận
thông báo bận và có thể thử lại. Chưa có hàng đợi hoặc tài khoản.

## Camera trực tiếp từ trình duyệt

Sau khi cài dependencies mới (`python -m pip install -r requirements-web.txt` trong
đúng venv), chạy `run_web_gpu.bat` hoặc Docker như trên. Chọn **Camera trực tiếp**,
chọn thiết bị, kích thước 480, tắt OCR để thử trước. Bấm **Chuẩn bị camera**, rồi
**START**, cấp quyền camera và chọn thiết bị quay trong WebRTC. Bấm **STOP** trước
khi đổi chế độ/cấu hình. Thiết bị và cấu hình bị khóa khi đang phát. Camera không
bị lật trước inference. Không thu âm, không lưu video hay ảnh camera ra đĩa.

Camera của mỗi người truy cập được truyền qua WebRTC tới máy chạy ứng dụng;
`VideoCapture(0)` không được dùng, nên camera trình duyệt vẫn dùng được khi hosting.
Mỗi phiên có cache OCR và log riêng; chỉ model/lock được dùng chung. Log trên trang
chỉ giữ detection gần đây. Callback xử lý frame mới nhất trong batch, bỏ backlog;
khi model bận xử lý video khác sẽ trả hình có chữ Server busy, không xếp hàng chờ.
Ngắt luồng sẽ dọn số liệu/cache; START lại sẽ tạo một lượt đo mới.

FPS là tốc độ callback nhận diện gần đây; ms/frame chỉ đo xử lý trên server, không
bao gồm truyền mạng, codec hay hiển thị. Chỉ số này không đo độ trễ đầu-cuối. OCR
vẫn dùng CPU, có thể giảm tốc đáng kể; chu kỳ OCR tính theo frame đầu vào được nhận.

Local dùng `http://localhost:8501`. Truy cập từ xa cần HTTPS (Streamlit Cloud đã có).
Mặc định cấu hình STUN. Một số mạng NAT/firewall và Streamlit Cloud có thể cần TURN;
không thể bảo đảm kết nối chỉ bằng STUN. Nếu START chờ mãi, kiểm tra quyền camera,
HTTPS, rồi cấu hình TURN của nhà cung cấp. Copy `.streamlit/secrets.example.toml`
sang `.streamlit/secrets.toml` khi chạy local; trên Streamlit Cloud dán nội dung
vào Secrets. Thay host/username/credential mẫu bằng giá trị thật, không commit secrets.
Docker cần mount secrets.toml riêng nếu sử dụng TURN. App không tự tạo dịch vụ TURN
hay gọi dịch vụ trả phí. Máy hosting chạy inference bằng phần cứng của chính nó;
không dùng GPU laptop của người truy cập.

Tài liệu WebRTC: https://whitphx.github.io/streamlit-webrtc/deployment/

Kiểm chứng local: runtime Streamlit mở được trang camera và chuẩn bị model GPU;
callback xử lý thành công 30 frame lấy từ video thật ở 640x360, imgsz 480, OCR tắt.
Lượt thử callback đạt khoảng 36,7 frame/giây, chưa gồm truyền WebRTC, encode hoặc
camera thật. Không phải cam kết FPS camera/hosting. Cần người dùng cấp quyền và
thử START trên camera thật để đo trải nghiệm đầu-cuối.

## Hosting

Có thể chạy cùng Docker Compose trên máy chủ Linux/VM hỗ trợ Docker. Copy mã và
model đến máy chủ, sau đó chạy `docker compose -f compose.web.yml up -d --build`.
Model web đã được Git theo dõi nhưng bị loại khỏi Docker build; cấp qua mount
`UI/models:/app/models:ro`. Có thể đổi `MODEL_PATH` nếu dùng đường dẫn khác.

Điểm khởi đầu để thử tải: 2–4 vCPU, RAM 4–8 GB; đây là gợi ý cấu hình, chưa phải
benchmark hay cam kết sức chứa. Tốc độ tùy video, CPU, OCR và số người dùng.
Docker web cài Torch bản CPU nên không gặp lỗi driver CUDA của image desktop.
Máy chủ cần dung lượng tạm cho video và Internet trong lần tải weights OCR đầu tiên.

Khi đưa lên Internet: cấu hình domain/HTTPS qua reverse proxy hỗ trợ WebSocket,
giới hạn request body ít nhất 100 MB, và thêm xác thực nếu chỉ phục vụ nội bộ.
Không tắt bảo vệ XSRF/CORS của Streamlit. Chưa có tác vụ hosting hay public URL
nào được thực hiện. Dịch vụ hosting chỉ phục vụ file tĩnh không chạy được ứng dụng này.

Nếu dùng Streamlit Community Cloud, chọn entrypoint `cloud/streamlit_app.py`
và Python 3.11. Cấu hình dùng `cloud/requirements.txt` thay vì dependencies desktop
ở root. Xem [CLOUD_DEPLOY.md](CLOUD_DEPLOY.md) để triển khai từ GitHub.

## Kiểm thử

```powershell
.\.venv-web\Scripts\python.exe -m pip install pytest
.\.venv-web\Scripts\python.exe -m pytest tests_web -q
```
