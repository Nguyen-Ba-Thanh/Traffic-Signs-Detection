# Deploy Traffic Vision lên Streamlit Community Cloud

## Cấu hình đã chuẩn bị

- Entrypoint: **cloud/streamlit_app.py**, không chọn app.py ở root khi deploy.
- Python: **3.11**, chọn trong Advanced settings trước khi Deploy.
- `cloud/requirements.txt` ở cạnh entrypoint: Torch/torchvision CPU đã pin,
  đủ dependencies upload/camera/OCR. Không dùng requirements desktop ở root.
- `packages.txt` ở root cung cấp libgl1/libglib2.0-0 cho OpenCV trên Debian.
- `UI/models/best.pt`: model 32 lớp, 57.045.857 byte, được Git theo dõi.
- Cấu hình giao diện: `.streamlit/config.toml` ở root repo.
- Demo Cloud: mặc định CPU, imgsz 320, OCR tắt, xử lý 10 giây đầu;
  cho phép tối đa 50 MB/video và 30 giây đầu.

Đây là cấu hình triển khai đã chuẩn bị; chỉ có thể xác nhận hạn mức bộ nhớ,
build Linux và camera qua Internet sau khi triển khai trên tài khoản Cloud thật.

## Đưa mã lên repo của bạn

Repo local hiện có remote trỏ tới tác giả gốc. Không push vào remote đó nếu bạn
không sở hữu/quản lý repo. Dùng repo của bạn hoặc fork, giữ LICENSE và thông tin
tác giả của dự án gốc. Không cần đưa `.venv-web*`, videos, output, dataset hoặc
secrets vào GitHub. `.gitignore` đã bỏ qua các mục này.

Các file tối thiểu cần có trong repo deploy:

```text
app.py
traffic_web/
cloud/streamlit_app.py
cloud/requirements.txt
packages.txt
.streamlit/config.toml
UI/models/best.pt
LICENSE
```

Có thể dùng trọn repo nếu bạn muốn giữ cả ứng dụng desktop và training scripts.
Model đã có trong Git hiện tại; quy tắc !UI/models/best.pt cũng cho phép thêm nó
vào repo mới. Đừng chỉ upload mã Python mà bỏ quên model.

## Deploy

1. Đăng nhập https://share.streamlit.io bằng tài khoản được quyền truy cập repo.
2. Create app → chọn repository của bạn và branch chứa mã web mới.
3. Main file path: `cloud/streamlit_app.py`.
4. Advanced settings → Python **3.11**. Secrets có thể để trống khi thử upload.
5. Deploy, đợi cài dependencies và mở URL `https://...streamlit.app`.
6. Thử video 5–10 giây, OCR tắt; kiểm tra MP4 phát được và CSV tải được.
7. Mở Camera trực tiếp → Chuẩn bị camera → START → cấp quyền camera.

Streamlit Cloud tự phục vụ HTTPS. Webcam dùng camera của người truy cập,
inference dùng CPU của Cloud, không dùng GPU laptop bạn.

## Camera và TURN

STUN mặc định có thể chưa đủ ở mạng NAT/firewall hoặc trên Cloud. Khi START chờ
mãi mà camera local chạy được, cấu hình TURN trong App settings → Secrets bằng
mẫu `.streamlit/secrets.example.toml`, thay các giá trị mẫu bằng credentials thật.
Không đưa secrets.toml vào repo. Chưa tạo tài khoản/dịch vụ TURN tự động cho bạn.

## Khi lỗi triển khai

- `No module named ultralytics`: kiểm tra entrypoint có phải cloud/streamlit_app.py;
  build logs phải dùng cloud/requirements.txt.
- Không tìm thấy model: kiểm tra UI/models/best.pt có thật trong branch deploy.
- `libGL.so.1`: kiểm tra packages.txt có ở root repo và build lại môi trường.
- Hết RAM/Killed: thử video 5 giây, OCR tắt, một người dùng. Nếu vẫn vượt giới hạn,
  cần tăng tài nguyên/chuyển hosting; không có cam kết ứng dụng sẽ luôn vừa Cloud miễn phí.
- Camera Media API not available: mở URL HTTPS trực tiếp trong trình duyệt.
- Camera không kết nối: kiểm tra quyền camera và TURN; không phải lỗi model GPU.

## Kiểm tra trước khi chia sẻ link

- Upload → xem video → tải MP4/CSV.
- Camera START/STOP, chuyển về video sau STOP.
- Hai trình duyệt hoặc phiên ẩn danh: không thấy kết quả/log của phiên khác;
  nếu inference đang bận thì thấy thông báo bận thay vì xếp hàng vô hạn.
- Trang Cloud hiển thị CPU, model 32 lớp; đo tốc độ lại trên Cloud.

Tài liệu chính thức:
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- https://whitphx.github.io/streamlit-webrtc/deployment/
