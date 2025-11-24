# Audio Source Separation Web App

Ứng dụng web tách hai giọng nói từ cùng một bản ghi bằng mô hình MossFormer2, tối ưu cho CPU.

## Cách cài đặt

### 1. Script tự động (khuyên dùng)
```bash
chmod +x run_cpu_only.sh
./run_cpu_only.sh
```
Script sẽ tạo virtual env, cài toàn bộ dependency (bao gồm PyTorch CPU), chạy kiểm tra nhanh và khởi động web app.

### 2. Thiết lập thủ công
1. Tạo và kích hoạt virtual env  
```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```
2. Cài đặt gói cần thiết  
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
3. Chạy server  
```bash
python app.py
```

## Hướng dẫn sử dụng
1. Mở trình duyệt tới `http://localhost:5000`
2. Chọn tab **Trải nghiệm**
3. Tải lên file WAV/MP3/FLAC/M4A (≤16MB) hoặc thu âm trực tiếp
4. Nhấn xử lý, đợi hệ thống tách giọng
5. Nghe thử và tải về từng kênh giọng nói

## Tính năng chính
- Upload file âm thanh nhiều định dạng
- Thu âm từ microphone
- Tách 2 giọng bằng MossFormer2 (8 kHz)
- Bộ demo mẫu sẵn để thử nhanh

## Cấu trúc thư mục đầu ra
```
outputs/
├── try/
│   ├── input/      # file do người dùng tải lên
│   └── output/     # kết quả đã tách
└── MossFormer2_SS_8K/  # bộ demo
```

## Yêu cầu hệ thống
- Python 3.8 trở lên
- RAM tối thiểu 4 GB (8 GB+ sẽ ổn định hơn)
- CPU bất kỳ, không cần GPU
- Trống khoảng 2 GB cho model + dependency

## Khắc phục sự cố
- **Model not found**: kiểm tra `checkpoints/Libri2Mix_min_adam/MossFormer2_SS_8K_clean/` phải có `last_best_checkpoint` hoặc `last_checkpoint`.
- **CUDA out of memory**: app chạy trên CPU, nếu vẫn báo lỗi hãy kiểm tra lại RAM.
- **Lỗi import**: chạy `python test_deps.py` để xác nhận dependency, sau đó cài lại gói thiếu.