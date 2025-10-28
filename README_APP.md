# Audio Source Separation Web App

Ứng dụng web để tách giọng nói từ hỗn hợp nhiều người nói sử dụng MossFormer2.

## Cài đặt và chạy

### Cách 1: Sử dụng script tự động (Khuyến nghị)

```bash
chmod +x run_cpu_only.sh
./run_cpu_only.sh
```

Script này sẽ:
1. Tạo virtual environment
2. Cài đặt các dependencies cần thiết
3. Cài đặt PyTorch CPU version
4. Test dependencies
5. Chạy web app

### Cách 2: Cài đặt thủ công

1. Tạo virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate    # Windows
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. Chạy ứng dụng:
```bash
python app.py
```

## Sử dụng

1. Mở trình duyệt và truy cập `http://localhost:5000`
2. Chọn tab "Trải nghiệm"
3. Upload file âm thanh hoặc thu âm từ microphone
4. Chờ quá trình xử lý hoàn tất
5. Nghe và tải xuống các file âm thanh đã tách

## Tính năng

- **Upload file**: Hỗ trợ WAV, MP3, FLAC, M4A (tối đa 16MB)
- **Thu âm**: Ghi âm trực tiếp từ microphone
- **Tách giọng**: Sử dụng MossFormer2 để tách 2 giọng nói
- **Demo**: Có sẵn các file demo để thử nghiệm

## Cấu trúc thư mục

```
outputs/
├── try/
│   ├── input/     # File âm thanh đầu vào
│   └── output/    # File âm thanh đã tách
└── MossFormer2_SS_8K/  # File demo
```

## Yêu cầu hệ thống

- Python 3.8+
- RAM: Tối thiểu 4GB (khuyến nghị 8GB+)
- CPU: Bất kỳ (chạy trên CPU)
- Disk: ~2GB cho model và dependencies

## Xử lý sự cố

### Lỗi "Model not found"
- Kiểm tra xem có file checkpoint trong `checkpoints/Libri2Mix_min_adam/MossFormer2_SS_8K_clean/`
- Đảm bảo có file `last_best_checkpoint` hoặc `last_checkpoint`

### Lỗi "CUDA out of memory"
- Ứng dụng được cấu hình để chạy trên CPU
- Nếu vẫn gặp lỗi, kiểm tra RAM có đủ không

### Lỗi import
- Chạy `python test_deps.py` để kiểm tra dependencies
- Cài đặt lại các package bị thiếu

## API Endpoints

- `GET /`: Trang chủ
- `POST /upload`: Upload và xử lý file âm thanh
- `GET /download/<filename>`: Tải file âm thanh
- `GET /demo_files`: Lấy danh sách file demo
- `GET /health`: Kiểm tra trạng thái ứng dụng
