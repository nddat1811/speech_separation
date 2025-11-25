## Nguồn tham khảo

- [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio) - AI-Powered Speech Processing Toolkit với các mô hình pre-trained state-of-the-art cho speech enhancement, separation, và target speaker extraction.
- [LibriMix](https://github.com/JorisCos/LibriMix) - Open source dataset cho speech separation, được tạo từ LibriSpeech và WHAM noise. Dataset này được sử dụng để huấn luyện mô hình MossFormer2 trong dự án này.

## Cấu trúc thư mục 
```
project-root/
├─ app.py                 # Flask entrypoint + route/controller
├─ run_cpu_only.sh        # script auto setup & run
├─ checkpoints/           # chứa mô hình đã huấn luyện (trọng số MossFormer2)
├─ config/                # file cấu hình huấn luyện/inference
├─ data/                  # metadata, file scp
├─ dataloader/            # PyTorch dataset + loader
├─ dataset/               # chứa dữ liệu gốc để train
├─ losses/                # hàm loss (SDR, SI-SNR,…)
├─ models/                # kiến trúc mô hình MossFormer2
│  └─ mossformer2/        # implementation chi tiết
│     ├─ mossformer2.py   # MossFormer2_SS, Encoder, Decoder, MossFormer
│     ├─ mossformer2_block.py  # MossformerBlock, Gated_FSMN, FLASH attention
│     ├─ conv_module.py   # ConvModule, DepthwiseConv1d, PointwiseConv1d
│     ├─ conv_stft.py     # ConvSTFT, ConviSTFT (STFT/inverse STFT)
│     ├─ fsmn.py          # Feedforward Sequential Memory Network
│     └─ layer_norm.py    # các loại layer normalization
├─ outputs/
│  ├─ clean|finetune|noise/
│  │  ├─ input/           # file gốc người dùng theo từng chế độ
│  │  └─ output/          # stem đã tách: <original>_<mode>_s1/s2.wav
│  └─ MossFormer2_SS_8K/  # demo sample
├─ templates/index.html   # giao diện web duy nhất
├─ utils/                 # helper xử lý âm thanh, logging, checkpoint
├─ scripts (*.sh/*.py)    # generate_librimix, inference, train, solver...
└─ note_script_generate... # ghi chú tạo dataset
```

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


## Yêu cầu hệ thống
- Python 3.8 trở lên
- RAM tối thiểu 4 GB (8 GB+ sẽ ổn định hơn)
- CPU bất kỳ, không cần GPU
- Trống khoảng 2 GB cho model + dependency

## Khắc phục sự cố
- **Model not found**: kiểm tra `checkpoints/Libri2Mix_min_adam/MossFormer2_SS_8K_clean/` phải có `last_best_checkpoint` hoặc `last_checkpoint`.
- **CUDA out of memory**: app chạy trên CPU, nếu vẫn báo lỗi hãy kiểm tra lại RAM.
- **Lỗi import**: chạy `python test_deps.py` để xác nhận dependency, sau đó cài lại gói thiếu.

## Tạo Dataset LibriMix

Dự án sử dụng [LibriMix](https://github.com/JorisCos/LibriMix) để tạo dataset cho training. LibriMix là dataset open source cho speech separation, được tạo từ LibriSpeech (clean subset) và WHAM noise.

### Cài đặt SoX

**Windows:**
```bash
conda install -c groakat sox
```

**Linux/Mac:**
```bash
conda install -c conda-forge sox
```

### Tạo LibriMix

1. Clone repository LibriMix:
```bash
git clone https://github.com/JorisCos/LibriMix
cd LibriMix
```

2. Chạy script tạo dataset:
```bash
./generate_librimix.sh storage_dir
```

Trong đó `storage_dir` là thư mục lưu dataset. Bạn có thể chỉnh sửa script `generate_librimix.sh` để tùy chỉnh:
- `n_src`: số lượng speakers (2 hoặc 3)
- Sample rate: 16kHz hoặc 8kHz
- Mode: `min` (mixture kết thúc khi source ngắn nhất kết thúc) hoặc `max` (mixture kết thúc khi source dài nhất kết thúc)
- Mixture types: `mix_clean` (chỉ utterances), `mix_both` (utterances + noise), `mix_single` (1 utterance + noise)

**Lưu ý về dung lượng:**
- Libri2Mix: ~430GB
- Libri3Mix: ~332GB
- Cần thêm ~30GB cho LibriSpeech và ~50GB cho wham_noise_augmented trong quá trình tạo

Sau khi tạo xong, sử dụng các script trong thư mục `scripts/` của dự án (như `generate_librimix.sh`, `generate_scp.py`) để xử lý và chuẩn bị dữ liệu cho training.

