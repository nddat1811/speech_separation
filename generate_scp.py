import os, glob
from pathlib import Path

# ---- cấu hình ----
ROOT = "/home/speech_separation/MiniLibriMix"   # thư mục dataset đã giải nén
USE_POSIX = True   # giữ True để luôn ra đường dẫn dạng "/" (Linux/Colab)

# Tạo tất cả splits trong một lần chạy
splits = ["train", "val", "test"]

# ---- thư mục output ----
out_dir = Path("/home/speech_separation/scpfile")  # <--- xuất ra đây
out_dir.mkdir(parents=True, exist_ok=True)         # tạo nếu chưa có

# ---- hàm tiện ích ----
def list_wavs(d: Path):
    return sorted(str(p) for p in d.glob("*.wav"))

def to_map(files):
    return {Path(p).name: p for p in files}

def norm(p: str) -> str:
    return Path(p).as_posix() if USE_POSIX else p

# ---- xử lý tất cả splits ----
for split in splits:
    out_scp = out_dir / f"{split}_MiniLibri_clean.scp"

    # ---- thư mục input ----
    base = Path(ROOT) / split
    mix_dir = base / "mix_clean"   # nếu muốn mix_both thì đổi ở đây
    s1_dir  = base / "s1"
    s2_dir  = base / "s2"

    mix_files = list_wavs(mix_dir)
    s1_files  = list_wavs(s1_dir)
    s2_files  = list_wavs(s2_dir)

    mix_map = to_map(mix_files)
    s1_map  = to_map(s1_files)
    s2_map  = to_map(s2_files)

    # lấy id chung giữa 3 thư mục
    common = sorted(set(mix_map) & set(s1_map) & set(s2_map))

    # báo file bị thiếu
    miss_in_s1 = sorted(set(mix_map) - set(s1_map))
    miss_in_s2 = sorted(set(mix_map) - set(s2_map))
    if miss_in_s1:
        print(f"⚠️ [{split}] Thiếu {len(miss_in_s1)} file trong s1, ví dụ:", miss_in_s1[:5])
    if miss_in_s2:
        print(f"⚠️ [{split}] Thiếu {len(miss_in_s2)} file trong s2, ví dụ:", miss_in_s2[:5])

    # ghi ra file scp
    with open(out_scp, "w", encoding="utf-8") as f:
        for uid in common:
            f.write(f"{norm(mix_map[uid])} {norm(s1_map[uid])} {norm(s2_map[uid])}\n")

    print(f"✅ Đã tạo {out_scp} với {len(common)} dòng")
    print(f"   [{split}] mix:{len(mix_map)}  s1:{len(s1_map)}  s2:{len(s2_map)}")
