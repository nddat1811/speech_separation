#!/bin/sh

# Usage:
#   bash train_new.sh <data_type> <network> <dataset_name>
# Example:
#   bash train_new.sh clean MossFormer2_SS_8K Libri2mix_8min
#
#   <data_type>: clean | noise
#   <network>:   MossFormer2_SS_16K | MossFormer2_SS_8K
#   <dataset_name>: tên dataset (ví dụ: Libri2mix_8min)

gpu_id=0                                # visible GPUs
n_gpu=1                                 # number of GPUs used for training

# Nhận 3 tham số từ command line
data_type=$1        # clean hoặc noise
network=$2          # MossFormer2_SS_16K hoặc MossFormer2_SS_8K
dataset_name=$3     # ví dụ: Libri2mix
# Ví dụ !bash train.sh clean MossFormer2_SS_16K Libri2mix
checkpoint_dir=checkpoints/${dataset_name}/${network}_${data_type}  # nơi lưu checkpoint
config_pth=config/train/${network}.yaml
train_from_last_checkpoint=1
init_checkpoint_path=../../speech_separation_test/checkpoints/${dataset_name}/${network}_${data_type}/last_best_checkpoint.pt

print_freq=300
#số này là lưu ở giữa checkpoint  --> để 1 số lớn
checkpoint_save_freq=13900000

if [ ! -d "${checkpoint_dir}" ]; then
  mkdir -p ${checkpoint_dir}
fi

cp $config_pth $checkpoint_dir/config.yaml

export PYTHONWARNINGS="ignore"
CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=$n_gpu \
--master_port=$(date '+88%S') \
train.py \
--config ${config_pth} \
--checkpoint_dir ${checkpoint_dir} \
--train_from_last_checkpoint ${train_from_last_checkpoint} \
--init_checkpoint_path ${init_checkpoint_path} \
--print_freq ${print_freq} \
--checkpoint_save_freq ${checkpoint_save_freq} 

