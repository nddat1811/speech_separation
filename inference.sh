#!/bin/bash 
# uses MossFormer2_SS_16K model for separating 16kHz speech waveforms
network=MossFormer2_SS_16K
config=config/inference/${network}.yaml

CUDA_VISIBLE_DEVICES=4 python3 -u inference.py --config $config
