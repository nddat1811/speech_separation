# ClearerVoice-Studio: Train Speech Separation Models

## 1. Introduction

This repository provides a flexible training or finetune scripts for speech separation models. Currently, it supports both 8kHz and 16kHz sampling rates:

|model name| sampling rate | Paper Link|
|----------|---------------|------------|
|MossFormer2_SS_8K  |8000| MossFormer2 ([Paper](https://arxiv.org/abs/2312.11825), ICASSP 2024)|
|MossFormer2_SS_16K  |16000| MossFormer2 ([Paper](https://arxiv.org/abs/2312.11825), ICASSP 2024)|

MossFormer2 has achieved state-of-the-art speech sesparation performance upon the paper published in ICASSP 2024. It is a hybrid model by integrating a recurrent module into
our previous [MossFormer](https://arxiv.org/abs/2302.11824) framework. MossFormer2 is capable to model not only long-range and coarse-scale dependencies but also fine-scale recurrent patterns. For efficient self-attention across the extensive sequence, MossFormer2 adopts the joint local-global self-attention strategy as proposed for MossFormer. MossFormer2 introduces a dedicated recurrent module to model intricate temporal dependencies within speech signals.

![github_fig1](https://github.com/alibabasglab/MossFormer2/assets/62317780/e69fb5df-4d7f-4572-88e6-8c393dd8e99d)


Instead of applying the recurrent neural networks (RNNs) that use traditional recurrent connections, we present a recurrent module based on a feedforward sequential memory network (FSMN), which is considered "RNN-free" recurrent network due to the ability to capture recurrent patterns without using recurrent connections. Our recurrent module mainly comprises an enhanced dilated FSMN block by using gated convolutional units (GCU) and dense connections. In addition, a bottleneck layer and an output layer are also added for controlling information flow. The recurrent module relies on linear projections and convolutions for seamless, parallel processing of the entire sequence. 

![github_fig2](https://github.com/alibabasglab/MossFormer2/assets/62317780/7273174d-01aa-4cc5-9a67-1fa2e8f7ac2e)


MossFormer2 demonstrates remarkable performance in WSJ0-2/3mix, Libri2Mix, and WHAM!/WHAMR! benchmarks. Please refer to our [Paper](https://arxiv.org/abs/2312.11825) or the individual models using the standalone script ([link](https://github.com/alibabasglab/MossFormer2/tree/main/MossFormer2_standalone)). 

We will provide performance comparisons of our released models with the publically available models in [ClearVoice](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice) page.

## 2. Usage

### Step-by-Step Guide

If you haven't created a Conda environment for ClearerVoice-Studio yet, follow steps 1 and 2. Otherwise, skip directly to step 3.

1. **Clone the Repository**

``` sh
git clone https://github.com/modelscope/ClearerVoice-Studio.git
```

2. **Create Conda Environment**

``` sh
cd ClearerVoice-Studio
conda create -n ClearerVoice-Studio python=3.8
conda activate ClearerVoice-Studio
pip install -r requirements.txt
```

3. **Prepare Dataset**

a. Use a pre-prepared toy [MiniLibriMix dataset](https://zenodo.org/records/3871592). It contains a train set of 800 mixtures and a validation set of 200 mixtures.

b. Create your own dataset

- WSJ0-2Mix dataset preparation: We assume you have purchased [WSJ0 speech dataset](https://catalog.ldc.upenn.edu/LDC93S6A)
  - Step 1: Download [WHAM! noise dataset](https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip). Go to [this page](http://wham.whisper.ai/) for more information.
  - Step 2: Use the mixture generation scripts in [python format](https://github.com/mpariente/pywsj0-mix) or [matlab format](https://www.merl.com/research/highlights/deep-clustering/) to generate mixture datasets. Use the sampling rate either 8000Hz or 16000Hz.
  - Step 3: Create scp files as formatted in `data/tr_wsj0_2mix_16k.scp` for train, validation, and test.
  - Step 4: Replace the `tr_list` and `cv_list` paths for scp files in `config/train/MossFormer2_SS_16K.yaml`
 
- LibriMix dataset preparation: If you don't have WSJ0 dataset, we suggest you to download [LibriSpeech dataset](https://www.openslr.org/12) (only 'train-clean-360.tar.gz' is required) and use the following steps to create LibriMix dataset.
  - Step 1. Download [WHAM! noise dataset](https://my-bucket-a8b4b49c25c811ee9a7e8bba05fa24c7.s3.amazonaws.com/wham_noise.zip). Go to [this page](http://wham.whisper.ai/) for more information.
  - Step 2. Clone the [repo](https://github.com/JorisCos/LibriMix) and run the main script : [generate_librimix.sh](https://github.com/JorisCos/LibriMix/blob/master/generate_librimix.sh)
    ```sh
    git clone https://github.com/JorisCos/LibriMix
    cd LibriMix 
    ./generate_librimix.sh storage_dir
    ```sh
  - Step 3: Create scp files as formatted in `data/tr_wsj0_2mix_16k.scp` for train, validation, and test.
  - Step 4: Replace the `tr_list` and `cv_list` paths for scp files in `config/train/MossFormer2_SS_16K.yaml`

4. **Start Training**

``` sh
bash train.sh
```

You may need to set the correct network in `train.sh` and choose either a fresh training or a finetune process using:
```
network=MossFormer2_SS_16K              #Train MossFormer2_SS_16K model
train_from_last_checkpoint=1            #Set 1 to start training from the last checkpoint if exists, 
init_checkpoint_path=./                 #Path to your initial model if starting fine-tuning; otherwise, set it to 'None'
```

