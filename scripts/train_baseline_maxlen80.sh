#!/bin/bash

# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

model_dir=../models
device=cuda2
src_lng=cn
tgt_lng=en
train_prefix=/home/zhengzx/nematus/data/zh-en
dev_prefix=/home/zhengzx/nematus/data/zh-en/MT03

mkdir -p ${model_dir}
export THEANO_FLAGS=device=$device,floatX=float32,gpuarray.preallocate=0.7

python -u ../nematus/nmt.py \
  --model ${model_dir}/model.npz \
  --prior_model ${train_prefix}/model-baseline-maxlen80.npz \
  --datasets ${train_prefix}/${src_lng}.txt.shuf ${train_prefix}/${tgt_lng}.txt.shuf \
  --dictionaries ${train_prefix}/${src_lng}.txt.shuf.json ${train_prefix}/${tgt_lng}.txt.shuf.json \
  --dim_word 512 \
  --dim 1024 \
  --n_words_src 30000 \
  --n_words 30000 \
  --maxlen 80 \
  --optimizer adam \
  --anneal_restarts 10 \
  --lrate 0.0002 \
  --batch_size 80 \
  --dispFreq 10 \
  --finish_after 1000000 \
  --reload \
  --patience 30 \
  --saveFreq 5000 \
  --valid_datasets ${dev_prefix}/ch ${dev_prefix}/${tgt_lng}0 \
  --validFreq 100 \
  --valid_batch_size 50 \
  --external_validation_script "bash validate.sh ../nematus ${device} ${dev_prefix}/ch ${dev_prefix}/en" \
  --start_external_valid 68 \
  --external_validFreq 2500
