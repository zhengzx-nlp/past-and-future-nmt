#!/bin/bash

# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

model_dir=$1
mkdir -p ${model_dir}

device=$2
export THEANO_FLAGS=device=$device,floatX=float32,lib.cnmem=$3

src_lng=cn
tgt_lng=en
train_prefix=/home/zhengzx/nematus/data/zh-en
dev_prefix=/home/zhengzx/nematus/data/zh-en/MT03

python -u ./nmt.py \
  --model ${model_dir}/model.npz \
  --datasets ${train_prefix}/${src_lng}.txt.shuf ${train_prefix}/${tgt_lng}.txt.shuf \
  --dictionaries ${train_prefix}/${src_lng}.txt.shuf.json ${train_prefix}/${tgt_lng}.txt.shuf.json \
  --dim_word 128 \
  --dim 256 \
  --n_words_src 3000 \
  --n_words 3000 \
  --maxlen 80 \
  --optimizer adam \
  --anneal_restarts 5 \
  --lrate 0.0002 \
  --batch_size 5 \
  --dispFreq 10 \
  --finish_after 1000000 \
  --reload \
  --patience 70 \
  --saveFreq 10000 \
  --valid_datasets ${dev_prefix}/ch ${dev_prefix}/${tgt_lng}0 \
  --validFreq 100 \
  --valid_batch_size 50 \
  --external_validation_script "bash validate.sh . ${device} ${dev_prefix}/ch ${dev_prefix}/en"
