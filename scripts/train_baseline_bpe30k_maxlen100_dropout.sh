#!/bin/bash

# warning: this test is useful to check if training fails, and what speed you can achieve
# the toy datasets are too small to obtain useful translation results,
# and hyperparameters are chosen for speed, not for quality.
# For a setup that preprocesses and trains a larger data set,
# check https://github.com/rsennrich/wmt16-scripts/tree/master/sample

model_dir=../models100
device=cuda1
src_lng=zh
tgt_lng=en
train_prefix=/home/user_data/zhengzx/mt/data/nmtdata/zh2en_134w/bpe30k/
dev_prefix=/home/user_data/zhengzx/mt/data/nmtdata/zh2en_134w/bpe30k/MT03.bpe
ori_dev_prefix=/home/user_data/zhengzx/mt/data/nmtdata/zh2en_134w/testsets/mt03


mkdir -p ${model_dir}
export THEANO_FLAGS=device=$device,floatX=float32,gpuarray.preallocate=0.7

python -u ../nematus/nmt.py \
  --model ${model_dir}/model.npz \
  --datasets ${train_prefix}/corpus.bpe.${src_lng} ${train_prefix}/corpus.bpe.${tgt_lng}\
  --dictionaries ${train_prefix}/corpus.bpe.${src_lng}.json ${train_prefix}/corpus.bpe.${tgt_lng}.json \
  --dim_word 512 \
  --dim 1024 \
  --n_words_src 30000 \
  --n_words 30000 \
  --maxlen 100 \
  --optimizer adam \
  --anneal_restarts 1000 \
  --lrate 0.0002 \
  --batch_size 80 \
  --dispFreq 10 \
  --finish_after 1000000 \
  --reload \
  --patience 30 \
  --saveFreq 5000 \
  --valid_datasets ${dev_prefix}.zh ${dev_prefix}.${tgt_lng}0 \
  --validFreq 100 \
  --valid_batch_size 50 \
  --external_validation_script "bash validate.sh ../nematus ${device} ${dev_prefix}.zh ${ori_dev_prefix}.ref" \
  --start_external_valid 79 \
  --external_validFreq 2500 \
  --use_dropout
