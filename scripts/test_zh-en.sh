#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
#nematus=$1

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/nematus/data/mosesdecoder

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu3
p=3
test_name=MT05
dev=/home/zhengzx/nematus/data/zh-en/$test_name/ch
ref=/home/zhengzx/nematus/data/zh-en/$test_name/en
model_saveto=../models/model.iter79000.npz
saveto=${test_name}.trans.iter79000

mkdir -p test_trans

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python ../nematus/translate.py \
     -m $model_saveto \
     -i $dev \
     -o test_trans/$saveto \
     -k 12 -n -p $p

bash ./postprocess-dev.sh < test_trans/$saveto > test_trans/$saveto.post
# for zh-en convention, use case-insensitive bleu
cp test_trans/$saveto test_trans/$saveto.post

## get BLEU
BLEU=`perl ../utils/multi-bleu.perl $ref < test_trans/$saveto.post | cut -f 3 -d ' ' | cut -f 1 -d ','`
echo "$saveto BLEU = $BLEU" >> test_trans/test_bleu_scores
perl ../utils/multi-bleu.perl $ref < test_trans/$saveto.post >> test_trans/test_bleu_scores
