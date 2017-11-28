#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
#nematus=$1

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/nematus/data/mosesdecoder

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu0
p=1

dev=/home/zhengzx/nematus/data/zh-en/MT03/ch
ref=/home/zhengzx/nematus/data/zh-en/MT03/en
model_saveto=./model.npz
saveto=mt03.trans

mkdir -p test_trans

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python ../nematus/translate.py \
     -m $model_saveto \
     -i $dev \
     -o test_trans/$saveto \
     -k 12 -n -p $p

bash ./postprocess-dev.sh < test_trans/$saveto > test_trans/$saveto
# for zh-en convention, use case-insensitive bleu
cp test_trans/$saveto test_trans/$saveto

## get BLEU
BLEU=`perl ../utils/multi-bleu.perl $ref < test_trans/$saveto | cut -f 3 -d ' ' | cut -f 1 -d ','`
echo "$saveto BLEU = $BLEU" >> test_trans/test_bleu_scores
perl ../utils/multi-bleu.perl $ref < test_trans/$saveto >> test_trans/test_bleu_scores
