#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$1

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/nematus/data/mosesdecoder

# theano device, in case you do not want to compute on gpu, change it to cpu
device=$2

#model prefix
#prefix=model/model.npz
# prefix=$3
# device=gpu2

dev=$3
ref=$4
model_saveto=$5
saveto=$6
# cp model-BPE64k/model.npz.json $model_saveto.json
mkdir -p test_trans

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/translate.py \
     -m $model_saveto \
     -i $dev \
     -o test_trans/$saveto \
     -k 12 -n -p 1 

## get BLEU
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl -lc $ref < test_trans/$saveto | cut -f 3 -d ' ' | cut -f 1 -d ','`
echo "$saveto BLEU = $BLEU" >> test_trans/test_bleu_scores
$mosesdecoder/scripts/generic/multi-bleu.perl -lc $ref < test_trans/$saveto >> test_trans/test_bleu_scores
