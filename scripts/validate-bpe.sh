#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=$1

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/nematus/data/mosesdecoder

# theano device, in case you do not want to compute on gpu, change it to cpu
device=$2
dev=$3
ref=$4
model_saveto=$5
saveto=$6
mkdir -p tmp_trans

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/translate.py \
     -m $model_saveto \
     -i $dev \
     -o tmp_trans/$saveto \
     -k 12 -n -p 2


bash ./postprocess-dev.sh < tmp_trans/$saveto > tmp_trans/$saveto.post

## get BLEU
BEST=`cat tmp_trans/best_bleu || echo 0`
perl ../utils/multi-bleu-detok.perl $ref < tmp_trans/$saveto.post >> tmp_trans/bleu_scores
BLEU=`perl ../utils/multi-bleu-detok.perl $ref < tmp_trans/$saveto.post | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "$saveto BLEU = $BLEU"

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > tmp_trans/best_bleu
  cp $model_saveto tmp_trans/best_bleu.npz
fi
