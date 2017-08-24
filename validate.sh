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
mkdir -p tmp_trans58000

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/translate.py \
     -m $model_saveto \
     -i $dev \
     -o tmp_trans58000/$saveto \
     -k 12 -n -p 1


bash ./postprocess-dev.sh < tmp_trans58000/$saveto> tmp_trans58000/$saveto.post


## get BLEU
BEST=`cat tmp_trans58000/best_bleu || echo 0`
$mosesdecoder/scripts/generic/multi-bleu.perl -lc $ref < tmp_trans58000/$saveto.post >> tmp_trans58000/bleu_scores
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl -lc $ref < tmp_trans58000/$saveto.post | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "$saveto BLEU = $BLEU"

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > tmp_trans58000/best_bleu
  cp $model_saveto tmp_trans58000/best_bleu.npz
fi
