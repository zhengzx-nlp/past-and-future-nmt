#/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/nematus/data/mosesdecoder

# suffix of target language files
lng=en

sed 's/\@\@ //g' | \
$mosesdecoder/scripts/recaser/detruecase.perl
$mosesdecoder/scripts/tokenizer/detokenizer.perl -l en
