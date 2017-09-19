#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=24:00:00
#PBS -N session2_default
#PBS -A course
#PBS -q ShortQ

export THEANO_FLAGS=device=gpu3,floatX=float32

mkdir -p align_accuracy
model=$1
python -u ./align_accuracy.py  -a\
    -m $model.json \
	$model \
	$HOME/nematus/data/zh-en/cn.txt.shuf.json\
	$HOME/nematus/data/zh-en/en.txt.shuf.json\
	$HOME/dl4mt/data/align/src.txt.utf8 \
	$HOME/dl4mt/data/align/tgt.txt \
	$HOME/dl4mt/data/align/ref.align \
	align_accuracy/align_acc