# past-and-future-nmt
Modeling Past and Future for Neural Machine Translation

-----
If you use the code, which is implemeted on the popular codebase <a href="https://github.com/EdinburghNLP/nematus">Nematus</a>, please cite our paper:

<pre><code>@article{Zheng:2018:TACL,
  author    = {Zheng, Zaixiang and Zhou, Hao and Huang, Shujian and Mou, Lili and Dai Xinyu and Chen, Jiajun and Tu, Zhaopeng},
  title     = {Modeling Past and Future for Neural Machine Translation},
  journal   = {Transactions of the Association for Computational Linguistics},
  year      = {2018},
}
</code></pre>

## Requirements
1. python2.7
2. Theano >= 0.9
3. mosesdecoder (only scripts needed)
4. cuda >= 8.0

## Usage
1. Data preparation
2. Pretraining a RNNSearch model on Nematus
3. Training
4. Testing

### Data preparation
- **Data Cleaning**: filter out bad characters, unaligned sentence pairs
- **Tokenization**: [tokenizer.pl from MosesDecoder](https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer)
- **Lowercase**: if needed
- **Subword**: use [Byte-Pair Encoding](https://github.com/rsennrich/subword-nmt)

### Pretrain a RNNSearch model on Nematus
- Use [Nematus](https://github.com/EdinburghNLP/nematus) to train a baseline model
- Or you can download the pretrained models [here(not uploaded yet)]()

### Training
Run `./scripts/train.sh` (edit it if needed) for training. See `./scripts/train.sh` for details.

#### model-specific options 
| option                 | description (value)|
| ----------------------- | --- |
| --use_past_layer       | (bool, default: False) whether to apply past layer|
| --use_future_layer     | (bool, default: False) whether to apply future layer|
| --future_layer_type    | (str, default: "gru_inside") type of RNN cell for future layer, <br> only support \["gru", "gru_outside", "gru_inside"\]|
| --use_subtractive_loss | (bool, default: False) whether to use subtractive loss on past or(and) future layer|
| --use_testing_loss     | (bool, default: False) whether to use subtractive loss during testing phase|

### Testing
Run `./scripts/test.sh` (edit it if needed) for testing. See `./scripts/test.sh` for details.
