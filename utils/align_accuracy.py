'''
Translates a source file using a translation model.
'''
import argparse
import logging
import os
import sys

import ipdb
import numpy
import theano

sys.path.insert(0, '..')
from nematus.nmt import (load_params, build_model, load_dict, load_config, init_theano_params, profile
                         )

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

trng = RandomStreams(1234)
use_noise = theano.shared(numpy.float32(0.))

def main(model, dictionary, dictionary_target, source_file, target_file, gold_align, saveto, k=5, pkl_file=None,
         normalize=False, output_attention=False):
    # load model model_options
    # if pkl_file is None:
    #     pkl_file = model + '.pkl'
    # with open(pkl_file, 'rb') as f:
    #     options = pkl.load(f)

    options = load_config(model)
    options['factor'] = 1
    # load source dictionary and invert
    word_dict = load_dict(dictionary)
    word_idict = dict() # id2word
    for kk, vv in word_dict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    # # load target dictionary and invert
    word_dict_trg = load_dict(dictionary_target)
    word_idict_trg = dict()
    for kk, vv in word_dict_trg.iteritems():
        word_idict_trg[vv] = kk
    word_idict_trg[0] = '<eos>'
    word_idict_trg[1] = 'UNK'

    # utility function
    def _seqs2words(caps):
        capsw = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(word_idict_trg[w])
            capsw.append(' '.join(ww))
        return capsw

    def _send_jobs(fx_name, fy_name):
        retval = []
        retval_ori = []
        with open(fx_name, 'r') as fx, open(fy_name, 'r') as fy:
            for idx, (line_x, line_y) in enumerate(zip(fx, fy)):
                words = line_x.strip().split()
                x = map(lambda w: word_dict[w] if w in word_dict else 1, words)
                x = map(lambda ii: ii if ii < options['n_words_src'] else 1, x)
                # x += [0]

                words = line_y.strip().split()
                y = map(lambda w: word_dict_trg[w] if w in word_dict_trg else 1, words)
                y = map(lambda ii: ii if ii < options['n_words'] else 1, y)
                # y += [0]

                retval_ori.append((line_x.strip(), line_y.strip()))
                retval.append((x, y))

        return retval, retval_ori


    # load params
    param_list = numpy.load(model).files
    param_list = dict.fromkeys(
        [key for key in param_list if not key.startswith('adam_')], 0)
    params = load_params(model, param_list)
    tparams = init_theano_params(params)

    # build model
    trng, use_noise, \
    x, x_mask, y, y_mask, \
    opt_ret, \
    cost = \
        build_model(tparams, options)

    inps = [x, x_mask, y, y_mask]

    # compile f_align
    logging.info('Building f_align...')
    # f_align= theano.function(inps, opt_ret['dec_alphas'], profile=profile)
    f_align= theano.function(inps, cost, profile=profile)
    logging.info('Done')

    print 'Processing ', source_file, '...'
    sys.stdout.flush()

    n_samples, n_samples_src = _send_jobs(source_file, target_file)

    atts = []
    idx = 0

    def _prepare_data(x, y):
        # print len(x), len(y)
        x = numpy.array([x]).T
        y = numpy.array([y]).T
        return x[None, :, :], numpy.ones_like(x, dtype='float32'), y, numpy.ones_like(y, dtype='float32')

    start_time = datetime.datetime.now()
    words = 0.
    for (x, y) in n_samples:
        # print x
        x, x_mask, y, y_mask = _prepare_data(x, y)

        att = f_align(x, x_mask, y, y_mask) # (len_y, nsample=1, len_x)
        # att = numpy.squeeze(att, 1)
        # atts.append(att.T)
        # ipdb.set_trace()
        # print idx
        # idx += 1
        # if idx % 100 == 0:
        #     print idx,
            # break
    last =  datetime.datetime.now() - start_time
    print last.total_seconds(), len(n_samples) / last.total_seconds()

    def _force_decode(x, y):
        # sample given an input sequence and obtain scores
        att = f_force_decode(numpy.array(x)[:, None], numpy.array(y)[:, None])
        _output_attention(0, att[0].squeeze(1).T)

    def _output_attention(sent_idx, att):
        dirname = saveto + '.attention'
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        with open(dirname + '/' + str(sent_idx), 'w') as fp:
            fp.write("%d %d\n" % (att.shape[0], att.shape[1]))
            for row in att:
                # fp.write(str(row.argmax()) + " " + ' '.join([str(x) for x in row]) + '\n')
                fp.write('['+','.join([str(x) for x in row])+'],')
                # fp.write(att)

    if output_attention:
        with open(saveto + '.att', 'w') as f:
            for idx, ((x, y), att) in enumerate(zip(n_samples_src, atts)):
                print >> f, ' '.join(["{}:{}".format(idx + 1, hehe.argmax() + 1) for idx, hehe in enumerate(att)])
                # print >> f
    with open(saveto + '.att') as f_att, open(gold_align) as f_gold:
        AER = []
        count_S, count_P, len_A, len_S = 0., 0., 0., 0.
        for idx, (cand, gold) in enumerate(zip(f_att, f_gold)):
            aer, count_s, count_p, len_a, len_s = calc_aer(cand, gold)
            AER.append(aer)
            count_S += count_s
            count_P += count_p
            len_A += len_a
            len_S += len_s
        ave_AER = numpy.average(AER)
        overall_AER = 1 - (count_S + count_P) / (len_A + len_S)
        print 'ave_AER ', ave_AER
        print 'overall_AER ', overall_AER
        ipdb.set_trace()
    print 'Done'


def calc_aer(cand, gold):
    def _preprocess(cand, gold):
        A = cand.strip().split()
        S = []
        P = []
        for item in gold.strip().split():
            if item[-1] == '1':
                S.append(item[:-2])
                P.append(item[:-2])
            else:
                P.append(item[:-2])
        return A, S, P

    A, S, P = _preprocess(cand, gold)
    count_S, count_P = 0., 0.
    for a in A:
        if a in S:
            count_S += 1
        if a in P:
            count_P += 1

    aer = 1 - (count_S + count_P) / (len(A) + len(S))

    return aer, count_S, count_P, len(A), len(S)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # k: beam size
    parser.add_argument('-k', type=int, default=5)
    # n: if normalize
    parser.add_argument('-n', action="store_true", default=True)
    # if output attention
    parser.add_argument('-a', action="store_true", default=False)
    # pkl model
    parser.add_argument('-m', type=str, default=None)
    # model.npz
    parser.add_argument('model', type=str)
    # source side dictionary
    parser.add_argument('dictionary', type=str)
    # target side dictionary
    parser.add_argument('dictionary_target', type=str)
    # source file
    parser.add_argument('source', type=str)
    # target file
    parser.add_argument('target', type=str)

    parser.add_argument('gold_align', type=str)
    # translation file
    parser.add_argument('saveto', type=str)

    args = parser.parse_args()

    import datetime

    start_time = datetime.datetime.now()

    main(args.model, args.dictionary, args.dictionary_target, args.source, args.target, args.gold_align,
         args.saveto, k=args.k, pkl_file=args.m, normalize=args.n,
         output_attention=args.a)

    print 'Elapsed Time: %s' % str(datetime.datetime.now() - start_time)
