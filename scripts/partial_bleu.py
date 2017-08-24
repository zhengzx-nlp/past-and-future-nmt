import os
import numpy
import re


def bleu(cand, ref):
    pattern = re.compile(r'(\d*\.\d*)')
    cmd = 'perl multi-bleu.perl %s < %s' % (ref, cand)
    ret = os.popen(cmd).read()
    # print ret
    bleu_score = re.search(pattern, ret).group()
    bleu_score = float(bleu_score)
    return bleu_score


def calc_bleu(idx, b, m):
    ftmps = [open('./tmp%d' % i, 'w') for i in range(4)]
    for i, f in enumerate(ftmps):
        print >> f, frefs[i][idx]
    [f.close() for f in ftmps]
    with open('./b', 'w') as fb, open('./m', 'w') as fm:
        print >> fb, b
        print >> fm, m

    return bleu('b'), bleu('m')


def main(models, k):
    if not os.path.exists('length'):
        os.mkdir('length')
    collection = dict()
    for ii in range(2, 7):
        fch = open('/home/zhengzx/nematus/data/zh-en/MT0%d/ch' % ii)
        fen = [open('/home/zhengzx/nematus/data/zh-en/MT0%d/en%d'%(ii, i)).readlines() for i in range(4)]
        # fds = [open(m).readlines() for m in ['../dl4mt-baseline/mt0%d.trans.iter280000'%ii,
        #                                      './mt0%d.trans.iter270000' % ii,
        #                                      '../couplin_net.trirnn5-addloss/mt0%d.trans.iter169500'%ii,]]
        fds = [open(m).readlines() for m in ['../test_trans/MT0%d.trans.iter130000'%ii,]]

        for idx, ch in enumerate(fch):
            # print len()
            catg = len(ch.strip().split()) / 10 * 10
            # if catg > 70:
            #     catg = 70
            if catg >=0:
                catg = 0
            if not collection.get(catg):
                collection[catg] = []
            # l = len(s[idx])
            # start = int(l*k)
            # end = start + l/2
            collection[catg].append(
                [ch] +
                [ss[idx] for ss in fen]
            )
            for s in fds:
                line = s[idx].strip().split()
                l = len(line)
                start = int(l*k)
                end = int(start + l/2)
                collection[catg][-1] += [' '.join(line[start:end])]

                # collection[catg].append([ch] +
                #                     [s[idx] for s in fen] +
                #                     [' '.join(s[idx].strip().split()[int(len(s[idx])*k):int(len(s[idx])*k) + len(s[idx])/2]) for s in fds])
            # collection[catg].append([ch] + [s[idx] for s in fen] + [s[idx] for s in fds])

    for kk, vv in collection.iteritems():

        if not os.path.exists('length/%d'%kk):
            os.mkdir('length/%d'%kk)
        fws = [open("length/%d/%s" % (kk, m), 'w') for m in (['ch', 'en0', 'en1', 'en2', 'en3'] + [1])]
        for line in vv:
            for idx, fw in enumerate(fws):
                print >> fw, line[idx].strip()

    from collections import OrderedDict
    ret = OrderedDict()
    for l in sorted([i for i in collection.iterkeys()]):
        print l, len(collection[l])
        ret[l] = []
        for m in [1]:
            cand = 'length/%s/%s' % (l, m)
            ref = 'length/%s/en' % l
            # print cand, ref
            ret[l].append(bleu(cand, ref))
    import numpy as np
    ret = np.array([vv for kk, vv in ret.iteritems()])
    # print ret.mean(0)
    print ret

    np.save('all_bleu_by_length', ret)


if __name__ == '__main__':
    import sys

    # refs_dir = '../data/MT03/'
    # frefs = [open(os.path.join(refs_dir, r)).readlines() for r in os.listdir(refs_dir)]
    for k in [0, 0.25, 0.5]:
        main('1', k)
