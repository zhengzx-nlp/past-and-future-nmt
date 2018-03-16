# coding=utf-8
'''
Layer definitions
'''

import json
import cPickle as pkl
import numpy
from collections import OrderedDict

import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from initializers import *
from util import *
from theano_util import *
from alignment_util import *

#from theano import printing

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'lstm_cond': ('param_init_lstm_cond', 'lstm_cond_layer'),
          'embedding': ('param_init_embedding_layer', 'embedding_layer')
          }


def dropout_constr(options, use_noise, trng, sampling):
    """This constructor takes care of the fact that we want different
    behaviour in training and sampling, and keeps backward compatibility:
    on older versions, activations need to be rescaled at test time;
    on newer veresions, they are rescaled at training time.
    """

    # if dropout is off, or we don't need it because we're sampling, multiply by 1
    # this is also why we make all arguments optional
    def get_layer(shape=None, dropout_probability=0, num=1):
        if num > 1:
            return theano.shared(numpy.array([1.]*num, dtype=floatX))
        else:
            return theano.shared(numpy_floatX(1.))

    if options['use_dropout']:
        # models trained with old dropout need to be rescaled at test time
        if sampling and options['model_version'] < 0.1:
            def get_layer(shape=None, dropout_probability=0, num=1):
                if num > 1:
                    return theano.shared(numpy.array([1-dropout_probability]*num, dtype=floatX))
                else:
                    return theano.shared(numpy_floatX(1-dropout_probability))
        elif not sampling:
            if options['model_version'] < 0.1:
                scaled = False
            else:
                scaled = True
            def get_layer(shape, dropout_probability=0, num=1):
                if num > 1:
                    return shared_dropout_layer((num,) + shape, use_noise, trng, 1-dropout_probability, scaled)
                else:
                    return shared_dropout_layer(shape, use_noise, trng, 1-dropout_probability, scaled)

    return get_layer


def get_layer_param(name):
    param_fn, constr_fn = layers[name]
    return eval(param_fn)

def get_layer_constr(name):
    param_fn, constr_fn = layers[name]
    return eval(constr_fn)

# dropout that will be re-used at different time steps
def shared_dropout_layer(shape, use_noise, trng, value, scaled=True):
    #re-scale dropout at training time, so we don't need to at test time
    if scaled:
        proj = tensor.switch(
            use_noise,
            trng.binomial(shape, p=value, n=1,
                                        dtype=floatX)/value,
            theano.shared(numpy_floatX(1.)))
    else:
        proj = tensor.switch(
            use_noise,
            trng.binomial(shape, p=value, n=1,
                                        dtype=floatX),
            theano.shared(numpy_floatX(value)))
    return proj

# layer normalization
# code from https://github.com/ryankiros/layer-norm
def layer_norm(x, b, s):
    _eps = numpy_floatX(1e-5)
    if x.ndim == 3:
        output = (x - x.mean(2)[:,:,None]) / tensor.sqrt((x.var(2)[:,:,None] + _eps))
        output = s[None, None, :] * output + b[None, None,:]
    else:
        output = (x - x.mean(1)[:,None]) / tensor.sqrt((x.var(1)[:,None] + _eps))
        output = s[None, :] * output + b[None,:]
    return output

def weight_norm(W, s):
    """
    Normalize the columns of a matrix
    """
    _eps = numpy_floatX(1e-5)
    W_norms = tensor.sqrt((W * W).sum(axis=0, keepdims=True) + _eps)
    W_norms_s = W_norms * s # do this first to ensure proper broadcasting
    return W / W_norms_s

# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True, weight_matrix=True, bias=True, followed_by_softmax=False):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    if weight_matrix:
        params[pp(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    if bias:
       params[pp(prefix, 'b')] = numpy.zeros((nout,)).astype(floatX)

    if options['layer_normalisation'] and not followed_by_softmax:
        scale_add = 0.0
        scale_mul = 1.0
        params[pp(prefix,'ln_b')] = scale_add * numpy.ones((1*nout)).astype(floatX)
        params[pp(prefix,'ln_s')] = scale_mul * numpy.ones((1*nout)).astype(floatX)

    if options['weight_normalisation'] and not followed_by_softmax:
        scale_mul = 1.0
        params[pp(prefix,'W_wns')] = scale_mul * numpy.ones((1*nout)).astype(floatX)

    return params


def fflayer(tparams, state_below, options, dropout, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', W=None, b=None, dropout_probability=0, followed_by_softmax=False, **kwargs):
    if W == None:
        W = tparams[pp(prefix, 'W')]
    if b == None:
        b = tparams[pp(prefix, 'b')]

    # for three-dimensional tensors, we assume that first dimension is number of timesteps
    # we want to apply same mask to all timesteps
    if state_below.ndim == 3:
        dropout_shape = (state_below.shape[1], state_below.shape[2])
    else:
        dropout_shape = state_below.shape
    dropout_mask = dropout(dropout_shape, dropout_probability)

    if options['weight_normalisation'] and not followed_by_softmax:
         W = weight_norm(W, tparams[pp(prefix, 'W_wns')])
    preact = tensor.dot(state_below*dropout_mask, W) + b

    if options['layer_normalisation'] and not followed_by_softmax:
        preact = layer_norm(preact, tparams[pp(prefix,'ln_b')], tparams[pp(prefix,'ln_s')])

    return eval(activ)(preact)

# embedding layer
def param_init_embedding_layer(options, params, n_words, dims, factors=None, prefix='', suffix=''):
    if factors == None:
        factors = 1
        dims = [dims]
    for factor in xrange(factors):
        params[prefix+embedding_name(factor)+suffix] = norm_weight(n_words, dims[factor])
    return params

def embedding_layer(tparams, ids, factors=None, prefix='', suffix=''):
    do_reshape = False
    if factors == None:
        if ids.ndim > 1:
            do_reshape = True
            n_timesteps = ids.shape[0]
            n_samples = ids.shape[1]
        emb = tparams[prefix+embedding_name(0)+suffix][ids.flatten()]
    else:
        if ids.ndim > 2:
          do_reshape = True
          n_timesteps = ids.shape[1]
          n_samples = ids.shape[2]
        emb_list = [tparams[prefix+embedding_name(factor)+suffix][ids[factor].flatten()] for factor in xrange(factors)]
        emb = concatenate(emb_list, axis=1)
    if do_reshape:
        emb = emb.reshape((n_timesteps, n_samples, -1))

    return emb

# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None,
                   recurrence_transition_depth=1,
                   **kwargs):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    scale_add = 0.0
    scale_mul = 1.0

    for i in xrange(recurrence_transition_depth):
        suffix = '' if i == 0 else ('_drt_%s' % i)
        # recurrent transformation weights for gates
        params[pp(prefix, 'b'+suffix)] = numpy.zeros((2 * dim,)).astype(floatX)
        U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
        params[pp(prefix, 'U'+suffix)] = U
        # recurrent transformation weights for hidden state proposal
        params[pp(prefix, 'bx'+suffix)] = numpy.zeros((dim,)).astype(floatX)
        Ux = ortho_weight(dim)
        params[pp(prefix, 'Ux'+suffix)] = Ux
        if options['layer_normalisation']:
            params[pp(prefix,'U%s_lnb' % suffix)] = scale_add * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'U%s_lns' % suffix)] = scale_mul * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        if options['weight_normalisation']:
            params[pp(prefix,'U%s_wns' % suffix)] = scale_mul * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_wns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)

        if i == 0:
            # embedding to gates transformation weights, biases
            W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
            params[pp(prefix, 'W'+suffix)] = W
            # embedding to hidden state proposal weights, biases
            Wx = norm_weight(nin, dim)
            params[pp(prefix, 'Wx'+suffix)] = Wx
            if options['layer_normalisation']:
                params[pp(prefix,'W%s_lnb' % suffix)] = scale_add * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'W%s_lns' % suffix)] = scale_mul * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
            if options['weight_normalisation']:
                params[pp(prefix,'W%s_wns' % suffix)] = scale_mul * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_wns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)

    return params


def gru_layer(tparams, state_below, options, dropout, prefix='gru',
              mask=None, one_step=False,
              init_state=None,
              dropout_probability_below=0,
              dropout_probability_rec=0,
              recurrence_transition_depth=1,
              truncate_gradient=-1,
              profile=False,
              **kwargs):

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    dim = tparams[pp(prefix, 'Ux')].shape[1]

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']:
            return weight_norm(param, tparams[param_name+'_wns'])
        else:
            return param

    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim))

    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    below_dropout = dropout((n_samples, dim_below), dropout_probability_below, num=2)
    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=2*(recurrence_transition_depth))

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_list, state_belowx_list = [], []

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*below_dropout[0], wn(pp(prefix, 'W'))) + tparams[pp(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*below_dropout[1], wn(pp(prefix, 'Wx'))) + tparams[pp(prefix, 'bx')]
    if options['layer_normalisation']:
        state_below_ = layer_norm(state_below_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
        state_belowx = layer_norm(state_belowx, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])
    state_below_list.append(state_below_)
    state_belowx_list.append(state_belowx)

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(*args):
        n_ins = 1
        m_ = args[0]
        x_list = args[1:1+n_ins]
        xx_list = args[1+n_ins:1+2*n_ins]
        h_, rec_dropout = args[-2], args[-1]

        h_prev = h_
        for i in xrange(recurrence_transition_depth):
            suffix = '' if i == 0 else ('_drt_%s' % i)
            if i == 0:
                x_cur = x_list[i]
                xx_cur = xx_list[i]
            else:
                x_cur = tparams[pp(prefix, 'b'+suffix)]
                xx_cur = tparams[pp(prefix, 'bx'+suffix)]

            preact = tensor.dot(h_prev*rec_dropout[0+2*i], wn(pp(prefix, 'U'+suffix)))
            if options['layer_normalisation']:
                preact = layer_norm(preact, tparams[pp(prefix, 'U%s_lnb' % suffix)], tparams[pp(prefix, 'U%s_lns' % suffix)])
            preact += x_cur

            # reset and update gates
            r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

            # compute the hidden state proposal
            preactx = tensor.dot(h_prev*rec_dropout[1+2*i], wn(pp(prefix, 'Ux'+suffix)))
            if options['layer_normalisation']:
                preactx = layer_norm(preactx, tparams[pp(prefix, 'Ux%s_lnb' % suffix)], tparams[pp(prefix, 'Ux%s_lns' % suffix)])
            preactx = preactx * r
            preactx = preactx + xx_cur

            # hidden state proposal
            h = tensor.tanh(preactx)

            # leaky integrate and obtain next hidden state
            h = u * h_prev + (1. - u) * h
            h = m_[:, None] * h + (1. - m_)[:, None] * h_prev
            h_prev = h

        return h

    # prepare scan arguments
    seqs = [mask] + state_below_list + state_belowx_list
    _step = _step_slice
    shared_vars = [rec_dropout]

    if one_step:
        rval = _step(*(seqs + [init_state] + shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_state,
                                non_sequences=shared_vars,
                                name=pp(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=truncate_gradient,
                                profile=profile,
                                strict=False)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options, params, prefix='gru_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None,
                        recurrence_transition_depth=2):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    scale_add = 0.0
    scale_mul = 1.0

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype(floatX)
    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    for i in xrange(recurrence_transition_depth - 1):
        suffix = '' if i == 0 else ('_drt_%s' % i)
        U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                              ortho_weight(dim_nonlin)], axis=1)
        params[pp(prefix, 'U_nl'+suffix)] = U_nl
        params[pp(prefix, 'b_nl'+suffix)] = numpy.zeros((2 * dim_nonlin,)).astype(floatX)
        Ux_nl = ortho_weight(dim_nonlin)
        params[pp(prefix, 'Ux_nl'+suffix)] = Ux_nl
        params[pp(prefix, 'bx_nl'+suffix)] = numpy.zeros((dim_nonlin,)).astype(floatX)

        if options['layer_normalisation']:
            params[pp(prefix,'U_nl%s_lnb' % suffix)] = scale_add * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'U_nl%s_lns' % suffix)] = scale_mul * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        if options['weight_normalisation']:
            params[pp(prefix,'U_nl%s_wns') % suffix] = scale_mul * numpy.ones((2*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_wns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)

        # context to LSTM
        if i == 0:
            Wc = norm_weight(dimctx, dim*2)
            params[pp(prefix, 'Wc'+suffix)] = Wc
            Wcx = norm_weight(dimctx, dim)
            params[pp(prefix, 'Wcx'+suffix)] = Wcx
            if options['layer_normalisation']:
                params[pp(prefix,'Wc%s_lnb') % suffix] = scale_add * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'Wc%s_lns') % suffix] = scale_mul * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_lnb') % suffix] = scale_add * numpy.ones((1*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_lns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)
            if options['weight_normalisation']:
                params[pp(prefix,'Wc%s_wns') % suffix] = scale_mul * numpy.ones((2*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_wns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype(floatX)
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype(floatX)
    params[pp(prefix, 'c_tt')] = c_att

    if options['layer_normalisation']:
        # layer-normalization parameters
        params[pp(prefix,'W_lnb')] = scale_add * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'W_lns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'U_lnb')] = scale_add * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'U_lns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'Wx_lnb')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Wx_lns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_lnb')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_lns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'W_comb_att_lnb')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'W_comb_att_lns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_lnb')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_lns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    if options['weight_normalisation']:
        params[pp(prefix,'W_wns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'U_wns')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
        params[pp(prefix,'Wx_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'W_comb_att_wns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_wns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'U_att_wns')] = scale_mul * numpy.ones((1*1)).astype(floatX)

    return params


def gru_cond_layer(tparams, state_below, options, dropout, prefix='gru',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   dropout_probability_below=0,
                   dropout_probability_ctx=0,
                   dropout_probability_rec=0,
                   pctx_=None,
                   recurrence_transition_depth=2,
                   truncate_gradient=-1,
                   profile=False,
                   init_ctx=None, init_state_left=None, init_state_right=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    # mask
    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num= 1 + 2 * recurrence_transition_depth)

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']:
            return weight_norm(param, tparams[param_name+'_wns'])
        else:
            return param

    below_dropout = dropout((n_samples, dim_below),  dropout_probability_below, num=2)
    ctx_dropout = dropout((n_samples, 2*options['dim']), dropout_probability_ctx, num=4)
    extra_rec_dropout = dropout((n_samples, options['dim']), dropout_probability_rec, num=6)
    extra_ctx_dropout = dropout((n_samples, 2*options['dim']), dropout_probability_ctx, num=6)

    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim))
    if init_state_left is None:
        init_state_left = tensor.zeros((n_samples, dim))
    if init_ctx is None:
        init_ctx = tensor.zeros((n_samples, dim*2))

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    if pctx_ is None:
        pctx_ = tensor.dot(context*ctx_dropout[0], wn(pp(prefix, 'Wc_att'))) +\
            tparams[pp(prefix, 'b_att')]

    if options['layer_normalisation']:
        pctx_ = layer_norm(pctx_, tparams[pp(prefix,'Wc_att_lnb')], tparams[pp(prefix,'Wc_att_lns')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the previous output word embedding
    state_belowx = tensor.dot(state_below*below_dropout[0], wn(pp(prefix, 'Wx'))) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*below_dropout[1], wn(pp(prefix, 'W'))) +\
        tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_,
                    h_, ctx_, alpha_, hl_, hr_,
                    pctx_, cc_, rec_dropout, ctx_dropout, extra_rec_dropout, extra_ctx_dropout):
        if options['layer_normalisation']:
            x_ = layer_norm(x_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
            xx_ = layer_norm(xx_, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])

        preact1 = tensor.dot(h_*rec_dropout[0], wn(pp(prefix, 'U')))
        if options['layer_normalisation']:
            preact1 = layer_norm(preact1, tparams[pp(prefix, 'U_lnb')], tparams[pp(prefix, 'U_lns')])
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        r1 = _slice(preact1, 0, dim)
        u1 = _slice(preact1, 1, dim)

        preactx1 = tensor.dot(h_*rec_dropout[1], wn(pp(prefix, 'Ux')))
        if options['layer_normalisation']:
            preactx1 = layer_norm(preactx1, tparams[pp(prefix, 'Ux_lnb')], tparams[pp(prefix, 'Ux_lns')])
        preactx1 *= r1
        preactx1 += xx_

        h1 = tensor.tanh(preactx1)

        h1 = u1 * h_ + (1. - u1) * h1
        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_

        # # TODO past and future layers
        # # current right hidden state
        # # hc = right_manipulate_layer(tparams, hc_, y_, prefix='right_manipulate_y', m_=m_)
        # hr = right_manipulate_layer(tparams, hr_, ctx_, prefix='right_manipulate_c', m_=m_)
        # hl = left_manipulate_layer(tparams, hl_, ctx_, prefix='left_manipulate_c', m_=m_)
        # # h_l_and_r = h1.dot()])

        # feed past or(and) future content(s) into decoder state
        more_context = []
        if options['use_past_layer']:
            more_context.append(hl_)
        if options['use_future_layer']:
            more_context.append(hr_)
        if more_context:
            h3, r3, u3 = gru_unit_layer(tparams, h1, concatenate(more_context, axis=h1.ndim - 1),
                                        rec_dropout=extra_rec_dropout[0:2], ctx_dropout=extra_ctx_dropout[0:2],
                                        prefix='m_rnn_gru',
                                        m_=m_)
            h1 = h3


        # attention
        pstate_ = tensor.dot(h1*rec_dropout[2], wn(pp(prefix, 'W_comb_att')))
        if options['layer_normalisation']:
            pstate_ = layer_norm(pstate_, tparams[pp(prefix, 'W_comb_att_lnb')], tparams[pp(prefix, 'W_comb_att_lns')])
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__*ctx_dropout[1], wn(pp(prefix, 'U_att')))+tparams[pp(prefix, 'c_tt')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        h2_prev = h1
        for i in xrange(recurrence_transition_depth - 1):
            suffix = '' if i == 0 else ('_drt_%s' % i)

            preact2 = tensor.dot(h2_prev*rec_dropout[3+2*i], wn(pp(prefix, 'U_nl'+suffix)))+tparams[pp(prefix, 'b_nl'+suffix)]
            if options['layer_normalisation']:
                preact2 = layer_norm(preact2, tparams[pp(prefix, 'U_nl%s_lnb' % suffix)], tparams[pp(prefix, 'U_nl%s_lns' % suffix)])
            if i == 0:
                ctx1_ = tensor.dot(ctx_*ctx_dropout[2], wn(pp(prefix, 'Wc'+suffix))) # dropout mask is shared over mini-steps
                if options['layer_normalisation']:
                    ctx1_ = layer_norm(ctx1_, tparams[pp(prefix, 'Wc%s_lnb' % suffix)], tparams[pp(prefix, 'Wc%s_lns' % suffix)])
                preact2 += ctx1_
            preact2 = tensor.nnet.sigmoid(preact2)

            r2 = _slice(preact2, 0, dim)
            u2 = _slice(preact2, 1, dim)

            preactx2 = tensor.dot(h2_prev*rec_dropout[4+2*i], wn(pp(prefix, 'Ux_nl'+suffix)))+tparams[pp(prefix, 'bx_nl'+suffix)]
            if options['layer_normalisation']:
               preactx2 = layer_norm(preactx2, tparams[pp(prefix, 'Ux_nl%s_lnb' % suffix)], tparams[pp(prefix, 'Ux_nl%s_lns' % suffix)])
            preactx2 *= r2
            if i == 0:
               ctx2_ = tensor.dot(ctx_*ctx_dropout[3], wn(pp(prefix, 'Wcx'+suffix))) # dropout mask is shared over mini-steps
               if options['layer_normalisation']:
                   ctx2_ = layer_norm(ctx2_, tparams[pp(prefix, 'Wcx%s_lnb' % suffix)], tparams[pp(prefix, 'Wcx%s_lns' % suffix)])
               preactx2 += ctx2_
            h2 = tensor.tanh(preactx2)

            h2 = u2 * h2_prev + (1. - u2) * h2
            h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h2_prev
            h2_prev = h2

        # TODO past and future layers
        # update states of past and future layers
        if options['use_past_layer']:
            hl = left_manipulate_layer(tparams, hl_, ctx_, options,
                                   rec_dropout=extra_rec_dropout[2:4], ctx_dropout=extra_ctx_dropout[2:4],
                                   prefix='left_manipulate_c', m_=m_) if options['use_past_layer'] else hl_
        else:
            hl = hl_
        if options['use_future_layer']:
            hr = right_manipulate_layer(tparams, hr_, ctx_, options,
                                    rec_dropout=extra_rec_dropout[4:6], ctx_dropout=extra_ctx_dropout[4:6],
                                    prefix='right_manipulate_c', m_=m_) if options['use_future_layer'] else hr_
        else:
            hr = hr_

        return h2, ctx_, alpha.T, hl, hr  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = []

    if one_step:
        rval = _step(*(seqs + [init_state, init_ctx, None, init_state_left, init_state_right, pctx_, context,
                               rec_dropout, ctx_dropout, extra_rec_dropout, extra_ctx_dropout] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  init_ctx,
                                                  tensor.zeros((n_samples,
                                                               context.shape[0])),
                                                  init_state_left,
                                                  init_state_right,],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout, extra_rec_dropout,
                                                   extra_ctx_dropout]+shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=truncate_gradient,
                                    profile=profile,
                                    strict=False)
    return rval



# LSTM layer
def param_init_lstm(options, params, prefix='lstm', nin=None, dim=None,
                   recurrence_transition_depth=1,
                   **kwargs):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    scale_add = 0.0
    scale_mul = 1.0

    for i in xrange(recurrence_transition_depth):
        suffix = '' if i == 0 else ('_drt_%s' % i)

        # recurrent transformation weights for gates

        U = numpy.concatenate([ortho_weight(dim),
                               ortho_weight(dim),
                               ortho_weight(dim)],
                               axis=1)

        params[pp(prefix, 'U'+suffix)] = U
        params[pp(prefix, 'b'+suffix)] = numpy.zeros((3 * dim,)).astype(floatX)

        # recurrent transformation weights for hidden state proposal
        Ux = ortho_weight(dim)
        params[pp(prefix, 'Ux'+suffix)] = Ux
        params[pp(prefix, 'bx'+suffix)] = numpy.zeros((dim,)).astype(floatX)

        if options['layer_normalisation']:
            params[pp(prefix,'U%s_lnb' % suffix)] = scale_add * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'U%s_lns' % suffix)] = scale_mul * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        if options['weight_normalisation']:
            params[pp(prefix,'U%s_wns' % suffix)] = scale_mul * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'Ux%s_wns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)

        if i == 0:
            # embedding to gates transformation weights
            W = numpy.concatenate([norm_weight(nin, dim),
                                   norm_weight(nin, dim),
                                   norm_weight(nin, dim)],
                                   axis=1)
            params[pp(prefix, 'W'+suffix)] = W
            # embedding to hidden state proposal weights
            Wx = norm_weight(nin, dim)
            params[pp(prefix, 'Wx'+suffix)] = Wx
            if options['layer_normalisation']:
                params[pp(prefix,'W%s_lnb' % suffix)] = scale_add * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'W%s_lns' % suffix)] = scale_mul * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
            if options['weight_normalisation']:
                params[pp(prefix,'W%s_wns' % suffix)] = scale_mul * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'Wx%s_wns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)

    return params


def lstm_layer(tparams, state_below, options, dropout, prefix='lstm',
              mask=None, one_step=False,
              init_state=None,
              dropout_probability_below=0,
              dropout_probability_rec=0,
              recurrence_transition_depth=1,
              truncate_gradient=-1,
              profile=False,
              **kwargs):

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    dim = tparams[pp(prefix, 'Ux')].shape[1]

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']:
            return weight_norm(param, tparams[param_name+'_wns'])
        else:
            return param

    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim*2))

    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    below_dropout = dropout((n_samples, dim_below), dropout_probability_below, num=2)
    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num=2*(recurrence_transition_depth))

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_list, state_belowx_list = [], []

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below*below_dropout[0], wn(pp(prefix, 'W'))) + tparams[pp(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below*below_dropout[1], wn(pp(prefix, 'Wx'))) + tparams[pp(prefix, 'bx')]
    if options['layer_normalisation']:
        state_below_ = layer_norm(state_below_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
        state_belowx = layer_norm(state_belowx, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])
    state_below_list.append(state_below_)
    state_belowx_list.append(state_belowx)

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(*args):
        n_ins = 1
        m_ = args[0]
        x_list = args[1:1+n_ins]
        xx_list = args[1+n_ins:1+2*n_ins]
        h_, rec_dropout = args[-2], args[-1]

        h_prev = _slice(h_, 0, dim)
        c_prev = _slice(h_, 1, dim)

        for i in xrange(recurrence_transition_depth):
            suffix = '' if i == 0 else ('_drt_%s' % i)
            if i == 0:
                x_cur = x_list[i]
                xx_cur = xx_list[i]
            else:
                x_cur = tparams[pp(prefix, 'b'+suffix)]
                xx_cur = tparams[pp(prefix, 'bx'+suffix)]

            preact = tensor.dot(h_prev*rec_dropout[0+2*i], wn(pp(prefix, 'U'+suffix)))
            if options['layer_normalisation']:
                preact = layer_norm(preact, tparams[pp(prefix, 'U%s_lnb' % suffix)], tparams[pp(prefix, 'U%s_lns' % suffix)])
            preact += x_cur

            # gates
            gate_i = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            gate_f = tensor.nnet.sigmoid(_slice(preact, 1, dim))
            gate_o = tensor.nnet.sigmoid(_slice(preact, 2, dim))

            # compute the hidden state proposal
            preactx = tensor.dot(h_prev*rec_dropout[1+2*i], wn(pp(prefix, 'Ux'+suffix)))
            if options['layer_normalisation']:
                preactx = layer_norm(preactx, tparams[pp(prefix, 'Ux%s_lnb' % suffix)], tparams[pp(prefix, 'Ux%s_lns' % suffix)])
            preactx += xx_cur

            c = tensor.tanh(preactx)
            c = gate_f * c_prev + gate_i * c
            h = gate_o * tensor.tanh(c)

            # if state is masked, simply copy previous
            h = m_[:, None] * h + (1. - m_)[:, None] * h_prev
            c = m_[:, None] * c + (1. - m_)[:, None] * c_prev
            h_prev = h
            c_prev = c

        h = concatenate([h, c], axis=1)

        return h

    # prepare scan arguments
    seqs = [mask] + state_below_list + state_belowx_list
    _step = _step_slice
    shared_vars = [rec_dropout]

    if one_step:
        rval = _step(*(seqs + [init_state] + shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_state,
                                non_sequences=shared_vars,
                                name=pp(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=truncate_gradient,
                                profile=profile,
                                strict=False)
    rval = [rval]
    return rval

# Conditional LSTM layer with Attention
def param_init_lstm_cond(options, params, prefix='lstm_cond',
                        nin=None, dim=None, dimctx=None,
                        nin_nonlin=None, dim_nonlin=None,
                        recurrence_transition_depth=2):
    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']
    if nin_nonlin is None:
        nin_nonlin = nin
    if dim_nonlin is None:
        dim_nonlin = dim

    scale_add = 0.0
    scale_mul = 1.0

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim)],
                           axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((3 * dim,)).astype(floatX)

    U = numpy.concatenate([ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin),
                           ortho_weight(dim_nonlin)],
                           axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(nin_nonlin, dim_nonlin)
    params[pp(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim_nonlin)
    params[pp(prefix, 'Ux')] = Ux
    params[pp(prefix, 'bx')] = numpy.zeros((dim_nonlin,)).astype(floatX)

    for i in xrange(recurrence_transition_depth - 1):
        suffix = '' if i == 0 else ('_drt_%s' % i)
        U_nl = numpy.concatenate([ortho_weight(dim_nonlin),
                                  ortho_weight(dim_nonlin),
                                  ortho_weight(dim_nonlin)],
                                  axis=1)
        params[pp(prefix, 'U_nl'+suffix)] = U_nl
        params[pp(prefix, 'b_nl'+suffix)] = numpy.zeros((3 * dim_nonlin,)).astype(floatX)
        Ux_nl = ortho_weight(dim_nonlin)
        params[pp(prefix, 'Ux_nl'+suffix)] = Ux_nl
        params[pp(prefix, 'bx_nl'+suffix)] = numpy.zeros((dim_nonlin,)).astype(floatX)

        if options['layer_normalisation']:
            params[pp(prefix,'U_nl%s_lnb' % suffix)] = scale_add * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'U_nl%s_lns' % suffix)] = scale_mul * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_lnb' % suffix)] = scale_add * numpy.ones((1*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_lns' % suffix)] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        if options['weight_normalisation']:
            params[pp(prefix,'U_nl%s_wns') % suffix] = scale_mul * numpy.ones((3*dim)).astype(floatX)
            params[pp(prefix,'Ux_nl%s_wns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)

        # context to LSTM
        if i == 0:
            Wc = norm_weight(dimctx, dim*3)
            params[pp(prefix, 'Wc'+suffix)] = Wc
            Wcx = norm_weight(dimctx, dim)
            params[pp(prefix, 'Wcx'+suffix)] = Wcx
            if options['layer_normalisation']:
                params[pp(prefix,'Wc%s_lnb') % suffix] = scale_add * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'Wc%s_lns') % suffix] = scale_mul * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_lnb') % suffix] = scale_add * numpy.ones((1*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_lns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)
            if options['weight_normalisation']:
                params[pp(prefix,'Wc%s_wns') % suffix] = scale_mul * numpy.ones((3*dim)).astype(floatX)
                params[pp(prefix,'Wcx%s_wns') % suffix] = scale_mul * numpy.ones((1*dim)).astype(floatX)

    # attention: combined -> hidden
    W_comb_att = norm_weight(dim, dimctx)
    params[pp(prefix, 'W_comb_att')] = W_comb_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[pp(prefix, 'Wc_att')] = Wc_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype(floatX)
    params[pp(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype(floatX)
    params[pp(prefix, 'c_tt')] = c_att

    if options['layer_normalisation']:
        # layer-normalization parameters
        params[pp(prefix,'W_lnb')] = scale_add * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'W_lns')] = scale_mul * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'U_lnb')] = scale_add * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'U_lns')] = scale_mul * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'Wx_lnb')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Wx_lns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_lnb')] = scale_add * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_lns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'W_comb_att_lnb')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'W_comb_att_lns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_lnb')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_lns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    if options['weight_normalisation']:
        params[pp(prefix,'W_wns')] = scale_mul * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'U_wns')] = scale_mul * numpy.ones((3*dim)).astype(floatX)
        params[pp(prefix,'Wx_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'Ux_wns')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
        params[pp(prefix,'W_comb_att_wns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'Wc_att_wns')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
        params[pp(prefix,'U_att_wns')] = scale_mul * numpy.ones((1*1)).astype(floatX)

    return params


def lstm_cond_layer(tparams, state_below, options, dropout, prefix='lstm',
                   mask=None, context=None, one_step=False,
                   init_memory=None, init_state=None,
                   context_mask=None,
                   dropout_probability_below=0,
                   dropout_probability_ctx=0,
                   dropout_probability_rec=0,
                   pctx_=None,
                   recurrence_transition_depth=2,
                   truncate_gradient=-1,
                   profile=False,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        dim_below = state_below.shape[2]
    else:
        n_samples = 1
        dim_below = state_below.shape[1]

    # mask
    if mask is None:
        mask = tensor.ones((state_below.shape[0], 1))

    dim = tparams[pp(prefix, 'Wcx')].shape[1]

    rec_dropout = dropout((n_samples, dim), dropout_probability_rec, num= 1 + 2 * recurrence_transition_depth)

    # utility function to look up parameters and apply weight normalization if enabled
    def wn(param_name):
        param = tparams[param_name]
        if options['weight_normalisation']:
            return weight_norm(param, tparams[param_name+'_wns'])
        else:
            return param

    below_dropout = dropout((n_samples, dim_below),  dropout_probability_below, num=2)
    ctx_dropout = dropout((n_samples, 2*options['dim']), dropout_probability_ctx, num=4)

    # initial/previous state
    if init_state is None:
        init_state = tensor.zeros((n_samples, dim*2))


    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    if pctx_ is None:
        pctx_ = tensor.dot(context*ctx_dropout[0], wn(pp(prefix, 'Wc_att'))) +\
            tparams[pp(prefix, 'b_att')]

    if options['layer_normalisation']:
        pctx_ = layer_norm(pctx_, tparams[pp(prefix,'Wc_att_lnb')], tparams[pp(prefix,'Wc_att_lns')])

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # state_below is the previous output word embedding
    state_belowx = tensor.dot(state_below*below_dropout[0], wn(pp(prefix, 'Wx'))) +\
        tparams[pp(prefix, 'bx')]
    state_below_ = tensor.dot(state_below*below_dropout[1], wn(pp(prefix, 'W'))) +\
        tparams[pp(prefix, 'b')]

    def _step_slice(m_, x_, xx_,
                    h_, ctx_, alpha_, hl_, hr_,
                    pctx_, cc_, rec_dropout, ctx_dropout):
        if options['layer_normalisation']:
            x_ = layer_norm(x_, tparams[pp(prefix, 'W_lnb')], tparams[pp(prefix, 'W_lns')])
            xx_ = layer_norm(xx_, tparams[pp(prefix, 'Wx_lnb')], tparams[pp(prefix, 'Wx_lns')])

        h_prev = _slice(h_, 0, dim)
        c_prev = _slice(h_, 1, dim)

        preact1 = tensor.dot(h_prev*rec_dropout[0], wn(pp(prefix, 'U')))
        if options['layer_normalisation']:
            preact1 = layer_norm(preact1, tparams[pp(prefix, 'U_lnb')], tparams[pp(prefix, 'U_lns')])
        preact1 += x_
        preact1 = tensor.nnet.sigmoid(preact1)

        i1 = _slice(preact1, 0, dim)
        f1 = _slice(preact1, 1, dim)
        o1 = _slice(preact1, 2, dim)

        preactx1 = tensor.dot(h_prev*rec_dropout[1], wn(pp(prefix, 'Ux')))
        if options['layer_normalisation']:
            preactx1 = layer_norm(preactx1, tparams[pp(prefix, 'Ux_lnb')], tparams[pp(prefix, 'Ux_lns')])
        preactx1 += xx_

        c1 = tensor.tanh(preactx1)
        c1 = f1 * c_prev + i1 * c1
        h1 = o1 * tensor.tanh(c1)

        h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_prev
        c1 = m_[:, None] * c1 + (1. - m_)[:, None] * c_prev

        # TODO past and future layers
        # current right hidden state
        # hc = right_manipulate_layer(tparams, hc_, y_, prefix='right_manipulate_y', m_=m_)
        hr = right_manipulate_layer(tparams, hr_, ctx_, prefix='right_manipulate_c', m_=m_)
        hl = left_manipulate_layer(tparams, hl_, ctx_, prefix='left_manipulate_c', m_=m_)
        # h_l_and_r = h1.dot()])
        h3, r3, u3 = gru_unit_layer(tparams, h1, concatenate([hl, hr], axis=h1.ndim - 1),
                                    prefix='m_rnn_gru',
                                    m_=m_)
        h1 = h3


        # attention
        pstate_ = tensor.dot(h1*rec_dropout[2], wn(pp(prefix, 'W_comb_att')))
        if options['layer_normalisation']:
            pstate_ = layer_norm(pstate_, tparams[pp(prefix, 'W_comb_att_lnb')], tparams[pp(prefix, 'W_comb_att_lns')])
        pctx__ = pctx_ + pstate_[None, :, :]
        #pctx__ += xc_
        pctx__ = tensor.tanh(pctx__)
        alpha = tensor.dot(pctx__*ctx_dropout[1], wn(pp(prefix, 'U_att')))+tparams[pp(prefix, 'c_tt')]
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha - alpha.max(0, keepdims=True))
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        h2_prev = h1
        c2_prev = c1
        for i in xrange(recurrence_transition_depth - 1):
            suffix = '' if i == 0 else ('_drt_%s' % i)

            preact2 = tensor.dot(h2_prev*rec_dropout[3+2*i], wn(pp(prefix, 'U_nl'+suffix)))+tparams[pp(prefix, 'b_nl'+suffix)]
            if options['layer_normalisation']:
                preact2 = layer_norm(preact2, tparams[pp(prefix, 'U_nl%s_lnb' % suffix)], tparams[pp(prefix, 'U_nl%s_lns' % suffix)])
            if i == 0:
                ctx1_ = tensor.dot(ctx_*ctx_dropout[2], wn(pp(prefix, 'Wc'+suffix))) # dropout mask is shared over mini-steps
                if options['layer_normalisation']:
                    ctx1_ = layer_norm(ctx1_, tparams[pp(prefix, 'Wc%s_lnb' % suffix)], tparams[pp(prefix, 'Wc%s_lns' % suffix)])
                preact2 += ctx1_
            preact2 = tensor.nnet.sigmoid(preact2)

            i2 = _slice(preact2, 0, dim)
            f2 = _slice(preact2, 1, dim)
            o2 = _slice(preact2, 2, dim)

            preactx2 = tensor.dot(h2_prev*rec_dropout[4+2*i], wn(pp(prefix, 'Ux_nl'+suffix)))+tparams[pp(prefix, 'bx_nl'+suffix)]
            if options['layer_normalisation']:
               preactx2 = layer_norm(preactx2, tparams[pp(prefix, 'Ux_nl%s_lnb' % suffix)], tparams[pp(prefix, 'Ux_nl%s_lns' % suffix)])
            if i == 0:
               ctx2_ = tensor.dot(ctx_*ctx_dropout[3], wn(pp(prefix, 'Wcx'+suffix))) # dropout mask is shared over mini-steps
               if options['layer_normalisation']:
                   ctx2_ = layer_norm(ctx2_, tparams[pp(prefix, 'Wcx%s_lnb' % suffix)], tparams[pp(prefix, 'Wcx%s_lns' % suffix)])
               preactx2 += ctx2_

            c2 = tensor.tanh(preactx2)
            c2 = f2 * c2_prev + i2 * c2
            h2 = o2 * tensor.tanh(c2)

            h2 = m_[:, None] * h2 + (1. - m_)[:, None] * h2_prev
            h2_prev = h2
            c2_prev = c2

        h2 = concatenate([h2, c2], axis=1)

        return h2, ctx_, alpha.T, hl, hr  # pstate_, preact, preactx, r, u

    seqs = [mask, state_below_, state_belowx]
    #seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = []

    if one_step:
        rval = _step(*(seqs + [init_state, None, None, pctx_, context, rec_dropout, ctx_dropout] +
                       shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.zeros((n_samples,
                                                               context.shape[2])),
                                                  tensor.zeros((n_samples,
                                                               context.shape[0]))],
                                    non_sequences=[pctx_, context, rec_dropout, ctx_dropout]+shared_vars,
                                    name=pp(prefix, '_layers'),
                                    n_steps=nsteps,
                                    truncate_gradient=truncate_gradient,
                                    profile=profile,
                                    strict=False)
    return rval


def params_init_att(options, params, prefix='gru', nin=None, dim=None):
    # attention: hemb -> hidden
    Wc_att = norm_weight(nin)
    params[pp(prefix, 'Wc_att')] = Wc_att
    # attention: hidden bias
    b_att = numpy.zeros((nin,)).astype('float32')
    params[pp(prefix, 'b_att')] = b_att

    # attention: combined -> hemb
    W_comb_att = norm_weight(dim, nin)
    params[pp(prefix, 'W_comb_att')] = W_comb_att
    # attention:
    U_att = norm_weight(nin, 1)
    params[pp(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[pp(prefix, 'c_tt')] = c_att

    return params


def att_layer(tparams, state_below, context, pctx_, prefix='gru_cond', context_mask=None):

    pstate_ = tensor.dot(state_below, tparams[pp(prefix, 'W_comb_att')])
    # if state_below.ndim == 3:
    pctx__ = pctx_ + pstate_[None, :, :]
    # pctx__ += xc_
    pctx__ = tensor.tanh(pctx__)
    alpha = tensor.dot(pctx__, tparams[pp(prefix, 'U_att')]) + tparams[pp(prefix, 'c_tt')]
    alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
    alpha = tensor.exp(alpha)
    if context_mask:
        alpha = alpha * context_mask
    alpha = tensor.switch(tensor.eq(alpha, 0),
                          alpha,
                          alpha / alpha.sum(0, keepdims=True))
    # alpha = alpha / alpha.sum(0, keepdims=True)
    ctx_ = (context * alpha[:, :, None]).sum(0)  # current context
    return ctx_, alpha


def params_init_gate(options, params, prefix='gru', dim_i1=None, dim_i2=None, dim_o1=None):
    W1 = norm_weight(dim_i1, dim_o1)
    params[pp(prefix, 'W1')] = W1
    W2 = norm_weight(dim_i2, dim_o1)
    params[pp(prefix, 'W2')] = W2

    U = norm_weight(dim_o1, 1)
    params[pp(prefix, 'U')] = U
    c = numpy.zeros((1,)).astype('float32')
    params[pp(prefix, 'c')] = c
    return params


def gate_layer(tparams, inp1, inp2, outp, prefix='gru_cond', m_=None):
    preact = tensor.dot(inp1, tparams[pp(prefix, 'W1')])
    preact += tensor.dot(inp2, tparams[pp(prefix, 'W2')])
    preact = tanh(preact)
    preact = tensor.dot(preact, tparams[pp(prefix, 'U')]) + tparams[pp(prefix, 'c')]
    preact = preact.reshape([preact.shape[0]])
    g = tensor.nnet.sigmoid(preact)
    # ipdb.set_trace()

    outp_ = outp * g[:, None]
    outp_ = m_[:, None] * outp_ + (1. - m_)[:, None] * outp
    g = tensor.patternbroadcast(g.reshape([g.shape[0], 1]), [False, False])
    return outp_, g


def params_init_gru_unit(options, params, prefix='gru', dim_h=None, dim_ctx=None):
    W = numpy.concatenate([norm_weight(dim_ctx, dim_h),
                           norm_weight(dim_ctx, dim_h)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim_h,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim_h),
                           ortho_weight(dim_h)], axis=1)
    params[pp(prefix, 'U')] = U

    Wx = norm_weight(dim_ctx, dim_h)
    params[pp(prefix, 'Wx')] = Wx
    params[pp(prefix, 'bx')] = numpy.zeros((dim_h,)).astype('float32')
    Ux = ortho_weight(dim_h)
    params[pp(prefix, 'Ux')] = Ux

    return params


def gru_unit_layer(tparams, h_, ctx_, rec_dropout=None, ctx_dropout=None, prefix='gru_cond', m_=None):
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    dim = tparams[pp(prefix, 'Wx')].shape[1]

    preact1 = tensor.dot(h_*rec_dropout[0], tparams[pp(prefix, 'U')]) + tparams[pp(prefix, 'b')]
    preact1 += tensor.dot(ctx_*ctx_dropout[0], tparams[pp(prefix, 'W')])
    preact1 = tensor.nnet.sigmoid(preact1)

    r1 = _slice(preact1, 0, dim)
    u1 = _slice(preact1, 1, dim)

    preactx1 = tensor.dot(h_*rec_dropout[1], tparams[pp(prefix, 'Ux')]) + tparams[pp(prefix, 'bx')]
    preactx1 *= r1
    preactx1 += tensor.dot(ctx_*ctx_dropout[1], tparams[pp(prefix, 'Wx')])

    h1 = tensor.tanh(preactx1)

    h1 = u1 * h_ + (1. - u1) * h1
    h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_
    return h1, r1, u1


def params_init_mgru_unit(options, params, prefix='mgru', dim_h=None, dim_ctx=None):
    params[pp(prefix, 'U_c')] = numpy.concatenate([norm_weight(dim_ctx, dim_h),
                                                   norm_weight(dim_ctx, dim_h)], axis=1)
    params[pp(prefix, 'U_h')] = numpy.concatenate([ortho_weight(dim_h),
                                                   ortho_weight(dim_h)], axis=1)
    params[pp(prefix, 'c')] = numpy.zeros((2 * dim_h,)).astype('float32')

    params[pp(prefix, 'W_h')] = ortho_weight(dim_h)
    params[pp(prefix, 'W_c')] = norm_weight(dim_ctx, dim_h)
    params[pp(prefix, 'W')] = ortho_weight(dim_h, )
    params[pp(prefix, 'b')] = numpy.zeros((dim_h,)).astype('float32')

    return params


def mgru_unit_layer(tparams, h_, ctx_, rec_dropout=None, ctx_dropout=None, prefix='mgru', m_=None):
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    dim = h_.shape[-1]

    preact1 = tensor.dot(h_*rec_dropout[0], tparams[pp(prefix, 'U_h')]) + tparams[pp(prefix, 'c')]
    preact1 += tensor.dot(ctx_*ctx_dropout[0], tparams[pp(prefix, 'U_c')])
    preact1 = tensor.nnet.sigmoid(preact1)

    r1 = _slice(preact1, 0, dim)
    u1 = _slice(preact1, 1, dim)

    preactx1 = tensor.dot(h_*rec_dropout[1], tparams[pp(prefix, 'W_h')]) \
               - r1 * tensor.dot(ctx_*ctx_dropout[1], tparams[pp(prefix, 'W_c')])
    preactx1 = preactx1.dot(tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')]
    h1 = tensor.tanh(preactx1)

    h1 = u1 * h_ + (1. - u1) * h1
    h1 = m_[:, None] * h1 + (1. - m_)[:, None] * h_
    return h1, r1, u1


def params_init_map_minus_layer(options, params, prefix='mf', dim=None):
    params[pp(prefix, 'W_l')] = ortho_weight(dim, )
    params[pp(prefix, 'W_r')] = ortho_weight(dim, )
    params[pp(prefix, 'W')] = norm_weight(dim, options['dim_word'])
    params[pp(prefix, 'b')] = numpy.zeros((options['dim_word'],)).astype('float32')

    # params = get_layer_param('ff')(options, params, prefix=prefix + '_ff_logit',
    #                             nin=options['dim_word'],
    #                             nout=options['n_words'])
    return params


def map_minus_manipulate_layer(tparams, h_left_, h_right_, options, dropout, prefix='mf', m_=None, **ekwargs):
    preact = h_left_.dot(tparams[pp(prefix, 'W_l')]) - h_right_.dot(tparams[pp(prefix, 'W_r')])
    preact = preact.dot(tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')]
    delta = tanh(preact)
    # logit = get_layer_constr('ff')(tparams, delta, options, dropout,
    #                            prefix=prefix + '_ff_logit',
    #                            activ='linear')

    return delta


def params_init_right_manipulate_layer(options, params, prefix='gru', dim_right=None, dim_h=None):
    if options['future_layer_type'] == 'gru':
        params = params_init_gru_unit(options, params, prefix=prefix + '_gru', dim_h=dim_right, dim_ctx=dim_right)
    elif options['future_layer_type'] == 'gru_outside':
        params[pp(prefix, 'W_c')] = norm_weight(dim_right, dim_right)
        params[pp(prefix, 'W_h')] = norm_weight(dim_h, dim_right)
        params[pp(prefix, 'W')] = ortho_weight(dim_right, )
        params[pp(prefix, 'b')] = numpy.zeros((dim_right,)).astype('float32')
        params = params_init_gru_unit(options, params, prefix=prefix + '_gru', dim_h=dim_right, dim_ctx=dim_right)
    elif options['future_layer_type'] == 'gru_inside':
        params = params_init_mgru_unit(options, params, prefix=prefix + '_mgru', dim_h=dim_right, dim_ctx=dim_h)
    else:
        print 'Future_layer_type only supports: gru, gru_outside, gru_inside'
        exit(1)

    return params


def right_manipulate_layer(tparams, h_right_, h_, options, rec_dropout=None, ctx_dropout=None, prefix='right_manipulate', m_=None):
    if options['future_layer_type'] == 'gru':
        h_right, r, u, = gru_unit_layer(tparams, h_right_, h_, rec_dropout=rec_dropout, ctx_dropout=ctx_dropout,
                                        prefix=prefix + '_gru', m_=m_)
    elif options['future_layer_type'] == 'gru_outside':
        preact = h_right_.dot(tparams[pp(prefix, 'W_c')]) - h_.dot(tparams[pp(prefix, 'W_h')])
        preact = preact.dot(tparams[pp(prefix, 'W')]) + tparams[pp(prefix, 'b')]
        h_ = tanh(preact)
        h_right, r, u, = gru_unit_layer(tparams, h_right_, h_, rec_dropout=rec_dropout, ctx_dropout=ctx_dropout,
                                        prefix=prefix + '_gru', m_=m_)
    elif options['future_layer_type'] == 'gru_inside':
        h_right, r, u, = mgru_unit_layer(tparams, h_right_, h_, rec_dropout=rec_dropout, ctx_dropout=ctx_dropout,
                                         prefix=prefix + '_mgru', m_=m_)
    else:
        print 'Future_layer_type only supports: gru, gru_outside, gru_inside'
        exit(1)

    h_right = m_[:, None] * h_right + (1. - m_)[:, None] * h_right_
    return h_right


def params_init_left_manipulate_layer(options, params, prefix='gru', dim_left=None, dim_h=None):
    params = params_init_gru_unit(options, params, prefix=prefix + '_gru', dim_h=dim_left, dim_ctx=dim_h)

    return params


def left_manipulate_layer(tparams, h_left_, h_, options, rec_dropout=None, ctx_dropout=None, prefix='left_manipulate', m_=None):
    h_left, r, u, = gru_unit_layer(tparams, h_left_, h_, rec_dropout=rec_dropout, ctx_dropout=ctx_dropout,
                                   prefix=prefix + '_gru', m_=m_)
    h_left = m_[:, None] * h_left + (1. - m_)[:, None] * h_left_
    return h_left


def init_transition_diff(options, params):
    params["diff_ctx_W"] = norm_weight(options["dim"]*2, options['dim_word'])
    # params["diff_delta_W"] = norm_weight(options["dim_word"], options['dim_word'])

    return params


def get_transition_diff(tparams, deltas, ctxs, mapping=True):
    """
    :param deltas: [timesteps, batch, dim_word]
    :param ctxs: [timesteps, batch, dim_ctx]
    :return: diff [batch]
    """

    def _distance(a, b):
        return tensor.sum((a - b) ** 2, axis=a.ndim - 1)

    if mapping:
        ctxs = tensor.dot(ctxs, tparams["diff_ctx_W"])
    # deltas = tensor.dot(deltas, tparams["diff_delta_W"])
    diff = _distance(deltas, ctxs)

    return tensor.sum(diff, axis=0)


def init_word_predictor(options, params, prefix=""):
    params[pp(prefix, "V")] = norm_weight(options["dim"] * 2, options['dim_word'])
    params[pp(prefix, "c")] = numpy.zeros((options['dim_word'],)).astype('float32')

    params = get_layer_param('ff')(options, params, prefix=prefix,
                                   nin=options['dim_word'],
                                   nout=options['n_words'],
                                   weight_matrix=False,
                                   followed_by_softmax=True)
    return params

def word_predictor(x, x_mask, ctx, tparams, dropout, options, loss_fn, prefix=""):
    logit_W = tparams['Wemb'].T
    logit = tanh(tensor.dot(ctx, tparams[pp(prefix, "V")]) + tparams[pp(prefix, "c")])
    logit = get_layer_constr('ff')(tparams, logit, options, dropout,
                                   dropout_probability=options['dropout_hidden'],
                                   prefix=prefix, activ='linear', W=logit_W, followed_by_softmax=True)
    return loss_fn(x.reshape([x.shape[1], x.shape[2]]), x_mask, logit, options)



def params_init_multi_att(options, params, prefix='gru', dim_q=None, dim_k=None):
    # attention: query -> key
    Wc_att = norm_weight(dim_q, dim_k)
    params[pp(prefix, 'W_q')] = Wc_att
    # attention: hidden bias
    b_att = numpy.zeros((dim_k,)).astype('float32')
    params[pp(prefix, 'b')] = b_att
    # attention: key -> key
    W_comb_att = norm_weight(dim_k, dim_k)
    params[pp(prefix, 'W_k')] = W_comb_att

    # attention:
    U_att = norm_weight(dim_k, 1)
    params[pp(prefix, 'v')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[pp(prefix, 'c')] = c_att

    return params


def multi_att_layer(tparams, query, key, value, prefix='gru_cond', value_mask=None):
    """
    :param tparams:
    :param query: [len_q, batch, dim]
    :param key:   [len_k, batch, dim]
    :param value: [len_v, batch, dim]
    :param prefix:
    :param value_mask: [len_v, batch]
    :return:
        alpha: [len_q, batch, len_k]
        context: [len_q, batch, dim]
    """
    # transpose to [batch, len_q, len_k, dim]
    len_q = query.shape[0]
    len_k = key.shape[0]
    query = tensor.tile(tensor.transpose(query, [1, 0, 'x', 2]), [1, 1, len_k, 1])
    key = tensor.tile(tensor.transpose(key, [1, 'x', 0, 2]), [1, len_q, 1, 1])
    value = tensor.tile(tensor.transpose(value, [1, 'x', 0, 2]), [1, len_q, 1, 1])

    preact = tensor.dot(query, tparams[pp(prefix, 'W_q')]) + tensor.dot(key, tparams[pp(prefix, 'W_k')]) + \
        tparams[pp(prefix, 'b')]
    combined = tanh(preact)
    alpha = tensor.dot(combined, tparams[pp(prefix, 'v')]) + tparams[pp(prefix, 'c')]
    # [batch, len_q, len_k]
    alpha = alpha.reshape([alpha.shape[0], alpha.shape[1], alpha.shape[2]])

    if value_mask:
        # [batch, 1, len_k]
        value_mask = tensor.transpose(value_mask, [1, 'x', 0])
        alpha *= value_mask

    # [batch, len_q, dim]
    context = (value * alpha[:, :, :, None]).sum(2)
    context = tensor.transpose(context, [1, 0, 2])
    return context, alpha


def params_init_wpd(options, params, prefix='wpd'):
    params = get_layer_param('ff')(options, params, prefix="ff_logit_wpd",
                                   nin=options['dim_word'],
                                   nout=options['dim_word'])

    return params


def wpd(tparams, logit, options, y_mask, y, dropout):
    logit_v = get_layer_constr('ff')(tparams, logit, options, dropout,
                                 prefix='ff_logit_wpd', activ='tanh')
    logit_v = get_layer_constr('ff')(tparams, logit_v, options, dropout,
                                 prefix='ff_logit', activ='linear')

    logit_v_shp = logit_v.shape
    probs_v = tensor.nnet.softmax(logit_v.reshape([logit_v_shp[0] * logit_v_shp[1],
                                                   logit_v_shp[2]]))
    # cost
    vv = y.shape[0]
    y_v = tensor.arange(y.shape[1]) * options['n_words'] + y
    y_v, updates = theano.scan(lambda l, y_t: y_v,
                               sequences=tensor.arange(vv),
                               non_sequences=y_v)

    y_v_mask = tensor.zeros_like(y_mask)

    def add_cost(y_t, l, prob, y, y_mask, y_z_mask):
        temp = (y_t + l * options['n_words']).flatten()
        prob_t = prob.flatten()[temp]
        prob_t = -tensor.log(prob_t)
        prob_t = prob_t.reshape([y.shape[0], y.shape[1]])

        y_mask = tensor.set_subtensor(y_mask[l:], y_z_mask[l:])

        prob_t *= y_mask
        prob_t = prob_t.sum(0)
        y_mask_tt = y_mask.sum(0) + 1
        prob_t = prob_t / y_mask_tt
        return prob_t

    cost_v, updates = theano.scan(fn=add_cost,
                                  outputs_info=None,
                                  sequences=[y_v, tensor.arange(vv)],
                                  non_sequences=[probs_v, y, y_mask, y_v_mask])
    cost_v = cost_v.mean(0)

    return cost_v


def params_init_wpe(options, params, prefix='wpe'):
    params = get_layer_param(options['decoder'])(options, params,
                                                 prefix='wpe',
                                                 nin=options['dim_word'],
                                                 dim=options['dim'],
                                                 dimctx=options["dim"]*2,
                                                 recurrence_transition_depth=options[
                                                     'dec_base_recurrence_transition_depth'])
    return params


def wpe(tparams, emb, options, y_mask, y, ctx, x_mask, init_state, dropout):

    proj_r = get_layer_constr(options['decoder'])(tparams, emb[0], options, dropout,
                                                prefix='wpe',
                                                mask=None, context=ctx,
                                                context_mask=x_mask,
                                                one_step=True,
                                                init_state=init_state)
    proj_h_r = proj_r[0]
    ctxs_r = proj_r[1]
    # opt_ret['dec_alphas_r'] = proj_r[2]

    logit_lstm_r = get_layer_constr('ff')(tparams, proj_h_r, options, dropout,
                                      prefix='ff_logit_lstm', activ='linear')
    logit_prev_r = get_layer_constr('ff')(tparams, emb, options, dropout,
                                      prefix='ff_logit_prev', activ='linear')
    logit_ctx_r = get_layer_constr('ff')(tparams, ctxs_r, options, dropout,
                                     prefix='ff_logit_ctx', activ='linear')

    logit_r = tensor.tanh(logit_lstm_r + logit_prev_r + logit_ctx_r)

    logit_r = get_layer_constr('ff')(tparams, logit_r, options, dropout,
                                 prefix='ff_logit', activ='linear')
    logit_shp_r = logit_r.shape

    probs_r = tensor.nnet.softmax(logit_r.reshape([logit_shp_r[0] * logit_shp_r[1],
                                                   logit_shp_r[2]]))

    # cost
    # change the reverse cost option
    y_tt = tensor.arange(y.shape[1]) * options['n_words'] + y
    cost_tt = -tensor.log(probs_r.flatten()[y_tt])
    cost_tt = cost_tt.reshape([y_tt.shape[0], y_tt.shape[1]])
    cost_tt = (cost_tt * y_mask).sum(0)
    y_mast_tt = y_mask.sum(0)
    cost_tt = cost_tt / y_mast_tt

    return cost_tt

