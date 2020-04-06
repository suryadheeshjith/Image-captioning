from __future__ import print_function, division
from builtins import range
import numpy as np



def word_embedding_forward(x, W):

    out, cache = None, None
    out = W[x]
    cache = (x, W)

    return out, cache


def word_embedding_backward(dout, cache):

    dW = None
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW,x,dout)

    return dW


def sigmoid(x):

    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)



def temporal_affine_forward(x, w, b):

    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):

    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx

################################################################################
# RNN
################################################################################
def rnn_step_forward(x, prev_h, Wx, Wh, b):

    next_h, cache = None, None
    next_h = np.tanh(x.dot(Wx)+prev_h.dot(Wh)+b)
    cache = (x, prev_h, Wx, Wh, b, next_h)

    return next_h, cache


def rnn_step_backward(dnext_h, cache):

    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    x, prev_h, Wx, Wh, b, tanh = cache
    dtanh = 1 - tanh**2
    dnext_dtanh = dnext_h*dtanh
    dx = dnext_dtanh.dot(Wx.T)
    dprev_h = dnext_dtanh.dot(Wh.T)
    dWx = x.T.dot(dnext_dtanh)
    dWh = prev_h.T.dot(dnext_dtanh)
    db = dnext_dtanh.sum(axis=0)

    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):

    h, cache = None, None
    N,T,D = x.shape
    _,H = h0.shape
    h = np.zeros((N,T,H))
    cache = []

    prev_h = h0
    for i in range(T):
        next_h,cache_h = rnn_step_forward(x[:,i,:], prev_h, Wx, Wh, b)
        cache.append(cache_h)
        h[:,i,:] = next_h
        prev_h = next_h

    return h, cache


def rnn_backward(dh, cache):

    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N,T,H = dh.shape
    dx1, dprev_h1, dWx1, dWh1, db1 = rnn_step_backward(dh[:,T-1,:], cache[T-1])
    D = dx1.shape[1]
    dx = np.zeros((N,T,D))

    dx[:,T-1,:] = dx1
    dWx= dWx1
    dWh= dWh1
    db=db1

    for i in range(T-2,-1,-1):
        dx1, dprev_h1, dWx1, dWh1, db1 = rnn_step_backward(dh[:,i,:]+dprev_h1, cache[i])
        dx[:,i,:] = dx1
        dWx+= dWx1
        dWh+= dWh1
        db+=db1
    dh0 = dprev_h1

    return dx, dh0, dWx, dWh, db


################################################################################
# LSTM
################################################################################
def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):

    next_h, next_c, cache = None, None, None
    a = x.dot(Wx) + prev_h.dot(Wh) + b
    ai,af,ao,ag = np.split(a, 4,axis=1)
    i,f,o = sigmoid(ai),sigmoid(af),sigmoid(ao)
    g = np.tanh(ag)
    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)
    cache = {'x':x, 'a':a, 'i':i, 'f':f, 'o':o, 'g':g, 'next_c':next_c,
                    'next_h':next_h, 'prev_c':prev_c, 'prev_h':prev_h, 'Wh':Wh, 'Wx':Wx}

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):

    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None

    x,a,i,f,o,g,next_c,next_h,prev_c,prev_h,Wh,Wx = cache['x'],cache['a'],cache['i'],cache['f'],cache['o'],cache['g'],cache['next_c'],\
                                                    cache['next_h'],cache['prev_c'],cache['prev_h'],cache['Wh'],cache['Wx']


    dnext_h__dnext_c = o * (1 - np.tanh(next_c)**2)
    dnext_c += dnext_h * dnext_h__dnext_c

    dprev_c = dnext_c * f

    dk_df = dnext_c * prev_c
    dk_di = dnext_c * g
    dk_dg = dnext_c * i
    dk_do = dnext_h * np.tanh(next_c)

    dk_dai = dk_di * (i*(1-i))
    dk_daf = dk_df * (f*(1-f))
    dk_dao = dk_do * (o*(1-o))
    dk_dag = dk_dg * (1 - g**2)

    dk_da = np.concatenate((dk_dai,dk_daf,dk_dao,dk_dag),axis=1)

    dx = dk_da.dot(Wx.T)
    dWx = x.T.dot(dk_da)
    dprev_h = dk_da.dot(Wh.T)
    dWh = prev_h.T.dot(dk_da)
    db = np.sum(dk_da,axis=0)


    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):

    h, cache = None, None

    N,T,D = x.shape
    _,H = h0.shape
    h = np.zeros((N,T,H))
    cache = []

    prev_h = h0
    prev_c = np.zeros((N,H))
    for i in range(T):
        next_h, next_c, cache_h = lstm_step_forward(x[:,i,:],prev_h, prev_c, Wx, Wh, b)
        cache.append(cache_h)
        h[:,i,:] = next_h
        prev_h = next_h
        prev_c = next_c


    return h, cache


def lstm_backward(dh, cache):

    dx, dh0, dWx, dWh, db = None, None, None, None, None


    N,T,H = dh.shape
    dnext_c0 = np.zeros((N,H)) ###### Init to 0s
    dx1, dprev_h1, dprev_c1, dWx1, dWh1, db1 = lstm_step_backward(dh[:,T-1,:], dnext_c0, cache[T-1])
    D = dx1.shape[1]
    dx = np.zeros((N,T,D))

    dx[:,T-1,:] = dx1
    dWx= dWx1
    dWh= dWh1
    db=db1

    for i in range(T-2,-1,-1):
        dx1, dprev_h1, dprev_c1, dWx1, dWh1, db1 = lstm_step_backward(dh[:,i,:]+dprev_h1, dprev_c1, cache[i])
        dx[:,i,:] = dx1
        dWx+= dWx1
        dWh+= dWh1
        db+=db1
    dh0 = dprev_h1

    return dx, dh0, dWx, dWh, db
