# -*- coding: utf-8 -*-
"""
Created on Tue May 23 19:21:44 2017

@author: ramon
"""
#from keras import layers
#import tensorflow as tf
#
#
#class LRN(layers.Layer):
#    def __init__(self, alpha=0.0001,k=1,beta=0.75,n=5, **kwargs):
#        self.alpha = alpha
#        self.k = k
#        self.beta = beta
#        self.n = n
#        super(LRN, self).__init__(**kwargs)
#
#    def call(self, x, mask=None):
#        b, r, c, ch = x.shape
#        half_n = self.n // 2 # half the local region
#        input_sqr = tf.square(x) # square the input
#        #	extra_channels = T.alloc(0., b, ch + 2*half_n, r, c) # make an empty tensor with zero pads along channel dimension
#        extra_channels = tf.Variable(tf.zeros(b,r,c,ch + 2*half_n))
#
#        input_sqr = T.set_subtensor(extra_channels[:, half_n:half_n+ch, :, :],input_sqr) # set the center to be the squared input
#        scale = tf.Variable(tf.zeros(b,r,c,ch + 2*half_n))
#        init_op = tf.global_variables_initializer()
#
#        with tf.Session() as sess:
#            sess.run(init_op)
#            sess.run(extra_channels[:, :, :, half_n:half_n+ch].assign(input_sqr))
#
#        scale = self.k # offset for the scale
#        norm_alpha = self.alpha / self.n # normalized alpha
#        for i in range(self.n):
#            scale += norm_alpha * input_sqr[:, :, :, i:i+ch]
#        scale = scale ** self.beta
#        x = x / scale
#        return x
#
#    def get_config(self):
#        config = {"alpha": self.alpha,
#		  "k": self.k,
#		  "beta": self.beta,
#		  "n": self.n}
#        base_config = super(LRN, self).get_config()
#        return dict(list(base_config.items()) + list(config.items()))

from keras.engine import Layer
from keras import backend as K


class LRN2D(Layer):

    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError(
                "LRN2D only works with odd n. n provided: " + str(n))
        super(LRN2D, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = int(n)

        # print('LRN', self.alpha, self.k, self.beta, self.n)

    def build(self, input_shape):
        self.shape = input_shape
        super(LRN2D, self).build(input_shape)

    def call(self, x, mask=None):
        if K.image_dim_ordering == "th":
            _, f, r, c = self.shape
        else:
            _, r, c, f = self.shape
        half_n = self.n // 2
        squared = K.square(x)
        pooled = K.pool2d(squared, (half_n, half_n), strides=(1, 1),
                          padding="same", pool_mode="avg")
        if K.image_dim_ordering == "th":
            summed = K.sum(pooled, axis=1, keepdims=True)
            averaged = (self.alpha / self.n) * K.repeat_elements(summed, f, axis=1)
        else:
            summed = K.sum(pooled, axis=3, keepdims=True)
            averaged = (self.alpha / self.n) * K.repeat_elements(summed, f, axis=3)
        denom = K.pow(self.k + averaged, self.beta)
        return x / denom

    # def get_output_shape_for(self, input_shape):
    #     return input_shape

    # def call(self, inputs, **kwargs):
    #     print('in call')
    #
    #     b, ch, r, c = self.shape
    #     if b is None:
    #         b = 1
    #
    #     print (b, ch, r, c)
    #     half_n = self.n // 2
    #     input_sqr = K.square(inputs)
    #
    #     extra_channels = K.zeros((b, int(ch) + 2 * half_n, r, c))
    #     input_sqr = K.concatenate([extra_channels[:, :int(half_n), :, :],
    #                                input_sqr,
    #                                extra_channels[:, int(half_n + ch):, :, :]],
    #                               axis=1)
    #     scale = self.k
    #     norm_alpha = self.alpha/self.n
    #
    #     for i in range(self.n):
    #         scale += norm_alpha * input_sqr[:, i:i + int(ch), :, :]
    #     scale = scale ** self.beta
    #
    #     outputs = inputs/scale
    #     return outputs

    # def get_output(self, train):
    #     X = self.get_input(train)
    #     b, ch, r, c = K.shape(X)
    #     half_n = self.n // 2
    #     input_sqr = K.square(X)
    #
    #     extra_channels = K.zeros((b, ch + 2 * half_n, r, c))
    #     input_sqr = K.concatenate([extra_channels[:, :half_n, :, :],
    #                                input_sqr,
    #                                extra_channels[:, half_n + ch:, :, :]],
    #                               axis=1)
    #     scale = self.k
    #
    #     for i in range(self.n):
    #         scale += self.alpha * input_sqr[:, i:i + ch, :, :]
    #     scale = scale ** self.beta
    #
    #     return X / scale

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "alpha": self.alpha,
                  "k": self.k,
                  "beta": self.beta,
                  "n": self.n}
        base_config = super(LRN2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
