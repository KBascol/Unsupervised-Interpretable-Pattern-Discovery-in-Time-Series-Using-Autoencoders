"""
    Module defining a trainable sparse autoencoder
"""

import os
import logging as log
import tensorflow as tf

import numpy as np

# Gaussian kernel (deviation of 1)
GAUSSIAN = [0.05399096651318806,
            0.24197072451914337,
            0.3989422804014327,
            0.24197072451914337,
            0.05399096651318806]


class AutoEnc(object):
    """
    Object defining the layers and operations of an autoencoder.
    """

    def __init__(self, weights_path=None, data_dict=None):
        """ Load weights file if exists """
        if data_dict is not None:
            self.data_dict = data_dict
        elif weights_path is not None and os.path.isfile(weights_path):
            self.data_dict = np.load(weights_path).item()
        else:
            log.info("Learning from scratch !")
            self.data_dict = None

        self.var_dict = {}

    def build(self, batch, nb_filters, filters_length):
        """ Build Autoencoder """

        batch_shape = batch.get_shape().as_list()
        batch_size = tf.shape(batch)[0]

        self.inputs = batch

        self.latent = self.encoder(nb_filters=nb_filters,
                                   filters_width=filters_length)

        self.sparsity(nb_filters)

        self.outputs = self.decoder(batch_size=batch_size,
                                    filters_height=batch_shape[1],
                                    filters_width=filters_length)

        self.data_dict = None

    def encoder(self, nb_filters, filters_width):
        """ Encoding, just one convolutional layer """

        with tf.variable_scope("encoder"):
            latent = self.conv_layer(self.inputs, nb_filters, filters_width,
                                     "encode_layer")
            return latent

    def decoder(self, batch_size, filters_height, filters_width):
        """ Decoding, just one deconvolutional layer """

        with tf.variable_scope("decoder"):
            output = self.deconv_layer(self.latent, batch_size, 1, filters_height, filters_width,
                                       "decode_layer")
            return output

    def conv_layer(self, inputs, out_channels, filters_length, name, stride=1):
        """ define a convolutional layer """

        with tf.variable_scope(name):
            inputs_shape = inputs.get_shape().as_list()

            self.encode_filt, conv_biases = self.get_conv_var(inputs_shape[1],
                                                              filters_length,
                                                              inputs_shape[-1],
                                                              out_channels,
                                                              name)

            conv = tf.nn.conv2d(inputs, self.encode_filt,
                                [1, stride, stride, 1], padding='VALID')
            bias = tf.nn.bias_add(conv, conv_biases)

            return tf.nn.relu(bias)

    def deconv_layer(self, latent, batch_size, out_channels, filters_height, filters_width,
                     name, stride=1):
        """ define a deconvolutional layer """

        with tf.variable_scope(name):
            latent_shape = latent.get_shape().as_list()

            self.decode_filt, deconv_biases = self.get_deconv_var(filters_height,
                                                                  filters_width,
                                                                  out_channels,
                                                                  latent_shape[-1],
                                                                  name)
            output_shape = (batch_size,
                            latent_shape[1]*stride+filters_height-stride,
                            latent_shape[2]*stride+filters_width-stride,
                            out_channels)

            deconv = tf.nn.conv2d_transpose(latent, self.decode_filt, output_shape,
                                            [1, stride, stride, 1], padding='VALID')
            bias = tf.nn.bias_add(deconv, deconv_biases)

            return tf.nn.relu(bias)

    def get_conv_var(self, filter_height, filter_width, in_channels, out_channels, name):
        """ Get variables for convolutional layers """

        # Initialization between 0 and 1/<nb weights in a filter>
        initial_value = tf.random_uniform([filter_height, filter_width, in_channels, out_channels],
                                          minval=0, maxval=1/(filter_height+filter_width))

        filters = self.get_var(initial_value, name + "/filter:0", 0, name + "/filter")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name + "/bias:0", 0, name + "/bias")

        return filters, biases

    def get_deconv_var(self, filter_height, filter_width, out_channels, in_channels, name):
        """ Get variables for convolutional layers """

        # Initialization between 0 and 1/<nb weights in a filter>
        initial_value = tf.random_uniform([filter_height, filter_width, out_channels, in_channels],
                                          minval=0, maxval=1/(filter_height+filter_width))

        filters = self.get_var(initial_value, name + "/filter:0", 0, name + "/filter")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name + "/bias:0", 0, name + "/bias")

        return filters, biases

    def get_var(self, initial_value, name, idx, var_name):
        """ Get an initialized tensorflow variable """

        var = tf.Variable(self.get_value(initial_value, name),
                          name=var_name)

        self.var_dict[name] = var

        assert var.get_shape() == initial_value.get_shape()

        return var

    def sparsity(self, nb_filters):
        """
        define a sparsity layer
            - padded convolution with a gaussian filter
            - substract input with its result (sharpen peaks)
            - apply AdaRelu = max(0, feature_maps.max()*0.66) (keep only peaks)
        """

        kern = np.array(GAUSSIAN, dtype=np.float32)
        kern = np.expand_dims(kern, 0)
        kern = np.expand_dims(kern, 2)
        kern = np.expand_dims(kern, 3)
        kern = np.repeat(kern, nb_filters, axis=2)

        filt = tf.constant(kern)

        sparse_latent = tf.nn.depthwise_conv2d(self.latent, filt, [1, 1, 1, 1], padding='SAME')

        self.latent = adarelu(tf.subtract(self.latent, sparse_latent))

    def get_value(self, initial_value, name):
        """ get a value from pretrained weights if exists """

        if self.data_dict is not None and name in self.data_dict:
            return self.data_dict[name]

        return initial_value

    def save_npy(self, sess, npy_path):
        """ Save tensorflow session in npy files """

        assert isinstance(sess, tf.Session)

        data_dict = {}

        for name, var in self.var_dict.items():
            var_out = sess.run(var)

            data_dict[name] = var_out

        np.save(npy_path, data_dict)

        log.info("Weights saved to %s", npy_path)

        return npy_path

def ch_layer(layer):
    """ Get number of channels in layer """

    return layer.get_shape().as_list()[-1]

def adarelu(latent):
    """
    AdaRelu activation function
    Threshold for each filter activations depends on the highest peak in it
    """

    zeros = tf.zeros_like(latent)

    return tf.where(tf.greater(latent, tf.reduce_max(latent, axis=2, keepdims=True)*0.66),
                    latent,
                    zeros)
