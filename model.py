# model.py
# Standard ResNet-18 adapted for CIFAR (TF1 style)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class Model(object):
  """ResNet-18 model for CIFAR-10."""

  def __init__(self, mode):
    """ResNet constructor.

    Args:
      mode: One of 'train' and 'eval'.
    """
    self.mode = mode
    self._build_model()

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    assert self.mode in ('train', 'eval')
    with tf.variable_scope('input'):
      # Inputs: expect pixel-valued images (0..255). Keep per-image standardization here,
      # but you can switch to dataset normalization if desired.
      self.x_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
      self.y_input = tf.placeholder(tf.int64, shape=None)

      # keep behavior similar to prior code: per-image standardization
      input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                                    self.x_input)

      # Initial conv: 3x3, 64 filters, stride 1
      x = self._conv('init_conv', input_standardized, filter_size=3,
                     in_filters=3, out_filters=64, strides=self._stride_arr(1))

    # ResNet-18: 4 stages, each with 2 basic blocks.
    # channels per stage: 64, 128, 256, 512
    x = self._resnet_stage(x, 64, 2, name='stage1', first_stride=1)
    x = self._resnet_stage(x, 128, 2, name='stage2', first_stride=2)
    x = self._resnet_stage(x, 256, 2, name='stage3', first_stride=2)
    x = self._resnet_stage(x, 512, 2, name='stage4', first_stride=2)

    with tf.variable_scope('post'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.0)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      self.pre_softmax = self._fully_connected(x, 10)

    # Predictions & metrics (same API as before)
    self.predictions = tf.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.equal(self.predictions, self.y_input)
    self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    with tf.variable_scope('costs'):
      self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.pre_softmax, labels=self.y_input)
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()

  # ------------------- ResNet building blocks -------------------
  def _resnet_stage(self, x, out_filters, num_blocks, name, first_stride):
    """Build a stage consisting of `num_blocks` basic blocks.

    Args:
      x: input tensor
      out_filters: number of filters for this stage
      num_blocks: number of basic blocks (for ResNet-18 it's 2)
      name: scope name
      first_stride: stride for the first block in the stage
    """
    with tf.variable_scope(name):
      # first block may change dimensions via stride and possibly increasing filters
      x = self._basic_block(x, out_filters, stride=first_stride, name='block0')
      for i in range(1, num_blocks):
        x = self._basic_block(x, out_filters, stride=1, name='block%d' % i)
    return x

  def _basic_block(self, x, out_filters, stride, name):
    """Basic ResNet block (2 conv layers) for ResNet-18/34.

    Implements:
      y = F(x) + shortcut(x)
    Where F = conv-bn-relu-conv-bn
    """
    in_filters = int(x.get_shape()[-1])
    with tf.variable_scope(name):
      with tf.variable_scope('sub1'):
        x1 = self._batch_norm('bn1', x)
        x1 = self._relu(x1, 0.0)
        x1 = self._conv('conv1', x1, filter_size=3, in_filters=in_filters, out_filters=out_filters, strides=self._stride_arr(stride))

      with tf.variable_scope('sub2'):
        x1 = self._batch_norm('bn2', x1)
        x1 = self._relu(x1, 0.0)
        x1 = self._conv('conv2', x1, filter_size=3, in_filters=out_filters, out_filters=out_filters, strides=self._stride_arr(1))

      # shortcut
      if in_filters != out_filters or stride != 1:
        # downsample spatially and adjust channels with 1x1 conv (following standard ResNet)
        with tf.variable_scope('shortcut'):
          shortcut = self._conv('conv_shortcut', x, filter_size=1, in_filters=in_filters, out_filters=out_filters, strides=self._stride_arr(stride))
      else:
        shortcut = x

      out = x1 + shortcut
      return out

  # ------------------- layers / helpers -------------------
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=(self.mode == 'train'))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution with Xavier/He initialization consistent with previous code."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])

  def _decay(self):
    """L2 weight decay loss. Keeps same variable-name pattern as original code."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    if costs:
      return tf.add_n(costs)
    else:
      return tf.constant(0.0)
