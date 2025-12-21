"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
import numpy as np

from model import Model
import cifar10_input

with open('config.json') as config_file:
    config = json.load(config_file)

data_path = config['data_path']

def run_attack(checkpoint, x_adv, epsilon):
  cifar = cifar10_input.CIFAR10Data(data_path)

  model = Model(mode='eval')

  saver = tf.train.Saver()

  # Use config values so behaviour is consistent across scripts
  num_eval_examples = int(config.get('num_eval_examples', 10000))
  eval_batch_size = int(config.get('eval_batch_size', 100))

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0

  # Ensure float32 for numeric ops
  x_nat = cifar.eval_data.xs.astype(np.float32)
  x_adv = x_adv.astype(np.float32)

  # Detect pixel scale (0..1 vs 0..255)
  max_nat = float(x_nat.max())
  pixel_scale = 255.0 if max_nat > 1.0 else 1.0

  # If epsilon is given in normalized units (e.g. 8/255), convert to pixel units
  eps_pixels = float(epsilon) * pixel_scale

  # Compute linf (in pixel units)
  l_inf = np.amax(np.abs(x_nat - x_adv))

  if l_inf > eps_pixels + 1e-6:
    print('maximum perturbation found: {}'.format(l_inf))
    print('maximum perturbation allowed (in pixel units): {}'.format(eps_pixels))
    return

  y_pred = [] # label accumulator

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)

    # Iterate over the samples batch-by-batch
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = x_adv[bstart:bend, :].astype(np.float32)
      y_batch = cifar.eval_data.ys[bstart:bend]

      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch}
      cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],
                                        feed_dict=dict_adv)

      total_corr += cur_corr
      y_pred.append(y_pred_batch)

  accuracy = total_corr / float(num_eval_examples)

  print('Accuracy: {:.2f}%'.format(100.0 * accuracy))
  y_pred = np.concatenate(y_pred, axis=0)
  np.save('pred.npy', y_pred)
  print('Output saved at pred.npy')

if __name__ == '__main__':
  import json

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_dir = config['model_dir']

  checkpoint = tf.train.latest_checkpoint(model_dir)
  x_adv = np.load(config['store_adv_path'])

  if checkpoint is None:
    print('No checkpoint found')
  elif x_adv.shape != (int(config.get('num_eval_examples', 10000)), 32, 32, 3):
    print('Invalid shape: expected ({}, 32, 32, 3), found {}'.format(
        int(config.get('num_eval_examples', 10000)), x_adv.shape))
  else:
    # Check pixel range and adjust expectation based on cifar eval data
    cifar = cifar10_input.CIFAR10Data(data_path)
    x_nat_sample = cifar.eval_data.xs[:1].astype(np.float32)
    max_nat = float(x_nat_sample.max())
    pixel_scale = 255.0 if max_nat > 1.0 else 1.0

    min_adv = float(x_adv.min())
    max_adv = float(x_adv.max())
    if max_adv > pixel_scale + 1e-6 or min_adv < -1e-6:
      print('Invalid pixel range. Expected [0, {}], found [{}, {}]'.format(
                                                                pixel_scale,
                                                                min_adv,
                                                                max_adv))
    else:
      # Convert epsilon in config (assumed normalized, e.g., 8/255) to pixel units for check
      epsilon = float(config.get('epsilon', 8.0))
      run_attack(checkpoint, x_adv, epsilon)

