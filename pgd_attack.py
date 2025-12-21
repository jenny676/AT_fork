"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import cifar10_input

class LinfPGDAttack:
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):
    self.model = model
    self.epsilon = epsilon         # expected in [0,1] (e.g., 8/255)
    self.num_steps = num_steps
    self.step_size = step_size     # expected in [0,1] (e.g., 2/255)
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    # gradient of loss wrt inputs (TF tensor). Will be evaluated in sess.run
    self.grad = tf.gradients(loss, model.x_input)[0]


  def perturb(self, x_nat, y, sess, restarts=1):
    """
    x_nat: numpy array of shape [B, H, W, C], dtype can be uint8 or float32
    y: numpy array of labels
    restarts: number of random restarts (default 1). This returns the adversarial
              example per input that leads to misclassification if any restart succeeds.
    Returns: x_adv_best (same shape as x_nat, dtype float32)
    """

    # Convert to float32 and detect scale (0..1 vs 0..255)
    x_nat = x_nat.astype(np.float32)
    maxv = x_nat.max()
    if maxv > 1.0:
      # input is in 0..255 range, convert to 0..1
      scale = 255.0
    else:
      scale = 1.0

    # convert epsilon / step_size from [0,1] units to actual pixel units in the input array
    eps_pixels = float(self.epsilon) * scale
    step_pixels = float(self.step_size) * scale

    B = x_nat.shape[0]
    x_nat_clipped = np.clip(x_nat, 0.0, scale)

    # keep track of best adversarial examples found (initially natural)
    x_best = x_nat_clipped.copy()
    # keep track of whether an example is currently misclassified (we want one that misclassifies if possible)
    # We'll use model.predictions to check — but to avoid extra sessions, we will track logits during restarts below.
    found_adv = np.zeros(B, dtype=np.bool)

    for r in range(restarts):
      # random init inside epsilon-ball
      if self.rand:
        x = x_nat_clipped + np.random.uniform(-eps_pixels, eps_pixels, x_nat.shape).astype(np.float32)
        x = np.clip(x, 0.0, scale)
      else:
        x = x_nat_clipped.copy()

      # iterative PGD steps
      for i in range(self.num_steps):
        # ensure float32 and feed into TF
        feed = {self.model.x_input: x, self.model.y_input: y}
        grad = sess.run(self.grad, feed_dict=feed)  # shape [B,H,W,C] float32

        # gradient step: x = x + step_pixels * sign(grad)
        x = x + step_pixels * np.sign(grad).astype(np.float32)

        # project into linf ball around x_nat and valid pixel range
        x = np.minimum(np.maximum(x, x_nat_clipped - eps_pixels), x_nat_clipped + eps_pixels)
        x = np.clip(x, 0.0, scale)

      # after this restart, check predictions on x to see which are successful
      # get model predictions (argmax)
      try:
        preds = sess.run(self.model.predictions, feed_dict={self.model.x_input: x})
      except AttributeError:
        # fallback to pre_softmax argmax if predictions isn't available
        preds = sess.run(tf.argmax(self.model.pre_softmax, 1), feed_dict={self.model.x_input: x})

      # update x_best for those that are now adversarial (pred != y)
      is_adv = (preds != y)
      # replace x_best where we found adv this restart
      x_best[is_adv & (~found_adv)] = x[is_adv & (~found_adv)]
      # mark as found (we want any restart that succeeds)
      found_adv = found_adv | is_adv

      # Optional: for examples already found adversarial, you might want to keep the one producing highest loss.
      # Implementing "choose worst over restarts" by loss would require computing loss outputs and comparing per-example; keep simple: first found wins.

    return x_best



if __name__ == '__main__':
  import json
  import sys
  import math


  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model(mode='eval')
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'])
  saver = tf.train.Saver()

  data_path = config['data_path']
  cifar = cifar10_input.CIFAR10Data(data_path)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))

