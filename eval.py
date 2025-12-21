"""
Infinite evaluation loop going through the checkpoints in the model directory
as they appear and evaluating them. Accuracy and average loss are printed and
added as tensorboard summaries.
"""
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

import cifar10_input
from model import Model
from pgd_attack import LinfPGDAttack

def append_metrics(out_dir, metrics):
  fname = os.path.join(out_dir, "metrics.jsonl")
  with open(fname, "a") as fh:
    fh.write(json.dumps(metrics) + "\n")

# Global constants
with open('config.json') as config_file:
  config = json.load(config_file)

num_eval_examples = config['num_eval_examples']
eval_batch_size = config.get('eval_batch_size', 100)
eval_on_cpu = config.get('eval_on_cpu', False)
data_path = config['data_path']

# evaluation-specific PGD settings (paper/protocol)
EVAL_NUM_STEPS = config.get('eval_num_steps', 20)    # PGD-20 for eval
EVAL_RESTARTS = config.get('eval_restarts', 5)       # number of random restarts

model_dir = config['model_dir']

# Set upd the data, hyperparameters, and the model
cifar = cifar10_input.CIFAR10Data(data_path)

if eval_on_cpu:
  with tf.device("/cpu:0"):
    model = Model(mode='eval')
    attack = LinfPGDAttack(model,
                       config['epsilon'],
                       EVAL_NUM_STEPS,        # use eval steps here
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])
else:
  model = Model(mode='eval')
  attack = LinfPGDAttack(model,
                       config['epsilon'],
                       EVAL_NUM_STEPS,        # use eval steps here
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])

global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval')
if not os.path.exists(eval_dir):
  os.makedirs(eval_dir)

last_checkpoint_filename = ''
already_seen_state = False

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)

# A function for evaluating a single checkpoint
def evaluate_checkpoint(filename):
  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, filename)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      dict_nat = {model.x_input: x_batch,
                  model.y_input: y_batch}

      x_batch_adv = attack.perturb(x_batch, y_batch, sess, restarts=EVAL_RESTARTS)

      dict_adv = {model.x_input: x_batch_adv,
                  model.y_input: y_batch}

      cur_corr_nat, cur_xent_nat = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_nat)
      cur_corr_adv, cur_xent_adv = sess.run(
                                      [model.num_correct,model.xent],
                                      feed_dict = dict_adv)

      print("Correctly classified natural examples: {}".format(cur_corr_nat))
      print("Correctly classified adversarial examples: {}".format(cur_corr_adv))
      total_xent_nat += cur_xent_nat
      total_xent_adv += cur_xent_adv
      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv

    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    summary = tf.Summary(value=[
          tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
          tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
    summary_writer.add_summary(summary, global_step.eval(sess))

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg adv loss: {:.4f}'.format(avg_xent_adv))
    # save metrics to metrics.jsonl
    metrics = {
      "checkpoint": os.path.basename(filename),
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      "num_eval_examples": int(num_eval_examples),
      "eval_batch_size": int(eval_batch_size),
      "eval_num_steps": int(EVAL_NUM_STEPS),
      "eval_restarts": int(EVAL_RESTARTS),
      "clean_acc": float(acc_nat),
      "robust_pgd{}_acc".format(EVAL_NUM_STEPS): float(acc_adv)
    }
    append_metrics(model_dir, metrics)


# Infinite eval loop
while True:
  cur_checkpoint = tf.train.latest_checkpoint(model_dir)

  # Case 1: No checkpoint yet
  if cur_checkpoint is None:
    if not already_seen_state:
      print('No checkpoint yet, waiting ...', end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
  # Case 2: Previously unseen checkpoint
  elif cur_checkpoint != last_checkpoint_filename:
    print('\nCheckpoint {}, evaluating ...   ({})'.format(cur_checkpoint,
                                                          datetime.now()))
    sys.stdout.flush()
    last_checkpoint_filename = cur_checkpoint
    already_seen_state = False
    evaluate_checkpoint(cur_checkpoint)
  # Case 3: Previously evaluated checkpoint
  else:
    if not already_seen_state:
      print('Waiting for the next checkpoint ...   ({})   '.format(
            datetime.now()),
            end='')
      already_seen_state = True
    else:
      print('.', end='')
    sys.stdout.flush()
    time.sleep(10)
