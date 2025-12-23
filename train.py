"""Trains a model, saving checkpoints and tensorboard summaries along
   the way. Modified to run PGD-10 training and PGD-20 evaluation logging.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import csv
import time
import shutil
from timeit import default_timer as timer
import math

import tensorflow as tf
import numpy as np

from model import Model
import cifar10_input
from pgd_attack import LinfPGDAttack

with open('config.json') as config_file:
    config = json.load(config_file)

# optional extra config defaults for evaluation
# (if not present in config.json these defaults will be used)
EVAL_NUM_STEPS = int(config.get('eval_num_steps', 20))    # PGD-20 for eval
EVAL_RESTARTS = int(config.get('eval_restarts', 5))      # 5 restarts recommended
EVAL_BATCH_SIZE = int(config.get('eval_batch_size', 256))

# seeding randomness
tf.set_random_seed(int(config['tf_random_seed']))
np.random.seed(int(config['np_random_seed']))

# Setting up training parameters
max_num_training_steps = int(config['max_num_training_steps'])
num_output_steps = int(config['num_output_steps'])
num_summary_steps = int(config['num_summary_steps'])
num_checkpoint_steps = int(config['num_checkpoint_steps'])
step_size_schedule = config['step_size_schedule']
weight_decay = float(config['weight_decay'])
data_path = config['data_path']
momentum = float(config['momentum'])
batch_size = int(config['training_batch_size'])

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

# Set up adversary for training (uses config['num_steps'], e.g. 10)
attack = LinfPGDAttack(model,
                       float(config['epsilon']),
                       int(config['num_steps']),
                       float(config['step_size']),
                       bool(config['random_start']),
                       config['loss_func'])

# Set up adversary for evaluation (PGD-20). This is separate so we can evaluate
# with PGD-20 while training uses PGD-10.
eval_attack = LinfPGDAttack(model,
                           float(config['epsilon']),
                           EVAL_NUM_STEPS,
                           float(config['step_size']),
                           bool(config['random_start']),
                           config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

# Helper: write metrics line as JSON into metrics.jsonl
def append_metrics(out_dir, metrics):
    fname = os.path.join(out_dir, "metrics.jsonl")
    with open(fname, "a") as fh:
        fh.write(json.dumps(metrics) + "\n")

# ---- CSV helpers ----------------------------------------------------------
METRICS_CSV = os.path.join(model_dir, "metrics.csv")

def init_metrics_csv(path=METRICS_CSV):
  header = ["epoch","train_time","test_time","lr",
            "train_loss","train_acc","train_robust_loss","train_robust_acc",
            "test_loss","test_acc","test_robust_loss","test_robust_acc"]
  if not os.path.exists(path):
    with open(path, "w", newline='') as fh:
      writer = csv.writer(fh)
      writer.writerow(header)

def append_metrics_csv(row, path=METRICS_CSV):
  """Row is an iterable of values in same order as header."""
  with open(path, "a", newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(row)

# ---- dataset metrics computation -----------------------------------------
def compute_dataset_metrics(sess, data_obj, attack_obj, batch_size, restarts):
  """
  Compute:
    - clean mean loss (model.mean_xent) and accuracy (model.num_correct / N)
    - robust mean loss and robust accuracy (PGD attack applied, worst-over-restarts semantics)
  Returns:
    (mean_clean_loss, clean_acc, mean_robust_loss, robust_acc, elapsed_seconds)
  """
  t0 = time.time()
  total_clean_loss = 0.0
  total_robust_loss = 0.0
  total_corr_clean = 0
  total_corr_robust = 0

  N = int(getattr(data_obj, "num_examples", (data_obj.xs.shape[0] if hasattr(data_obj, "xs") else 0)))
  num_batches = int(math.ceil(N / float(batch_size)))

  for ibatch in range(num_batches):
    bstart = ibatch * batch_size
    bend = min(bstart + batch_size, N)

    x_batch = data_obj.xs[bstart:bend].astype(np.float32)
    y_batch = data_obj.ys[bstart:bend]

    # clean metrics
    feed_nat = {model.x_input: x_batch, model.y_input: y_batch}
    cur_corr_nat, cur_loss_nat = sess.run([model.num_correct, model.mean_xent], feed_dict=feed_nat)
    total_corr_clean += int(cur_corr_nat)
    total_clean_loss += float(cur_loss_nat) * (bend - bstart)  # mean_xent -> multiply back by batch size

    # robust (adversarial) metrics: craft adversarial examples with restarts
    x_batch_adv = attack_obj.perturb(x_batch, y_batch, sess, restarts=restarts)
    feed_adv = {model.x_input: x_batch_adv, model.y_input: y_batch}
    cur_corr_adv, cur_loss_adv = sess.run([model.num_correct, model.mean_xent], feed_dict=feed_adv)
    total_corr_robust += int(cur_corr_adv)
    total_robust_loss += float(cur_loss_adv) * (bend - bstart)

  # get means
  mean_clean_loss = total_clean_loss / float(N)
  clean_acc = float(total_corr_clean) / float(N)
  mean_robust_loss = total_robust_loss / float(N)
  robust_acc = float(total_corr_robust) / float(N)
  elapsed = time.time() - t0
  return mean_clean_loss, clean_acc, mean_robust_loss, robust_acc, elapsed

# NOTE: we removed the older evaluate_checkpoint helper to avoid duplication.
# compute_dataset_metrics is used for both train and test evaluations.

with tf.Session() as sess:

  # initialize data augmentation (keeps internal train/eval pointers)
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

  # optional: subsample training set according to config['train_frac']
  train_frac = float(config.get('train_frac', 1.0))
  if train_frac <= 0 or train_frac > 1.0:
    raise ValueError("config['train_frac'] must be in (0,1]. Got: {}".format(train_frac))

  if train_frac < 1.0:
    try:
      xs = cifar.train_data.xs
      ys = cifar.train_data.ys
    except AttributeError:
      raise RuntimeError("cifar.train_data does not expose .xs/.ys; adapt subsample code to your data loader.")

    num_total = xs.shape[0]
    num_keep = int(np.floor(num_total * train_frac))
    if num_keep < 1:
      raise ValueError("train_frac too small, resulting in zero examples to train on")

    frac_seed = int(config.get('train_frac_seed', 0))
    rng = np.random.RandomState(seed=(int(config.get('np_random_seed', 0)) + frac_seed))

    perm = rng.permutation(num_total)
    keep_idx = np.sort(perm[:num_keep])  # sort to keep natural order if desired

    cifar.train_data.xs = xs[keep_idx].copy()
    cifar.train_data.ys = ys[keep_idx].copy()
    cifar.train_data.num_examples = cifar.train_data.xs.shape[0]

    print("Subsampled training set: keeping {}/{} examples ({:.2%})".format(
        num_keep, num_total, train_frac))

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
   # ensure metrics CSV has header
   init_metrics_csv()
   
   sess.run(tf.global_variables_initializer())
   
   # ---- RESUME FROM CHECKPOINT IF EXISTS ----
   ckpt = tf.train.latest_checkpoint(model_dir)
   if ckpt is not None:
       print("Restoring from checkpoint:", ckpt)
       saver.restore(sess, ckpt)
       start_step = sess.run(global_step)
   else:
       print("No checkpoint found. Training from scratch.")
       start_step = 0
   
   training_time = 0.0
   
   # Main training loop
   for ii in range(start_step, max_num_training_steps):
       x_batch, y_batch = cifar.train_data.get_next_batch(
           batch_size, multiple_passes=True
       )

    # Compute Adversarial Perturbations (training PGD, e.g. PGD-10)
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}

    # ------------------- OUTPUT & SUMMARIES -------------------
    # Output to stdout (train-time quick stats)
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0

    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Ensure CSV exists (first checkpoint will create header)
    # Call once before first checkpoint
    if ii == 0:
      init_metrics_csv()

    # Write a checkpoint, evaluate (PGD-20) and append metrics CSV/JSONL
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

      # ------------------- EVAL (TEST) -------------------
      print("Running PGD-{} evaluation ({} restarts) at step {}...".format(EVAL_NUM_STEPS, EVAL_RESTARTS, ii))
      eval_start = timer()
      test_loss, test_acc, test_robust_loss, test_robust_acc, test_time = compute_dataset_metrics(
          sess,
          cifar.eval_data,
          eval_attack,
          batch_size=int(config.get('eval_batch_size', EVAL_BATCH_SIZE)),
          restarts=int(config.get('eval_restarts', EVAL_RESTARTS))
      )
      eval_end = timer()
      print(" Eval done in {:.2f}s - clean_acc: {:.4f}, robust_pgd{}_acc: {:.4f}".format(
          eval_end - eval_start, test_acc, EVAL_NUM_STEPS, test_robust_acc))

      # ------------------- EVAL (TRAIN) -------------------
      print("Computing train-set metrics (this may be slow)...")
      train_start = timer()
      train_loss, train_acc, train_robust_loss, train_robust_acc, train_time_epoch = compute_dataset_metrics(
          sess,
          cifar.train_data,
          eval_attack,   # use PGD-20 for comparable robust metrics
          batch_size=int(config.get('eval_batch_size', EVAL_BATCH_SIZE)),
          restarts=int(config.get('eval_restarts', EVAL_RESTARTS))
      )
      train_end = timer()
      print(" Train eval done in {:.2f}s - train_acc: {:.4f}, train_robust_acc: {:.4f}".format(
          train_end - train_start, train_acc, train_robust_acc))

      # ------------------- LEARNING RATE -------------------
      try:
        current_lr = float(sess.run(learning_rate))
      except Exception:
        current_lr = float(config.get('initial_learning_rate', 0.0))

      # ------------------- write JSONL metrics (optional) -------------------
      metrics = {
          "global_step": int(sess.run(global_step)),
          "step": int(ii),
          "clean_acc": float(test_acc),
          "robust_pgd{}_acc".format(EVAL_NUM_STEPS): float(test_robust_acc),
          "train_clean_acc": float(train_acc),
          "train_robust_acc": float(train_robust_acc),
          "eval_restarts": int(EVAL_RESTARTS),
          "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      }
      append_metrics(model_dir, metrics)

      # ------------------- append CSV row -------------------
      # Row order:
      # epoch,train_time,test_time,lr,
      # train_loss,train_acc,train_robust_loss,train_robust_acc,
      # test_loss,test_acc,test_robust_loss,test_robust_acc
      row = [
        int(ii),
        float(train_time_epoch),
        float(test_time),
        float(current_lr),
        float(train_loss),
        float(train_acc),
        float(train_robust_loss),
        float(train_robust_acc),
        float(test_loss),
        float(test_acc),
        float(test_robust_loss),
        float(test_robust_acc)
      ]
      append_metrics_csv(row)

    # ------------------- ACTUAL TRAINING STEP -------------------
    start = timer()
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start

  # end training loop
# end session
