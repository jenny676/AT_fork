"""Trains a model, saving checkpoints and tensorboard summaries along
   the way. Modified to run PGD-10 training and PGD-20 evaluation logging.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
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
EVAL_NUM_STEPS = config.get('eval_num_steps', 20)    # PGD-20 for eval
EVAL_RESTARTS = config.get('eval_restarts', 5)      # 5 restarts recommended
EVAL_BATCH_SIZE = config.get('eval_batch_size', config.get('eval_batch_size', 256))

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']

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
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])

# Set up adversary for evaluation (PGD-20). This is separate so we can evaluate
# with PGD-20 while training uses PGD-10.
eval_attack = LinfPGDAttack(model,
                           config['epsilon'],
                           EVAL_NUM_STEPS,
                           config['step_size'],
                           config['random_start'],
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

# Helper: evaluate model on eval set with PGD-20 and multiple restarts
def evaluate_checkpoint(sess, cifar_augmented, eval_attack, eval_batch_size, eval_restarts):
    """Returns (clean_acc, robust_acc_pgd20) on the full eval set.

    Robustness is estimated by running `eval_restarts` random-start PGD attacks and
    marking an example as robust only if it remains correctly classified under ALL restarts
    (this is the 'worst-case over restarts' behaviour).
    """
    # reset eval pointer if CIFAR wrapper uses an internal pointer
    # The CIFAR10Data implementation in Madry's repo uses get_next_batch on eval_data,
    # which advances a pointer. There's no explicit reset API exposed here, so we rely
    # on reading exactly num_examples via successive calls.
    num_eval = cifar_augmented.eval_data.num_examples
    num_batches = int(math.ceil(num_eval / eval_batch_size))

    total = 0
    correct_clean = 0
    correct_robust = 0

    # iterate over the eval set once
    for _ in range(num_batches):
        x_batch_eval, y_batch_eval = cifar_augmented.eval_data.get_next_batch(eval_batch_size, multiple_passes=False)

        # clean accuracy (single forward)
        nat_dict = {model.x_input: x_batch_eval, model.y_input: y_batch_eval}
        clean_acc_batch = sess.run(model.accuracy, feed_dict=nat_dict)
        batch_size_actual = x_batch_eval.shape[0]
        correct_clean += clean_acc_batch * batch_size_actual

        # robust check across restarts:
        # robust_mask[i] remains True only if example i is correctly predicted in every restart.
        robust_mask = np.ones(batch_size_actual, dtype=np.bool)

        for r in range(eval_restarts):
            # produce adversarial examples with eval_attack (PGD-20)
            x_batch_adv = eval_attack.perturb(x_batch_eval, y_batch_eval, sess)

            # Get per-example predictions on adversarial images.
            # We expect the Model to expose a per-example prediction tensor, e.g. model.predictions
            # If your Model doesn't have `predictions`, replace with the appropriate op that yields
            # per-example argmax logits (for example: sess.run(tf.argmax(model.pre_softmax,1), ...))
            try:
                preds = sess.run(model.predictions, feed_dict={model.x_input: x_batch_adv})
            except AttributeError:
                # fallback: try to compute argmax of model.pre_softmax if available
                try:
                    preds = sess.run(tf.argmax(model.pre_softmax, 1), feed_dict={model.x_input: x_batch_adv})
                except Exception:
                    raise RuntimeError("Model doesn't expose 'predictions' or 'pre_softmax'. Please modify the code to obtain per-example predictions.")

            # update robust mask: must be correct on this restart as well
            robust_mask &= (preds == y_batch_eval)

        correct_robust += robust_mask.sum()
        total += batch_size_actual

    clean_acc = float(correct_clean) / total
    robust_acc = float(correct_robust) / total
    return clean_acc, robust_acc

with tf.Session() as sess:

  # initialize data augmentation (keeps internal train/eval pointers)
   cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)
   train_frac = float(config.get('train_frac', 1.0))
   if train_frac <= 0 or train_frac > 1.0:
     raise ValueError("config['train_frac'] must be in (0,1]. Got: {}".format(train_frac))
   
   # Only subsample if train_frac < 1.0
   if train_frac < 1.0:
     # Get original arrays (AugmentedCIFAR10Data keeps underlying arrays on train_data.xs / ys)
     try:
       xs = cifar.train_data.xs
       ys = cifar.train_data.ys
     except AttributeError:
       raise RuntimeError("cifar.train_data does not expose .xs/.ys; adapt subsample code to your data loader.")
   
     num_total = xs.shape[0]
     num_keep = int(np.floor(num_total * train_frac))
     if num_keep < 1:
       raise ValueError("train_frac too small, resulting in zero examples to train on")
   
     # deterministic sampling per run using config['train_frac_seed'] combined with global seed
     frac_seed = int(config.get('train_frac_seed', 0))
     # create rng seeded by (global np_random_seed, frac_seed) for repeatability
     rng = np.random.RandomState(seed=(int(config.get('np_random_seed', 0)) + frac_seed))
   
     perm = rng.permutation(num_total)
     keep_idx = np.sort(perm[:num_keep])  # sort to keep natural order if desired
   
     # subsample the CIFAR training arrays in-place so the rest of the repo continues to work
     cifar.train_data.xs = xs[keep_idx].copy()
     cifar.train_data.ys = ys[keep_idx].copy()
     # update num_examples so get_next_batch and epoch math work
     cifar.train_data.num_examples = cifar.train_data.xs.shape[0]
   
     print("Subsampled training set: keeping {}/{} examples ({:.2%})".format(
         num_keep, num_total, train_frac))


  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                       multiple_passes=True)

    # Compute Adversarial Perturbations (training PGD, e.g. PGD-10)
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch, sess)
    end = timer()
    training_time += end - start

    nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}

    adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}

    # Output to stdout
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

    # Write a checkpoint AND evaluate (PGD-20) and log metrics
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

      # Run evaluation on the full eval set using eval_attack (PGD-20)
      print("Running PGD-{} evaluation ({} restarts) at step {}...".format(EVAL_NUM_STEPS, EVAL_RESTARTS, ii))
      eval_start = timer()
      clean_acc, robust_acc = evaluate_checkpoint(sess, cifar, eval_attack, EVAL_BATCH_SIZE, EVAL_RESTARTS)
      eval_end = timer()
      print(" Eval done in {:.2f}s - clean_acc: {:.4f}, robust_pgd{}_acc: {:.4f}".format(eval_end - eval_start, clean_acc, EVAL_NUM_STEPS, robust_acc))

      # assemble metrics and append to metrics.jsonl
      metrics = {
          "global_step": int(sess.run(global_step)),
          "step": int(ii),
          "clean_acc": float(clean_acc),
          "robust_pgd{}_acc".format(EVAL_NUM_STEPS): float(robust_acc),
          "eval_restarts": int(EVAL_RESTARTS),
          "timestamp": datetim

