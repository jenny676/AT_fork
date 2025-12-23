"""
Trains a model, saving checkpoints and tensorboard summaries along
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
EVAL_NUM_STEPS = int(config.get('eval_num_steps', 20))
EVAL_RESTARTS = int(config.get('eval_restarts', 5))
EVAL_BATCH_SIZE = int(config.get('eval_batch_size', 256))

# seeding randomness
tf.set_random_seed(int(config['tf_random_seed']))
np.random.seed(int(config['np_random_seed']))

# training parameters
max_num_training_steps = int(config['max_num_training_steps'])
num_output_steps = int(config['num_output_steps'])
num_summary_steps = int(config['num_summary_steps'])
num_checkpoint_steps = int(config['num_checkpoint_steps'])
step_size_schedule = config['step_size_schedule']
weight_decay = float(config['weight_decay'])
data_path = config['data_path']
momentum = float(config['momentum'])
batch_size = int(config['training_batch_size'])

# data and model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule][1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values
)

total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(
    learning_rate, momentum
).minimize(total_loss, global_step=global_step)

# adversary (training)
attack = LinfPGDAttack(
    model,
    float(config['epsilon']),
    int(config['num_steps']),
    float(config['step_size']),
    bool(config['random_start']),
    config['loss_func']
)

# adversary (evaluation)
eval_attack = LinfPGDAttack(
    model,
    float(config['epsilon']),
    EVAL_NUM_STEPS,
    float(config['step_size']),
    bool(config['random_start']),
    config['loss_func']
)

# outputs
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

shutil.copy('config.json', model_dir)

# ---- metrics helpers ------------------------------------------------------
def append_metrics(out_dir, metrics):
    fname = os.path.join(out_dir, "metrics.jsonl")
    with open(fname, "a") as fh:
        fh.write(json.dumps(metrics) + "\n")


METRICS_CSV = os.path.join(model_dir, "metrics.csv")


def init_metrics_csv(path=METRICS_CSV):
    header = [
        "epoch", "train_time", "test_time", "lr",
        "train_loss", "train_acc", "train_robust_loss", "train_robust_acc",
        "test_loss", "test_acc", "test_robust_loss", "test_robust_acc"
    ]
    if not os.path.exists(path):
        with open(path, "w", newline='') as fh:
            csv.writer(fh).writerow(header)


def append_metrics_csv(row, path=METRICS_CSV):
    with open(path, "a", newline='') as fh:
        csv.writer(fh).writerow(row)


# ---- dataset metrics ------------------------------------------------------
def compute_dataset_metrics(sess, data_obj, attack_obj, batch_size, restarts):
    t0 = time.time()
    total_clean_loss = 0.0
    total_robust_loss = 0.0
    total_corr_clean = 0
    total_corr_robust = 0

    N = int(getattr(data_obj, "num_examples", data_obj.xs.shape[0]))
    num_batches = int(math.ceil(N / float(batch_size)))

    for ibatch in range(num_batches):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, N)

        x_batch = data_obj.xs[bstart:bend].astype(np.float32)
        y_batch = data_obj.ys[bstart:bend]

        feed_nat = {model.x_input: x_batch, model.y_input: y_batch}
        corr_nat, loss_nat = sess.run(
            [model.num_correct, model.mean_xent], feed_dict=feed_nat
        )

        total_corr_clean += int(corr_nat)
        total_clean_loss += float(loss_nat) * (bend - bstart)

        x_batch_adv = attack_obj.perturb(x_batch, y_batch, sess, restarts=restarts)
        feed_adv = {model.x_input: x_batch_adv, model.y_input: y_batch}
        corr_adv, loss_adv = sess.run(
            [model.num_correct, model.mean_xent], feed_dict=feed_adv
        )

        total_corr_robust += int(corr_adv)
        total_robust_loss += float(loss_adv) * (bend - bstart)

    return (
        total_clean_loss / N,
        total_corr_clean / N,
        total_robust_loss / N,
        total_corr_robust / N,
        time.time() - t0
    )


# ============================== TRAINING ===================================
with tf.Session() as sess:
    cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

    train_frac = float(config.get('train_frac', 1.0))
    if not (0 < train_frac <= 1.0):
        raise ValueError("train_frac must be in (0,1]")

    if train_frac < 1.0:
        xs = cifar.train_data.xs
        ys = cifar.train_data.ys

        num_total = xs.shape[0]
        num_keep = int(num_total * train_frac)

        rng = np.random.RandomState(
            int(config.get('np_random_seed', 0)) + int(config.get('train_frac_seed', 0))
        )
        keep_idx = np.sort(rng.permutation(num_total)[:num_keep])

        cifar.train_data.xs = xs[keep_idx].copy()
        cifar.train_data.ys = ys[keep_idx].copy()
        cifar.train_data.num_examples = num_keep

    summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
    init_metrics_csv()

    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.latest_checkpoint(model_dir)
    start_step = 0
    if ckpt:
        saver.restore(sess, ckpt)
        start_step = sess.run(global_step)

    training_time = 0.0

    for ii in range(start_step, max_num_training_steps):
        x_batch, y_batch = cifar.train_data.get_next_batch(
            batch_size, multiple_passes=True
        )

        start = timer()
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
        training_time += timer() - start

        nat_dict = {model.x_input: x_batch, model.y_input: y_batch}
        adv_dict = {model.x_input: x_batch_adv, model.y_input: y_batch}

        if ii % num_output_steps == 0:
            nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
            adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
            print(f"Step {ii}: {datetime.now()}")
            print(f"  nat acc: {nat_acc:.4%}")
            print(f"  adv acc: {adv_acc:.4%}")

        if ii % num_summary_steps == 0:
            summary = sess.run(merged_summaries, feed_dict=adv_dict)
            summary_writer.add_summary(summary, sess.run(global_step))

        if ii % num_checkpoint_steps == 0:
            saver.save(sess, os.path.join(model_dir, "checkpoint"),
                       global_step=global_step)

        start = timer()
        sess.run(train_step, feed_dict=adv_dict)
        training_time += timer() - start
