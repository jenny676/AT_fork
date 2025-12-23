# train_tf2.py
import json
import os
import csv
import time
import math
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

# load config
with open('config.json') as fh:
    config = json.load(fh)

# --- hyperparams from config
EVAL_NUM_STEPS = int(config.get('eval_num_steps', 20))
EVAL_RESTARTS = int(config.get('eval_restarts', 5))
EVAL_BATCH_SIZE = int(config.get('eval_batch_size', 256))

tf.random.set_seed(int(config['tf_random_seed']))
np.random.seed(int(config['np_random_seed']))

max_num_training_steps = int(config['max_num_training_steps'])
num_output_steps = int(config['num_output_steps'])
num_summary_steps = int(config['num_summary_steps'])
num_checkpoint_steps = int(config['num_checkpoint_steps'])
step_size_schedule = config['step_size_schedule']  # list of [step, lr]
weight_decay = float(config['weight_decay'])
data_path = config['data_path']
momentum = float(config['momentum'])
batch_size = int(config['training_batch_size'])

# Import your model and attack rewritten for TF2
from model import Model           # must be tf.keras.Model subclass
from pgd_attack import LinfPGDAttack  # must be TF2-friendly (see below)
import cifar10_input

# --- data
raw_cifar = cifar10_input.CIFAR10Data(data_path)  # if this returns numpy arrays, ok
# Expect raw_cifar to have train_data, test_data objects with .xs, .ys (numpy)
# or adapt to return a tf.data.Dataset below.

# --- build model (tf.keras.Model)
model = Model(mode='train')  # recommended: Model should subclass tf.keras.Model

# --- learning rate schedule
# build boundaries and values from step_size_schedule (same semantics as TF1)
boundaries = [int(sss[0]) for sss in step_size_schedule][1:]
values = [sss[1] for sss in step_size_schedule]
lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum)

# --- checkpointing
ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                           optimizer=optimizer,
                           model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, config['model_dir'], max_to_keep=3)
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Restored from", ckpt_manager.latest_checkpoint)

# --- summaries
writer = tf.summary.create_file_writer(config['model_dir'])

# --- attack (TF2 version)
attack = LinfPGDAttack(
    model,
    float(config['epsilon']),
    int(config['num_steps']),
    float(config['step_size']),
    bool(config['random_start']),
    config['loss_func']
)
eval_attack = LinfPGDAttack(
    model,
    float(config['epsilon']),
    EVAL_NUM_STEPS,
    float(config['step_size']),
    bool(config['random_start']),
    config['loss_func']
)

# --- helper metrics
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

# build tf.data.Dataset for training from numpy arrays (adapt if cifar already yields datasets)
def make_dataset(xs, ys, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((xs.astype(np.float32), ys.astype(np.int64)))
    if shuffle:
        ds = ds.shuffle(buffer_size=50000, seed=int(config['np_random_seed']))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# optionally support train_frac as in your TF1 script
train_frac = float(config.get('train_frac', 1.0))
cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar)  # adapt this to TF2 if needed

# We'll assume cifar.train_data.xs / .ys exist
if train_frac < 1.0:
    xs = cifar.train_data.xs
    ys = cifar.train_data.ys
    num_total = xs.shape[0]
    num_keep = int(num_total * train_frac)
    rng = np.random.RandomState(int(config.get('np_random_seed', 0)) +
                                int(config.get('train_frac_seed', 0)))
    keep_idx = np.sort(rng.permutation(num_total)[:num_keep])
    train_xs = xs[keep_idx].copy()
    train_ys = ys[keep_idx].copy()
else:
    train_xs, train_ys = cifar.train_data.xs, cifar.train_data.ys

train_ds = make_dataset(train_xs, train_ys, batch_size, shuffle=True)

# --- loss function (example: sparse categorical crossentropy from logits)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

# Adversarial training step: create adversarial examples then do gradient step
@tf.function
def train_step(x_batch, y_batch):
    # x_batch, y_batch are tensors
    # Produce adversarial examples (attack.perturb must be TF2/eager-compatible)
    x_batch_adv = attack.perturb(x_batch, y_batch)  # returns tf.Tensor same shape
    with tf.GradientTape() as tape:
        logits = model(x_batch_adv, training=True)
        per_example_loss = loss_fn(y_batch, logits)
        loss = tf.reduce_mean(per_example_loss)
        # add weight decay if model does not include it
        if weight_decay:
            wd_loss = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'bias' not in v.name])
            loss += wd_loss
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # update metrics
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y_batch, logits)

    ckpt.step.assign_add(1)
    return loss

# training loop
start_time = time.time()
step = int(ckpt.step.numpy())
for x_batch_np, y_batch_np in train_ds:
    if step >= max_num_training_steps:
        break
    step += 1
    x_batch = tf.convert_to_tensor(x_batch_np)
    y_batch = tf.convert_to_tensor(y_batch_np)
    t0 = timer()
    loss_value = train_step(x_batch, y_batch)
    t_elapsed = timer() - t0

    if step % num_output_steps == 0:
        print(f"Step {step}: {datetime.now()}  loss={train_loss_metric.result().numpy():.4f} acc={train_acc_metric.result().numpy():.4%}")

    if step % num_summary_steps == 0:
        with writer.as_default():
            tf.summary.scalar('train_loss', train_loss_metric.result(), step=step)
            tf.summary.scalar('train_acc', train_acc_metric.result(), step=step)
            writer.flush()

    if step % num_checkpoint_steps == 0:
        ckpt_manager.save()
        print("Saved checkpoint at step", step)

# final save
ckpt_manager.save()
print("Training finished in", time.time() - start_time)
