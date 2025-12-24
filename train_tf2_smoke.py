# ============================================================
# train_tf2.py -- end-to-end TF2 training script
#
# SMOKE TEST GOAL:
#   Verify environment, data, model, attack, optimizer,
#   checkpointing, and logging all work end-to-end.
#
# If this runs ~10–20 steps without crashing, the system is OK.
# ============================================================

import json
import os
import csv
import time
import math
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

# ------------------------------------------------------------
# [SMOKE TEST] Config file must exist and be valid JSON
# ------------------------------------------------------------
with open('config.json') as fh:
    config = json.load(fh)

# ---- hyperparams / config values
EVAL_NUM_STEPS = int(config.get('eval_num_steps', 20))
EVAL_RESTARTS = int(config.get('eval_restarts', 5))
EVAL_BATCH_SIZE = int(config.get('eval_batch_size', 256))

# ------------------------------------------------------------
# [SMOKE TEST] Determinism + TF import sanity
# ------------------------------------------------------------
tf.random.set_seed(int(config['tf_random_seed']))
np.random.seed(int(config['np_random_seed']))

max_num_training_steps = int(config['max_num_training_steps'])
num_output_steps = int(config['num_output_steps'])
num_summary_steps = int(config['num_summary_steps'])
num_checkpoint_steps = int(config['num_checkpoint_steps'])
step_size_schedule = config['step_size_schedule']
weight_decay = float(config['weight_decay'])
data_path = config['data_path']
momentum = float(config['momentum'])
batch_size = int(config['training_batch_size'])
train_frac = float(config.get('train_frac', 1.0))
train_frac_seed = int(config.get('train_frac_seed', 0))

# ------------------------------------------------------------
# [SMOKE TEST] Model directory must be writable
# ------------------------------------------------------------
model_dir = config['model_dir']
os.makedirs(model_dir, exist_ok=True)

# Copy config for reproducibility (optional)
try:
    import shutil
    shutil.copy('config.json', model_dir)
except Exception:
    pass

# ------------------------------------------------------------
# [SMOKE TEST] Local modules must import cleanly
#   - model.py
#   - pgd_attack.py
#   - cifar10_input.py
# ------------------------------------------------------------
from model import Model
from pgd_attack import LinfPGDAttack
import cifar10_input

# ------------------------------------------------------------
# [SMOKE TEST] CIFAR-10 data must exist at data_path
# ------------------------------------------------------------
raw_cifar = cifar10_input.CIFAR10Data(data_path)
cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, padding=4)

# Optional dataset subsampling (for fast smoke tests)
if not (0 < train_frac <= 1.0):
    raise ValueError("train_frac must be in (0,1]")
if train_frac < 1.0:
    xs = cifar.train_data.xs
    ys = cifar.train_data.ys
    num_total = xs.shape[0]
    num_keep = int(num_total * train_frac)
    rng = np.random.RandomState(
        int(config.get('np_random_seed', 0)) + train_frac_seed
    )
    keep_idx = np.sort(rng.permutation(num_total)[:num_keep])
    cifar.train_data.xs = xs[keep_idx].copy()
    cifar.train_data.ys = ys[keep_idx].copy()
    cifar.train_data.num_examples = num_keep

# ------------------------------------------------------------
# [SMOKE TEST] tf.data pipeline must yield tensors
# ------------------------------------------------------------
train_ds = cifar.train_dataset(
    batch_size=batch_size,
    augment=True,
    shuffle=True,
    repeat=True
)
eval_ds = cifar.eval_dataset(batch_size=EVAL_BATCH_SIZE)

train_iter = iter(train_ds)

# ------------------------------------------------------------
# Dataset for TRAIN evaluation (no augmentation)
# ------------------------------------------------------------
train_eval_ds = tf.data.Dataset.from_tensor_slices(
    (cifar.train_data.xs, cifar.train_data.ys)
).batch(EVAL_BATCH_SIZE)

# ------------------------------------------------------------
# [SMOKE TEST] Learning-rate schedule wiring
# ------------------------------------------------------------
if not step_size_schedule:
    lr_schedule = 0.001
elif len(step_size_schedule) == 1:
    lr_schedule = float(step_size_schedule[0][1])
else:
    boundaries = [int(s[0]) for s in step_size_schedule][1:]
    values = [float(s[1]) for s in step_size_schedule]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries, values
    )

optimizer = tf.keras.optimizers.SGD(
    learning_rate=lr_schedule,
    momentum=momentum
)

# ------------------------------------------------------------
# [SMOKE TEST] Model must build + accept CIFAR input
# ------------------------------------------------------------
model = Model(
    mode='train',
    num_classes=10,
    weight_decay=weight_decay
)

# ------------------------------------------------------------
# [SMOKE TEST] Checkpointing must work (read/write)
# ------------------------------------------------------------
ckpt = tf.train.Checkpoint(
    step=tf.Variable(0, dtype=tf.int64),
    optimizer=optimizer,
    model=model
)
ckpt_manager = tf.train.CheckpointManager(
    ckpt, model_dir, max_to_keep=3
)

if ckpt_manager.latest_checkpoint:
    dummy_x = tf.zeros([1, 32, 32, 3], dtype=tf.float32)
    _ = model(dummy_x, training=False)

    zero_grads = [
        tf.zeros_like(v) for v in model.trainable_variables
    ]
    optimizer.apply_gradients(
        zip(zero_grads, model.trainable_variables)
    )

    status = ckpt.restore(ckpt_manager.latest_checkpoint)
    status.expect_partial()
    print("Restored from", ckpt_manager.latest_checkpoint)
else:
    print("No checkpoint found, starting from scratch")

start_step = int(ckpt.step.numpy())

# ------------------------------------------------------------
# [SMOKE TEST] PGD attack must run forward/backward
# ------------------------------------------------------------
attack = LinfPGDAttack(
    model,
    float(config['epsilon']),
    int(config['num_steps']),
    float(config['step_size']),
    bool(config['random_start']),
    config.get('loss_func', 'xent')
)

eval_attack = LinfPGDAttack(
    model,
    float(config['epsilon']),
    EVAL_NUM_STEPS,
    float(config['step_size']),
    bool(config['random_start']),
    config.get('loss_func', 'xent')
)

# ------------------------------------------------------------
# [SMOKE TEST] Summary writer must be writable
# ------------------------------------------------------------
summary_writer = tf.summary.create_file_writer(model_dir)

# ------------------------------------------------------------
# Loss + metrics
# ------------------------------------------------------------
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True,
    reduction=tf.keras.losses.Reduction.NONE
)
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_acc'
)

# ------------------------------------------------------------
# [SMOKE TEST] Single training step (forward + backward)
# ------------------------------------------------------------
@tf.function
def train_step(x_batch_adv, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch_adv, training=True)
        per_example_loss = loss_fn(y_batch, logits)
        loss = tf.reduce_mean(per_example_loss)
        if model.losses:
            loss += tf.add_n(model.losses)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y_batch, logits)

# ------------------------------------------------------------
# [SMOKE TEST] Main training loop
# ------------------------------------------------------------
print("Starting training from step", start_step)

# If this loop runs ~10 steps without error, the system is good
