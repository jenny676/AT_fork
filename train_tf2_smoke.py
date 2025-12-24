#!/usr/bin/env python3
"""
train_smoke_runner.py
A minimal smoke-test runner that reads config_smoke.json and exercises:
 - loading config
 - importing local model/attack/dataset modules
 - building model and optimizer
 - running a few train steps (forward + backward)
 - running a tiny eval pass
 - writing a single JSONL metrics line to model_dir

Intended for quick verification only. Does not try to fully reproduce the main training script.
"""

import json
import os
import time
from timeit import default_timer as timer
import math

import numpy as np
import tensorflow as tf

# --- config filename (uses config_smoke.json)
CFG_FILE = "config_smoke.json"

# --- load config (fail fast if missing)
if not os.path.exists(CFG_FILE):
    raise SystemExit(f"Missing {CFG_FILE} in current directory. Please create it (see sample provided).")

with open(CFG_FILE, "r") as fh:
    cfg = json.load(fh)

# --- required / optional config keys (with defaults for smoke)
model_dir = cfg.get("model_dir", "./smoke_model_dir")
os.makedirs(model_dir, exist_ok=True)

data_path = cfg.get("data_path", "cifar10_data")
training_batch_size = int(cfg.get("training_batch_size", 32))
train_frac = float(cfg.get("train_frac", 0.01))
train_frac_seed = int(cfg.get("train_frac_seed", 0))
max_num_training_steps = int(cfg.get("max_num_training_steps", 5))
tf_random_seed = int(cfg.get("tf_random_seed", 0))
np_random_seed = int(cfg.get("np_random_seed", 0))

# attack / optimizer friendly defaults
epsilon = float(cfg.get("epsilon", 8/255))
num_steps = int(cfg.get("num_steps", 1))
step_size = float(cfg.get("step_size", 2/255))
random_start = bool(cfg.get("random_start", False))
momentum = float(cfg.get("momentum", 0.9))
weight_decay = float(cfg.get("weight_decay", 0.0))

EVAL_BATCH_SIZE = int(cfg.get("eval_batch_size", 16))
EVAL_NUM_STEPS = int(cfg.get("eval_num_steps", 1))
EVAL_RESTARTS = int(cfg.get("eval_restarts", 0))

# seeds
tf.random.set_seed(tf_random_seed)
np.random.seed(np_random_seed)

print("Smoke config:", {
    "model_dir": model_dir,
    "data_path": data_path,
    "batch_size": training_batch_size,
    "max_steps": max_num_training_steps,
})

# --- import local modules (must exist in same folder or PYTHONPATH)
try:
    from model import Model
    from pgd_attack import LinfPGDAttack
    import cifar10_input
except Exception as e:
    raise SystemExit(f"Failed to import local modules: {e}")

# --- dataset setup
raw_cifar = cifar10_input.CIFAR10Data(data_path)
cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, padding=4)

# optional subsample for speed
if not (0 < train_frac <= 1.0):
    raise ValueError("train_frac must be in (0,1]")
if train_frac < 1.0:
    xs = cifar.train_data.xs
    ys = cifar.train_data.ys
    num_total = xs.shape[0]
    num_keep = int(max(1, num_total * train_frac))
    rng = np.random.RandomState(np_random_seed + train_frac_seed)
    keep_idx = np.sort(rng.permutation(num_total)[:num_keep])
    cifar.train_data.xs = xs[keep_idx].copy()
    cifar.train_data.ys = ys[keep_idx].copy()
    cifar.train_data.num_examples = num_keep
    print(f"Subsampled training set -> {num_keep} examples")

train_ds = cifar.train_dataset(batch_size=training_batch_size, augment=True, shuffle=True, repeat=True)
train_iter = iter(train_ds)
train_eval_ds = tf.data.Dataset.from_tensor_slices(
    (cifar.train_data.xs, cifar.train_data.ys)
).batch(EVAL_BATCH_SIZE)

# --- model, optimizer
model = Model(mode='train', num_classes=10, weight_decay=weight_decay)
# ensure model variables are created
_ = model(tf.zeros([1, 32, 32, 3], dtype=tf.float32), training=False)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=momentum)

# --- checkpoint (minimal)
ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=2)

# --- attack
attack = LinfPGDAttack(model, epsilon, num_steps, step_size, random_start, cfg.get("loss_func", "xent"))
eval_attack = LinfPGDAttack(model, epsilon, EVAL_NUM_STEPS, step_size, random_start, cfg.get("loss_func", "xent"))

# --- loss & simple metrics
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

@tf.function
def train_step(x_batch_adv, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch_adv, training=True)
        per_example_loss = loss_fn(y_batch, logits)
        loss = tf.reduce_mean(per_example_loss)
        if model.losses:
            loss = loss + tf.add_n(model.losses)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y_batch, logits)

def compute_dataset_metrics(dataset, attack_obj, restarts):
    total_loss = 0.0
    total_corr = 0
    N = 0
    for x_batch, y_batch in dataset:
        B = x_batch.shape[0]
        N += int(B)
        logits = model(x_batch, training=False)
        per_ex_loss = loss_fn(y_batch, logits).numpy()
        preds = np.argmax(logits.numpy(), axis=1)
        total_corr += int(np.sum(preds == y_batch.numpy()))
        total_loss += float(np.sum(per_ex_loss))

        # adversarial pass (quick)
        x_adv = attack_obj.perturb(x_batch, y_batch, restarts=restarts)
        if not isinstance(x_adv, tf.Tensor):
            x_adv = tf.convert_to_tensor(x_adv, dtype=tf.float32)
        logits_adv = model(x_adv, training=False)
        # we won't accumulate robust metrics here — smoke only
    if N == 0:
        return (0.0, 0.0, 0.0)
    return (total_loss / float(N), float(total_corr) / float(N), float(N))

# --- metrics file helpers
METRICS_JSONL = os.path.join(model_dir, "smoke_metrics.jsonl")
def append_metrics(metrics):
    with open(METRICS_JSONL, "a") as fh:
        fh.write(json.dumps(metrics) + "\n")

# --- quick training loop
print("Starting smoke training for", max_num_training_steps, "steps")
start_time = time.time()
step = int(ckpt.step.numpy())

for _ in range(max_num_training_steps):
    x_batch, y_batch = next(train_iter)
    x_batch_adv = attack.perturb(x_batch, y_batch, restarts=1)
    if not isinstance(x_batch_adv, tf.Tensor):
        x_batch_adv = tf.convert_to_tensor(x_batch_adv, dtype=tf.float32)

    t0 = timer()
    train_step(x_batch_adv, y_batch)
    step_time = timer() - t0

    step += 1
    ckpt.step.assign(step)

    print(f"Step {step}: loss={train_loss_metric.result().numpy():.6f} acc={train_acc_metric.result().numpy():.4%} step_time={step_time:.3f}s")

# --- tiny eval (on a few training examples)
clean_loss, clean_acc, n_examples = compute_dataset_metrics(train_eval_ds.take(1), eval_attack, restarts=EVAL_RESTARTS)
print(f"Smoke eval -> examples={n_examples} clean_loss={clean_loss:.6f} clean_acc={clean_acc:.4%}")

# --- write a single metrics line
metrics = {
    "timestamp": time.time(),
    "steps_ran": int(step),
    "clean_loss": float(clean_loss),
    "clean_acc": float(clean_acc),
    "n_examples_eval": int(n_examples),
    "total_time_s": time.time() - start_time
}
append_metrics(metrics)
print("Wrote smoke metrics to", METRICS_JSONL)

# --- checkpoint save (best-effort)
try:
    ckpt_manager.save()
    print("Checkpoint saved to model_dir.")
except Exception as e:
    print("Checkpoint save failed (non-fatal):", e)

print("Smoke test finished successfully.")

