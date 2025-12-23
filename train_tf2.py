# train_tf2.py -- end-to-end TF2 training script
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

# ---- hyperparams / config values
EVAL_NUM_STEPS = int(config.get('eval_num_steps', 20))
EVAL_RESTARTS = int(config.get('eval_restarts', 5))
EVAL_BATCH_SIZE = int(config.get('eval_batch_size', 256))

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

model_dir = config['model_dir']
os.makedirs(model_dir, exist_ok=True)
# copy config for record
try:
    import shutil
    shutil.copy('config.json', model_dir)
except Exception:
    pass

# ---- imports of local modules (assume they are in same dir)
from model import Model
from pgd_attack import LinfPGDAttack
import cifar10_input

# ---- dataset setup
raw_cifar = cifar10_input.CIFAR10Data(data_path)
cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, padding=4)

# apply train_frac if necessary (slice numpy arrays)
if not (0 < train_frac <= 1.0):
    raise ValueError("train_frac must be in (0,1]")
if train_frac < 1.0:
    xs = cifar.train_data.xs
    ys = cifar.train_data.ys
    num_total = xs.shape[0]
    num_keep = int(num_total * train_frac)
    rng = np.random.RandomState(int(config.get('np_random_seed', 0)) + train_frac_seed)
    keep_idx = np.sort(rng.permutation(num_total)[:num_keep])
    cifar.train_data.xs = xs[keep_idx].copy()
    cifar.train_data.ys = ys[keep_idx].copy()
    cifar.train_data.num_examples = num_keep

# Build tf.data datasets (images are float32 in 0..255)
train_ds = cifar.train_dataset(batch_size=batch_size, augment=True, shuffle=True, repeat=True)
eval_ds = cifar.eval_dataset(batch_size=EVAL_BATCH_SIZE)

train_iter = iter(train_ds)

# ---- model, optimizer, lr schedule
# Convert step_size_schedule into boundaries/values for PiecewiseConstantDecay
# step_size_schedule is list of [step, lr] pairs
if len(step_size_schedule) == 0:
    boundaries = []
    values = [0.001]
else:
    boundaries = [int(sss[0]) for sss in step_size_schedule][1:]
    values = [sss[1] for sss in step_size_schedule]

# ---- learning rate setup (robust to 1-entry schedules)
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
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum)

model = Model(mode='train', num_classes=10, weight_decay=weight_decay)

# ----- checkpointing
ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                           optimizer=optimizer,
                           model=model)
ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
if ckpt_manager.latest_checkpoint:
    # 1) Build model variables (if not already built)
    dummy_x = tf.zeros([1, 32, 32, 3], dtype=tf.float32)
    _ = model(dummy_x, training=False)

    # 2) Create optimizer slot variables (momentum, etc.)
    zero_grads = [tf.zeros_like(v) for v in model.trainable_variables]
    optimizer.apply_gradients(zip(zero_grads, model.trainable_variables))

    # 3) Now restore — optimizer slots exist, so TF can map them
    status = ckpt.restore(ckpt_manager.latest_checkpoint)
    status.expect_partial()  # optional but safe
    print("Restored from", ckpt_manager.latest_checkpoint)
else:
    print("No checkpoint found, starting from scratch")
start_step = int(ckpt.step.numpy())

# ---- attack (training + eval)
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

# ---- summary writer
summary_writer = tf.summary.create_file_writer(model_dir)

# ---- metrics helpers (files)
METRICS_CSV = os.path.join(model_dir, "metrics.csv")
METRICS_JSONL = os.path.join(model_dir, "metrics.jsonl")

def append_metrics(out_dir, metrics):
    fname = os.path.join(out_dir, "metrics.jsonl")
    with open(fname, "a") as fh:
        fh.write(json.dumps(metrics) + "\n")

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

init_metrics_csv()

# ---- loss and metrics objects
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')

# ---- training step (expects x_batch_adv as tensor and y_batch)
@tf.function
def train_step(x_batch_adv, y_batch):
    with tf.GradientTape() as tape:
        logits = model(x_batch_adv, training=True)
        per_example_loss = loss_fn(y_batch, logits)
        loss = tf.reduce_mean(per_example_loss)
        # include model-registered weight decay (kernel_regularizer)
        if model.losses:
            loss = loss + tf.add_n(model.losses)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss_metric.update_state(loss)
    train_acc_metric.update_state(y_batch, logits)

# ---- evaluation helper (computes clean + robust metrics over dataset)
def compute_dataset_metrics(dataset, attack_obj, batch_size, restarts):
    """
    dataset: tf.data.Dataset yielding (x, y) (both tensors)
    attack_obj: LinfPGDAttack instance
    Returns: (clean_loss, clean_acc, robust_loss, robust_acc, elapsed_seconds)
    """
    t0 = time.time()
    total_clean_loss = 0.0
    total_robust_loss = 0.0
    total_corr_clean = 0
    total_corr_robust = 0
    N = 0

    for x_batch, y_batch in dataset:
        # convert shapes to numpy counts
        B = x_batch.shape[0]
        N += int(B)

        # run clean forward
        logits = model(x_batch, training=False)
        per_ex_loss = loss_fn(y_batch, logits).numpy()
        preds = np.argmax(logits.numpy(), axis=1)
        total_corr_clean += int(np.sum(preds == y_batch.numpy()))
        total_clean_loss += float(np.sum(per_ex_loss))

        # create adversarial examples (attack.perturb returns numpy array)
        x_batch_adv = attack_obj.perturb(x_batch, y_batch, restarts=restarts)
        # convert to tensor for evaluation
        if not isinstance(x_batch_adv, tf.Tensor):
            x_batch_adv = tf.convert_to_tensor(x_batch_adv, dtype=tf.float32)

        logits_adv = model(x_batch_adv, training=False)
        per_ex_loss_adv = loss_fn(y_batch, logits_adv).numpy()
        preds_adv = np.argmax(logits_adv.numpy(), axis=1)
        total_corr_robust += int(np.sum(preds_adv == y_batch.numpy()))
        total_robust_loss += float(np.sum(per_ex_loss_adv))

    # average
    return (
        total_clean_loss / float(N),
        float(total_corr_clean) / float(N),
        total_robust_loss / float(N),
        float(total_corr_robust) / float(N),
        time.time() - t0
    )

# ---- training loop
# epoch-local training timer (reset each epoch)
epoch_train_time = 0.0
step = start_step
start_time_total = time.time()

# -------------------------
# Epoch bookkeeping
# -------------------------
steps_per_epoch = int(cifar.train_data.num_examples // batch_size)
if steps_per_epoch <= 0:
    raise ValueError("batch_size is larger than training set size")

print("Starting training from step", step)
while step < max_num_training_steps:
    # fetch next batch
    x_batch, y_batch = next(train_iter)  # tensors: x float32 in 0..255, y int32
    # produce adversarial examples using attack (may return numpy)
    x_batch_adv = attack.perturb(x_batch, y_batch, restarts=1)
    if not isinstance(x_batch_adv, tf.Tensor):
        x_batch_adv = tf.convert_to_tensor(x_batch_adv, dtype=tf.float32)

    # per-step timing (accumulate into epoch_train_time)
    t0 = timer()
    # run train_step (tf.function)
    train_step(x_batch_adv, y_batch)
    step_duration = timer() - t0
    epoch_train_time += step_duration

    step = step + 1
    ckpt.step.assign(step)

    # output to console (keeps your existing output cadence)
    if step % num_output_steps == 0:
        print(f"Step {step}: {datetime.now()}")
        print(f"  train loss (streaming): {train_loss_metric.result().numpy():.6f}")
        print(f"  train acc (streaming): {train_acc_metric.result().numpy():.4%}")

    # summaries (keep as-is)
    if step % num_summary_steps == 0:
        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss_metric.result(), step=step)
            tf.summary.scalar('train_acc', train_acc_metric.result(), step=step)
            summary_writer.flush()

    # checkpointing (keep as-is)
    if step % num_checkpoint_steps == 0:
        saved = ckpt_manager.save()
        print("Saved checkpoint:", saved)

    # ----- EPOCH boundary: when we have completed steps_per_epoch training steps -----
    if step % steps_per_epoch == 0:
        epoch = step // steps_per_epoch
        print(f"=== End of epoch {epoch} (step {step}). Running epoch-level evaluations ===")

        # 1) Compute train metrics (clean + robust) using training attack
        train_clean_loss, train_clean_acc, train_robust_loss, train_robust_acc, train_eval_time = compute_dataset_metrics(
            train_eval_ds, attack, batch_size=EVAL_BATCH_SIZE, restarts=1
        )

        # 2) Compute test metrics (clean + robust)
        num_eval_examples = int(config.get('num_eval_examples', 1000))
        eval_ds_full = cifar.eval_dataset(batch_size=eval_batch_size)
        eval_ds_limited = eval_ds_full.take(int(math.ceil(num_eval_examples / eval_batch_size)))

        test_clean_loss, test_clean_acc, test_robust_loss, test_robust_acc, test_eval_time = compute_dataset_metrics(
            eval_ds_limited, eval_attack, batch_size=eval_batch_size, restarts=EVAL_RESTARTS
        )

        # current lr
        lr_val = float(lr_schedule(step)) if hasattr(lr_schedule, '__call__') else float(values[-1])

        # Build metrics dict & append CSV/JSONL — use epoch_train_time (per-epoch)
        metrics = {
            "epoch": int(epoch),
            "train_time": float(epoch_train_time),
            "test_time": float(test_eval_time),
            "lr": lr_val,
            "train_loss": float(train_clean_loss),
            "train_acc": float(train_clean_acc),
            "train_robust_loss": float(train_robust_loss),
            "train_robust_acc": float(train_robust_acc),
            "test_loss": float(test_clean_loss),
            "test_acc": float(test_clean_acc),
            "test_robust_loss": float(test_robust_loss),
            "test_robust_acc": float(test_robust_acc)
        }

        append_metrics_json_line(metrics)
        append_metrics_csv([
            metrics["epoch"],
            metrics["train_time"],
            metrics["test_time"],
            metrics["lr"],
            metrics["train_loss"],
            metrics["train_acc"],
            metrics["train_robust_loss"],
            metrics["train_robust_acc"],
            metrics["test_loss"],
            metrics["test_acc"],
            metrics["test_robust_loss"],
            metrics["test_robust_acc"]
        ])

        # optional: print epoch summary
        print(f"Epoch {epoch} summary: lr={lr_val:.6g} train_loss={train_clean_loss:.4f} train_acc={train_clean_acc:.4%} "
              f"train_robust_acc={train_robust_acc:.4%} test_robust_acc={test_robust_acc:.4%}")

        # Reset streaming metrics
        train_loss_metric.reset_state()
        train_acc_metric.reset_state()

        # Reset epoch-local timer for next epoch
        epoch_train_time = 0.0

    # loop continues until max_num_training_steps

# ---- evaluation at end (or you can schedule periodic evals)
print("Running evaluation ...")
# For evaluation we will use a dataset limited by config['num_eval_examples']
num_eval_examples = int(config.get('num_eval_examples', 1000))
eval_batch_size = int(config.get('eval_batch_size', 256))
# slice eval dataset to requested number of examples
eval_ds_full = cifar.eval_dataset(batch_size=eval_batch_size)
# take enough batches to cover num_eval_examples
eval_ds_limited = eval_ds_full.take(int(math.ceil(num_eval_examples / eval_batch_size)))

clean_loss, clean_acc, robust_loss, robust_acc, eval_time = compute_dataset_metrics(
    eval_ds_limited, eval_attack, batch_size=eval_batch_size, restarts=EVAL_RESTARTS
)

print(f"Eval results (on {num_eval_examples} examples):")
print(f"  clean loss: {clean_loss:.6f}, clean acc: {clean_acc:.4%}")
print(f"  robust loss: {robust_loss:.6f}, robust acc: {robust_acc:.4%}")
print(f"  eval time: {eval_time:.1f}s")

# log metrics
metrics = {
    "epoch": None,
    "train_time": epoch_train_time,  # last epoch duration
    "test_time": eval_time,
    "lr": float(lr_schedule(step)) if hasattr(lr_schedule, '__call__') else float(values[-1]),
    "train_loss": None,
    "train_acc": None,
    "train_robust_loss": None,
    "train_robust_acc": None,
    "test_loss": clean_loss,
    "test_acc": clean_acc,
    "test_robust_loss": robust_loss,
    "test_robust_acc": robust_acc
}
# append to CSV/JSONL
append_metrics_json_line = lambda d: open(METRICS_JSONL, "a").write(json.dumps(d) + "\n")
append_metrics_json_line(metrics)
append_metrics_csv([
    metrics["epoch"],
    metrics["train_time"],
    metrics["test_time"],
    metrics["lr"],
    metrics["train_loss"],
    metrics["train_acc"],
    metrics["train_robust_loss"],
    metrics["train_robust_acc"],
    metrics["test_loss"],
    metrics["test_acc"],
    metrics["test_robust_loss"],
    metrics["test_robust_acc"]
])

# final save
ckpt_manager.save()
print("Training complete. Total time:", time.time() - start_time_total)
