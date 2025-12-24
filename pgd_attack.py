# pgd_attack.py -- TF2 version of LinfPGDAttack
import numpy as np
import tensorflow as tf

class LinfPGDAttack:
    def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func='xent'):
        """
        TF2 Linf PGD attack.

        Args:
          model: a tf.keras.Model or callable that takes (x, training=False) and returns logits.
          epsilon: float in [0,1] (e.g., 8/255).
          num_steps: number of PGD steps.
          step_size: step size in [0,1] (e.g., 2/255).
          random_start: bool, whether to initialize with random noise inside the linf ball.
          loss_func: 'xent' or 'cw' (untargeted). 'xent' maximizes cross-entropy; 'cw' uses margin loss.
        """
        self.model = model
        self.epsilon = float(epsilon)
        self.num_steps = int(num_steps)
        self.step_size = float(step_size)
        self.random_start = bool(random_start)
        assert loss_func in ('xent', 'cw'), "loss_func must be 'xent' or 'cw'"
        self.loss_func = loss_func

    def _compute_loss(self, logits, labels):
        """
        Compute per-example loss to maximize (higher = more likely to be adversarial).
        For untargeted attack we maximize loss.
        """
        # logits: [B, num_classes], labels: [B]
        if self.loss_func == 'xent':
            # per-example cross-entropy (not reduced)
            per_ex = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # we maximize cross-entropy, so return per_ex
            return per_ex
        else:  # cw
            # construct one-hot
            num_classes = tf.shape(logits)[1]
            one_hot = tf.one_hot(labels, num_classes, dtype=logits.dtype)
            correct_logit = tf.reduce_sum(one_hot * logits, axis=1)
            # max over other logits
            INF = 1e4
            wrong_logits = logits - one_hot * INF
            max_wrong = tf.reduce_max(wrong_logits, axis=1)
            # CW loss: maximize (correct - wrong + kappa) negative; we want high value when misclassified
            # Use margin = correct - max_wrong; maximize -(margin) => minimize margin, so return -relu(...)
            margin = correct_logit - max_wrong
            # We'll use negative margin as "loss to maximize" with some margin constant; adjust if needed
            return -tf.nn.relu(margin + 50.0)  # kept +50 like original scale (can be tuned)

    @tf.function
    def _pgd_single(self, x_nat, y, eps, step, num_steps, rand_init):
        """
        Run PGD starting from an initial x (x_nat or randomly perturbed).
        This function expects tensors and returns the final adversarial tensor and per-example loss.
        All tensors are float32 and in the same scale as x_nat provided (i.e., if x_nat in 0..255 use that).
        """
        # x_nat, y: tensors
        # eps, step: scalars (same units as x_nat)
        # rand_init: bool
        # initialize delta
        if rand_init:
            delta = tf.random.uniform(tf.shape(x_nat), -eps, eps, dtype=x_nat.dtype)
        else:
            delta = tf.zeros_like(x_nat)

        # make sure within valid pixel range
        x_adv = tf.clip_by_value(x_nat + delta, 0.0, tf.reduce_max(x_nat) if tf.reduce_max(x_nat) > 1.0 else 1.0)
        delta = x_adv - x_nat

        # iterative update
        for _ in tf.range(num_steps):
            with tf.GradientTape() as tape:
                tape.watch(delta)
                cur = x_nat + delta
                logits = self.model(cur, training=False)
                per_ex_loss = self._compute_loss(logits, y)  # shape [B]
                loss = tf.reduce_mean(per_ex_loss)  # scalar just for gradient
            # grad w.r.t. delta
            grad = tape.gradient(loss, delta)
            # step: sign gradient ascent (we maximize loss)
            signed_grad = tf.sign(grad)
            delta = delta + step * signed_grad
            # project to linf ball and valid pixel range
            delta = tf.clip_by_value(delta, -eps, eps)
            x_adv = tf.clip_by_value(x_nat + delta, 0.0, tf.reduce_max(x_nat) if tf.reduce_max(x_nat) > 1.0 else 1.0)
            delta = x_adv - x_nat
        # compute final logits and per-example loss
        final_logits = self.model(x_nat + delta, training=False)
        final_per_ex_loss = self._compute_loss(final_logits, y)
        # return adversarial examples and per-example loss
        return x_nat + delta, final_per_ex_loss

    def perturb(self, x_nat, y, restarts=1):
        """
        Produce adversarial examples for x_nat, y.

        Args:
          x_nat: numpy array or tf.Tensor, shape [B,H,W,C]. dtype uint8 or float32.
                 Values may be in 0..1 or 0..255. Returns will be in same scale as x_nat input but dtype float32.
          y: numpy array or tf.Tensor, shape [B] integer labels.
          restarts: number of random restarts. We'll keep the best (largest loss) adversarial per example.

        Returns:
          x_best: numpy array float32 shape [B,H,W,C]
        """
        # convert inputs to tensors
        x_np = x_nat
        y_np = y
        # Detect scale and convert input to float32
        x = tf.convert_to_tensor(x_np)
        if x.dtype != tf.float32:
            x = tf.cast(x, tf.float32)
        # Determine scale: if any pixel > 1, assume 0..255
        maxv = tf.reduce_max(x)
        scale = tf.cond(maxv > 1.0, lambda: tf.constant(255.0, dtype=tf.float32), lambda: tf.constant(1.0, dtype=tf.float32))

        # convert eps and step to pixel units consistent with x's scale
        eps_pixels = tf.cast(self.epsilon, tf.float32) * scale
        step_pixels = tf.cast(self.step_size, tf.float32) * scale

        y_t = tf.convert_to_tensor(y_np, dtype=tf.int32)

        # We'll run `restarts` independent PGD runs and pick the per-example adversarial with maximum per-example loss.
        # Prepare accumulators on host (numpy) to keep best adv and best loss
        x_best = x.numpy().astype(np.float32).copy()  # start with natural as fallback
        # initialize best_losses to -inf so that any real loss will be larger
        best_losses = np.full((x.shape[0],), -np.inf, dtype=np.float32)

        # For each restart, run PGD and evaluate per-example loss; keep best per-example by loss
        for r in range(restarts):
            # choose random init flag for this restart (use the configured random_start)
            rand_init = self.random_start

            # run pgd single (this is compiled tf.function)
            x_adv_tensor, per_ex_loss = self._pgd_single(x, y_t, eps_pixels, step_pixels, self.num_steps, rand_init)

            # convert to numpy
            x_adv_np = x_adv_tensor.numpy().astype(np.float32)
            loss_np = per_ex_loss.numpy().astype(np.float32)

            # Update per-example bests where this restart produced higher loss
            mask = loss_np > best_losses
            if np.any(mask):
                x_best[mask] = x_adv_np[mask]
                best_losses[mask] = loss_np[mask]

        # final clipping to valid range (preserve scale)
        x_best = np.clip(x_best, 0.0, (255.0 if np.max(x_best) > 1.0 else 1.0)).astype(np.float32)
        return x_best


