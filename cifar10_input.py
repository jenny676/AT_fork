# cifar10_input.py
"""
Minimal CIFAR-10 loader compatible with older repos expecting
tensorflow.examples.tutorials style helpers.

It provides:
 - CIFAR10Data(data_path)
   with attributes: train_data, eval_data
 - Each of train_data / eval_data has:
   - xs : numpy array of shape [N, 32, 32, 3] (float32)
   - ys : numpy array of shape [N] (int labels)
   - num_examples : int
   - train_data.get_next_batch(batch_size, multiple_passes=True)
     returns (xs_batch, ys_batch)

 - AugmentedCIFAR10Data(raw_cifar, sess, model) returns a wrapper
   with the same attributes. This implementation uses simple random
   horizontal flips and random cropping when `get_next_batch` is called.
"""

import os
import numpy as np
from tensorflow.keras.datasets import cifar10

def _one_hot(labels, num_classes=10):
    n = labels.shape[0]
    out = np.zeros((n, num_classes), dtype=np.int32)
    out[np.arange(n), labels.flatten()] = 1
    return out

class _Dataset:
    def __init__(self, xs, ys, shuffle=True):
        # store as float32 in original 0-255; training script may expect that
        self.xs = xs.astype(np.float32)
        self.ys = ys.astype(np.int64).flatten()
        self.num_examples = self.xs.shape[0]
        self._shuffle = shuffle
        self._perm = np.arange(self.num_examples)
        self._cur = 0
        if shuffle:
            np.random.shuffle(self._perm)

    def get_next_batch(self, batch_size, multiple_passes=True):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        start = self._cur
        end = start + batch_size
        if end <= self.num_examples:
            idx = self._perm[start:end]
            self._cur = end
        else:
            # end > num_examples
            if not multiple_passes:
                # return the remainder then raise or pad; here return remainder
                idx = self._perm[start:self.num_examples]
                self._cur = self.num_examples
            else:
                # wrap-around: collect remainder and reshuffle for next epoch
                idx1 = self._perm[start:self.num_examples]
                if self._shuffle:
                    np.random.shuffle(self._perm)
                take = batch_size - idx1.shape[0]
                idx2 = self._perm[0:take]
                idx = np.concatenate([idx1, idx2], axis=0)
                self._cur = take
        return self.xs[idx].copy(), self.ys[idx].copy()

class CIFAR10Data:
    def __init__(self, data_path=None):
        # data_path ignored for now (could be used for caching)
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        # keep original uint8->float32 behavior (the training code may expect 0-255 floats)
        # If you want normalization adapt here (e.g., /255.0)
        self.train_data = _Dataset(x_train, y_train, shuffle=True)
        self.eval_data = _Dataset(x_test, y_test, shuffle=False)

class AugmentedCIFAR10Data:
    """
    Lightweight wrapper that provides the same interface but performs
    cheap augmentations (random crop + horizontal flip) on training batches.
    """

    def __init__(self, raw_cifar, sess=None, model=None, padding=4):
        # raw_cifar is expected to be CIFAR10Data instance
        self.train_data = raw_cifar.train_data
        self.eval_data = raw_cifar.eval_data
        self._padding = padding
        # we keep pointers to sess/model for API-compat if some code needs them
        self.sess = sess
        self.model = model

    def _random_crop_and_flip(self, batch):
        # batch: numpy array shape [B, 32, 32, 3], dtype float32
        B, H, W, C = batch.shape
        pad = self._padding
        if pad == 0:
            out = batch
        else:
            padded = np.pad(batch, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='reflect')
            out = np.zeros_like(batch)
            for i in range(B):
                top = np.random.randint(0, 2*pad + 1)
                left = np.random.randint(0, 2*pad + 1)
                out[i] = padded[i, top:top+H, left:left+W, :]
        # random horizontal flip
        flips = np.random.rand(B) < 0.5
        out[flips] = out[flips, :, ::-1, :]
        return out

    # Expose train_data.get_next_batch but with augmentation
    def get_next_batch(self, batch_size, multiple_passes=True):
        x_batch, y_batch = self.train_data.get_next_batch(batch_size, multiple_passes=multiple_passes)
        x_batch = self._random_crop_and_flip(x_batch)
        return x_batch, y_batch

    # For code that expects to call cifar.train_data.get_next_batch(...)
    # allow forwarding:
    @property
    def train_data(self):
        return self._train_data

    @train_data.setter
    def train_data(self, val):
        self._train_data = val

    @property
    def eval_data(self):
        return self._eval_data

    @eval_data.setter
    def eval_data(self, val):
        self._eval_data = val
