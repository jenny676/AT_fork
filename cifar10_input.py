# cifar10_input.py -- TF2-friendly CIFAR10 loader + tf.data pipelines
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def _one_hot(labels, num_classes=10):
    n = labels.shape[0]
    out = np.zeros((n, num_classes), dtype=np.int32)
    out[np.arange(n), labels.flatten()] = 1
    return out

class _Dataset:
    def __init__(self, xs, ys, shuffle=True):
        # keep pixel-valued images in 0..255 as float32 for parity with your TF1 code
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
            if not multiple_passes:
                idx = self._perm[start:self.num_examples]
                self._cur = self.num_examples
            else:
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
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        # keep original uint8->float32 behavior (0..255)
        self.train_data = _Dataset(x_train, y_train, shuffle=True)
        self.eval_data = _Dataset(x_test, y_test, shuffle=False)

class AugmentedCIFAR10Data:
    """
    Wrapper preserving the old API (get_next_batch) but also provides
    tf.data.Dataset pipelines. Inputs are kept in 0..255 float32 to match legacy code.
    """
    def __init__(self, raw_cifar, padding=4):
        # raw_cifar is expected to be CIFAR10Data instance
        self._raw = raw_cifar
        self._padding = padding

        # expose train_data / eval_data properties for backwards compatibility
        self.train_data = raw_cifar.train_data
        self.eval_data = raw_cifar.eval_data

    # --- backward-compatible augmentation used by get_next_batch (numpy-based)
    def _random_crop_and_flip_numpy(self, batch):
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
        flips = np.random.rand(B) < 0.5
        out[flips] = out[flips, :, ::-1, :]
        return out

    # Preserve the old get_next_batch API that returns augmented numpy arrays
    def get_next_batch(self, batch_size, multiple_passes=True):
        x_batch, y_batch = self._raw.train_data.get_next_batch(batch_size, multiple_passes=multiple_passes)
        x_batch = self._random_crop_and_flip_numpy(x_batch)
        return x_batch, y_batch

    # --- TF2-friendly dataset pipelines (recommended)
    def _preprocess_for_train(self, image, label):
        """
        image: tf.uint8 or tf.float32 in 0..255
        returns: image tf.float32 in 0..255 (no normalization), label int32
        Performs: random pad+crop and random horizontal flip using TF ops.
        """
        # Ensure float32
        image = tf.cast(image, tf.float32)
        pad = self._padding
        if pad > 0:
            # pad then random crop
            image = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]], mode='REFLECT')
            image = tf.image.random_crop(image, size=[32, 32, 3])
        # random flip
        image = tf.image.random_flip_left_right(image)
        return image, tf.cast(label, tf.int32)

    def _preprocess_for_eval(self, image, label):
        image = tf.cast(image, tf.float32)
        return image, tf.cast(label, tf.int32)

    def train_dataset(self, batch_size, augment=True, shuffle=True, repeat=True):
        """
        Returns a tf.data.Dataset yielding (image, label) pairs.
        Image dtype: tf.float32 in 0..255 (matches the TF1 pattern).
        """
        xs = self._raw.train_data.xs
        ys = self._raw.train_data.ys
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        if shuffle:
            ds = ds.shuffle(buffer_size=self._raw.train_data.num_examples, seed=None)
        if augment:
            ds = ds.map(self._preprocess_for_train, num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = ds.map(self._preprocess_for_eval, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        if repeat:
            ds = ds.repeat()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def eval_dataset(self, batch_size):
        xs = self._raw.eval_data.xs
        ys = self._raw.eval_data.ys
        ds = tf.data.Dataset.from_tensor_slices((xs, ys))
        ds = ds.map(self._preprocess_for_eval, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.
