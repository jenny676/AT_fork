# model.py -- ResNet-18 (TF2 / tf.keras)
import tensorflow as tf
from tensorflow.keras import layers, regularizers

class Model(tf.keras.Model):
    """ResNet-18 for CIFAR-10 as a tf.keras.Model (returns logits)."""

    def __init__(self, mode='train', num_classes=10, weight_decay=2e-4):
        """
        Args:
          mode: 'train' or 'eval' (affects BatchNorm behavior when calling with training flag).
          num_classes: number of classes (10 for CIFAR-10).
          weight_decay: L2 factor applied as kernel_regularizer on conv/dense layers.
        """
        super().__init__(name='resnet18_cifar')
        assert mode in ('train', 'eval')
        self.mode = mode
        self.num_classes = num_classes
        self.wd = weight_decay

        # Initial conv
        self.init_conv = layers.Conv2D(
            64, 3, strides=1, padding='same',
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(stddev=0.0),
            kernel_regularizer=regularizers.l2(self.wd),
            name='init_conv'
        )
        self.init_bn = layers.BatchNormalization(name='init_bn')

        # Build four stages. For ResNet-18, blocks per stage = [2,2,2,2]
        self.stage_filters = [64, 128, 256, 512]
        self.stage_blocks = [2, 2, 2, 2]

        # For each block we create conv/bn layers. We'll store them in lists for re-use.
        self.stage_layers = []  # list of lists: each stage holds list of block dicts
        in_channels = 64
        for stage_idx, (out_ch, num_blocks) in enumerate(zip(self.stage_filters, self.stage_blocks)):
            blocks = []
            for block_idx in range(num_blocks):
                stride = 1
                if stage_idx > 0 and block_idx == 0:
                    stride = 2  # downsample at first block of stage > 0

                # First conv of block
                conv1 = layers.Conv2D(
                    out_ch, 3, strides=stride, padding='same',
                    use_bias=False,
                    kernel_regularizer=regularizers.l2(self.wd),
                    name=f'stage{stage_idx+1}_block{block_idx}_conv1'
                )
                bn1 = layers.BatchNormalization(name=f'stage{stage_idx+1}_block{block_idx}_bn1')

                # Second conv of block
                conv2 = layers.Conv2D(
                    out_ch, 3, strides=1, padding='same',
                    use_bias=False,
                    kernel_regularizer=regularizers.l2(self.wd),
                    name=f'stage{stage_idx+1}_block{block_idx}_conv2'
                )
                bn2 = layers.BatchNormalization(name=f'stage{stage_idx+1}_block{block_idx}_bn2')

                # Shortcut conv if channel or stride changes
                if in_channels != out_ch or stride != 1:
                    shortcut_conv = layers.Conv2D(
                        out_ch, 1, strides=stride, padding='same',
                        use_bias=False,
                        kernel_regularizer=regularizers.l2(self.wd),
                        name=f'stage{stage_idx+1}_block{block_idx}_shortcut'
                    )
                    shortcut_bn = layers.BatchNormalization(
                        name=f'stage{stage_idx+1}_block{block_idx}_shortcut_bn'
                    )
                else:
                    shortcut_conv = None
                    shortcut_bn = None

                blocks.append({
                    'conv1': conv1, 'bn1': bn1,
                    'conv2': conv2, 'bn2': bn2,
                    'shortcut_conv': shortcut_conv, 'shortcut_bn': shortcut_bn,
                    'stride': stride, 'in_channels': in_channels, 'out_channels': out_ch
                })

                in_channels = out_ch  # after first block, in_channels becomes out_ch

            self.stage_layers.append(blocks)

        # Final layers
        self.final_bn = layers.BatchNormalization(name='final_bn')
        # No activation here - will apply relu then global avg pool
        self.classifier = layers.Dense(
            self.num_classes,
            kernel_regularizer=regularizers.l2(self.wd),
            name='fc'
        )

    def _per_image_standardize(self, images):
        # images expected in pixel range [0, 255] as float32
        # tf.image.per_image_standardization supports per-image whitening
        return tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)

    def call(self, inputs, training=False):
        """
        Forward pass.
        Args:
          inputs: tensor shape [B,32,32,3], dtype float32 (pixel values 0..255 is OK)
          training: boolean, True in training mode (affects BatchNorm)
        Returns:
          logits tensor shape [B, num_classes]
        """
        x = tf.cast(inputs, tf.float32)
        # keep same behavior as original: per-image standardization
        x = self._per_image_standardize(x)

        # initial conv
        x = self.init_conv(x)
        x = self.init_bn(x, training=training)
        x = tf.nn.relu(x)

        # Stages
        for stage_blocks in self.stage_layers:
            for block in stage_blocks:
                shortcut = x
                # pre-activation style follows original: bn -> relu -> conv
                # sub1
                x1 = block['bn1'](x, training=training)
                x1 = tf.nn.relu(x1)
                x1 = block['conv1'](x1)

                # sub2
                x1 = block['bn2'](x1, training=training)
                x1 = tf.nn.relu(x1)
                x1 = block['conv2'](x1)

                # handle shortcut if needed
                if block['shortcut_conv'] is not None:
                    shortcut = block['shortcut_conv'](shortcut)
                    shortcut = block['shortcut_bn'](shortcut, training=training)

                x = x1 + shortcut

        # post
        x = self.final_bn(x, training=training)
        x = tf.nn.relu(x)
        # global average pool
        x = tf.reduce_mean(x, axis=[1, 2])  # shape [B, C]
        logits = self.classifier(x)
        return logits

    def compute_weight_decay_loss(self):
        """Return L2 losses from kernel_regularizer (sum of model.losses)."""
        # Keras layers append kernel_regularizer losses into model.losses
        if self.losses:
            return tf.math.add_n(self.losses)
        return tf.constant(0.0)
