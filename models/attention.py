import tensorflow as tf
from tensorflow.keras import layers, models
from .base import compile_model

class SimpleAttention(layers.Layer):
    def __init__(self, units=64, **kwargs):
        # accept kwargs so get_config/from_config can pass through
        super().__init__(**kwargs)
        self.units = units
        self.w = layers.Dense(units)
        self.u = layers.Dense(1)

    def call(self, x):
        # x shape: (batch, time_steps, channels)
        s = tf.nn.tanh(self.w(x))        # (batch, time_steps, units)
        s = self.u(s)                    # (batch, time_steps, 1)
        w = tf.nn.softmax(s, axis=1)     # (batch, time_steps, 1)
        # weighted sum across time_steps -> (batch, channels)
        return tf.reduce_sum(w * x, axis=1)

    def get_config(self):
        # this allows Keras to serialize / deserialize the layer
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # optional but explicit: ensures layer can be reconstructed
        return cls(**config)

def build_attention(input_dim, num_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim, 1))(inp)
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    att = SimpleAttention(64)(x)
    x = layers.Dense(128, activation='relu')(att)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return compile_model(models.Model(inp, out))
