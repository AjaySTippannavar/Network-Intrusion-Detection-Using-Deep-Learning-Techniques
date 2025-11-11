from tensorflow.keras import layers, models
from .base import compile_model

def build_shallow(input_dim, num_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inp)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return compile_model(models.Model(inp, out))
