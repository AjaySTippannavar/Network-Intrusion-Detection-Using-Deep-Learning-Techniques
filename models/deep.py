from tensorflow.keras import layers, models
from .base import compile_model

def build_deep(input_dim, num_classes):
    inp = layers.Input(shape=(input_dim,))
    x = inp
    for n in [512,256,128]:
        x = layers.Dense(n, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return compile_model(models.Model(inp, out))
