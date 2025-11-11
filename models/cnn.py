from tensorflow.keras import layers, models
from .base import compile_model

def build_cnn(input_dim, num_classes):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Reshape((input_dim,1))(inp)
    x = layers.Conv1D(32,3,padding='same',activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64,3,padding='same',activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return compile_model(models.Model(inp, out))
