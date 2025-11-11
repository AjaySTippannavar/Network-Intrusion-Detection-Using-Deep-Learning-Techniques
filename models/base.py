import tensorflow as tf
from tensorflow.keras import layers, models

def compile_model(model, lr=1e-3):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
