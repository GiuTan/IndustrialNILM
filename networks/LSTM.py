import tensorflow as tf
from tensorflow.keras import backend as K
import os


def create_LSTM(window_size,classes):
    input_data = tf.keras.Input(shape=(window_size, 1))
    conv_1 = tf.keras.layers.Conv1D(filters=16, kernel_size=4, strides=1, padding='same',
                                    kernel_initializer='glorot_uniform')(input_data)
    act_1 = tf.keras.layers.Activation('linear')(conv_1)
    bi_direct = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128,return_sequences=True))(act_1)
    bi_direct_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=256,return_sequences=True))(bi_direct)
    dense_1 = tf.keras.layers.Dense(units=128, activation='tanh')(bi_direct_1)
    instance_level = tf.keras.layers.Dense(units=classes)(dense_1)
    output_level = tf.keras.layers.Activation('linear', name="strong_level")(instance_level)
    model_LSTM = tf.keras.Model(inputs=input_data, outputs=output_level,
                                name="LSTM")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    model_LSTM.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                       metrics=['mae'])

    return model_LSTM


