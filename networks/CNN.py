import tensorflow as tf

def create_CNN(input_window_length,num_appliances):

    input_layer = tf.keras.layers.Input(shape=(input_window_length,))


    reshape_layer = tf.keras.layers.Reshape((input_window_length, 1))(input_layer)


    conv_layers = [
        tf.keras.layers.Conv1D(filters=30, kernel_size=10, padding="same", activation="relu"),
        tf.keras.layers.Conv1D(filters=30, kernel_size=8, padding="same", activation="relu"),
        tf.keras.layers.Conv1D(filters=40, kernel_size=6, padding="same", activation="relu"),
        tf.keras.layers.Conv1D(filters=50, kernel_size=5, padding="same", activation="relu"),
        tf.keras.layers.Conv1D(filters=50, kernel_size=5, padding="same", activation="relu")
    ]

    x = reshape_layer
    for layer in conv_layers:
        x = layer(x)


    label_layer = tf.keras.layers.Dense(1024, activation="relu")(x)
    output_layer = tf.keras.layers.Dense(num_appliances, activation="linear")(label_layer)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                       metrics=['mae'])
    model.summary()
    return model

