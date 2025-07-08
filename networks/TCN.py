import tensorflow as tf

def TCN_block(filters, kernel_size, dilation_rate, dropout, input):
    residual = tf.keras.layers.Conv1D(1, 1)(input)
    # print('Residual shape:')
    # print(residual.shape)
    x1 = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)(input)  # _data)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Dropout(dropout)(x1)
    x1 = tf.keras.layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation_rate)(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)
    x1 = tf.keras.layers.Dropout(dropout)(x1)
    x1 = tf.keras.layers.Add()([residual, x1])
    # print('Block output shape:')
    # print(x1.shape)
    return x1


def create_TCN(input_window_length, n_blocks=4, kernel_sizes=[3], dilation_rate=[2, 4, 8, 16], filters=[32],
                 classes=2, lr=0.002, drop_out=0.1):
    input_data = tf.keras.Input(shape=(input_window_length, 1))
    residual = tf.keras.layers.Conv1D(1, 1)(input_data)

    x = tf.keras.layers.Conv1D(filters[0], kernel_sizes[0], padding="causal", dilation_rate=1)(input_data)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(drop_out)(x)
    x = tf.keras.layers.Conv1D(filters[0], kernel_sizes[0], padding="causal", dilation_rate=1)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(drop_out)(x)
    x = tf.keras.layers.Add()([residual, x])

    for filter in filters:
        for kernel_size in kernel_sizes:
            for i in range(n_blocks):
                print(dilation_rate[i])
                TCN_model = TCN_block(filter, kernel_size, dilation_rate[i], drop_out, input=x)
                x = TCN_model

    label_layer = tf.keras.layers.Dense(1024, activation="relu")(x)
    output_layer = tf.keras.layers.Dense(classes, activation="linear")(label_layer)
    model = tf.keras.Model(inputs=input_data, outputs=[output_layer],
                             name="TCN")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mae'])
    model.summary()
    return model