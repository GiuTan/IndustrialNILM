import tensorflow as tf
from tensorflow.keras import backend as K


def CRNN_block(x, kernel,drop_out,filters):
    conv_1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=(kernel, 1), strides=(1, 1), padding='same',
                                    kernel_initializer='glorot_uniform')(x)

    batch_norm_1 = tf.keras.layers.BatchNormalization()(conv_1, training=False)
    act_1 = tf.keras.layers.Activation('relu')(batch_norm_1)
    pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(1, 1))(act_1)
    drop_1 = tf.keras.layers.Dropout(drop_out)(pool_1)
    return drop_1


def CRNN_construction(window_size, lr=0.0, classes=0, drop_out = 1.0, kernel = 1, num_layers=1, gru_units=1):

    input_data = tf.keras.Input(shape=(window_size,1))
    x = tf.keras.layers.Reshape((window_size,1,1))(input_data)

    for i in range(num_layers):
        filters = 2 ** (i+5)
        CRNN = CRNN_block(x, kernel=kernel, drop_out=drop_out, filters=filters)
        x = CRNN


    spec_x = tf.keras.layers.Reshape((x.shape[1], x.shape[3]))(x)

    bi_direct = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=gru_units,return_sequences=True))(spec_x)

    instance_level = tf.keras.layers.Dense(units=classes)(bi_direct)

    frame_level = tf.keras.layers.Activation('linear', name = "strong_level")(instance_level)

    model_CRNN = tf.keras.Model(inputs=input_data, outputs=frame_level,
                                    name="CRNN")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model_CRNN.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['mae'])

    return model_CRNN


def CRNN_construction_final(window_size,initial_model, lr=0.0, classes=0, drop_out = 1.0, kernel = 1, num_layers=1, gru_units=1):

    input_data = tf.keras.Input(shape=(window_size, 1))
    x = tf.keras.layers.Reshape((window_size,1,1))(input_data)

    for i in range(num_layers):
        filters = 2 ** (i+5)
        CRNN = CRNN_block(x, kernel=kernel, drop_out=drop_out, filters=filters)
        x = CRNN


    spec_x = tf.keras.layers.Reshape((x.shape[1], x.shape[3]))(x)
    bi_direct = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=gru_units,return_sequences=True))(spec_x)
    model_CRNN = tf.keras.Model(inputs=input_data, outputs=bi_direct,
                                name="CRNN")

    model_CRNN.set_weights(initial_model[:24])

    if more == 'one':
        instance_level = tf.keras.layers.Dense(units=classes + 1)(model_CRNN.layers[-1].output)
    else:
        instance_level = tf.keras.layers.Dense(units=classes + 2)(model_CRNN.layers[-1].output)


    frame_level = tf.keras.layers.Activation('sigmoid', name = "strong_level")(instance_level)

    new_model_CRNN = tf.keras.Model(inputs=model_CRNN.inputs, outputs=frame_level,
                                name="CRNN")

    # new_model_CRNN.layers[0].trainable = False
    # new_model_CRNN.layers[1].trainable = False
    # new_model_CRNN.layers[2].trainable = True #
    # new_model_CRNN.layers[3].trainable = False
    # new_model_CRNN.layers[4].trainable = False
    # new_model_CRNN.layers[5].trainable = False
    # new_model_CRNN.layers[6].trainable = False
    # new_model_CRNN.layers[7].trainable = False  #
    # new_model_CRNN.layers[8].trainable = False
    # new_model_CRNN.layers[9].trainable = False
    # new_model_CRNN.layers[10].trainable = False
    # new_model_CRNN.layers[11].trainable = False
    # new_model_CRNN.layers[12].trainable = False #
    # new_model_CRNN.layers[13].trainable = False
    # new_model_CRNN.layers[14].trainable = False
    # new_model_CRNN.layers[15].trainable = False
    # new_model_CRNN.layers[16].trainable = False
    # new_model_CRNN.layers[17].trainable = False
    # new_model_CRNN.layers[18].trainable = False #
    # new_model_CRNN.layers[19].trainable = True #

    return new_model_CRNN


