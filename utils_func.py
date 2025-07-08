import numpy as np
import pandas as pd
import random as python_random
import matplotlib.pyplot as plt
from dependencies import *
import random
import tensorflow as tf
import os
from sklearn.metrics import mean_absolute_error as mae

def standardize_data(agg, mean, std):
    agg= agg -  mean
    agg /= std
    return agg


def read_multi_csv(i):
    return pd.read_csv(i)

def create_WAVENET(window_size):

    noise_mode = 0  # Noise mode: 0 - denoised (ensures output size is exactly len(app_inds))


    app_inds = [0, 1]  # Number of appliances to disaggregate

    # Network Parameters
    nb_filters = [512, 256, 256, 128, 128, 256, 256, 256,
                  512]  # Number of nodes per layer, use list for varying layer width.
    depth = 9  # Number of dilated convolutions per stack
    stacks = 1  # Number of dilated convolution stacks
    residual = False  # Whether network is residual or not (requires uniform number of nodes per layer, for now)
    use_bias = True  # Whether or not network uses bias
    activation = ReLU()  # activation - use function handle
    dropout = 0.1  # Dropout value

    sample_size = window_size  # YOUR "dimensione finestra"

    res_l2 = 0.  # l2 penalty weight

    MODEL = create_net(depth, sample_size, app_inds, nb_filters, use_bias, res_l2, residual, stacks, activation, noise_mode, dropout)
    return MODEL

def params_loading(device,approach):
    if device == 'milling':
        n_rows = 83168
    if device == 'pelletizer':
        n_rows = 271242
    model_path = 'models/' + approach + '_model__' + approach + '_' + device + '.weights.h5'
    return model_path,n_rows
def compute_sae(y_true, y_pred):
    total_error = np.abs(np.sum(y_true) - np.sum(y_pred)) #
    total_true = np.sum(y_true)
    return total_error / total_true
def setup_init():
    random.seed(123)
    np.random.seed(123)
    python_random.seed(123)
    tf.random.set_seed(1234)
    tf.experimental.numpy.random.seed(1234)
    os.environ['PYTHONHASHSEED'] = str(123)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
def data_load_window(data_path, device, n_rows,window_size):
    data_h1 = pd.read_csv(data_path + device + '.csv', sep=';', nrows=n_rows)
    data_test = pd.read_csv(data_path + device + '.csv', sep=';', skiprows=n_rows)
    data_h1 = [data_h1[i:i + window_size] for i in range(0, len(data_h1) - window_size, 1)]
    pad_h1 = window_size - len(data_h1[-1])
    data_h1[-1] = np.pad(data_h1[-1], pad_width=((0, pad_h1), (0, 0)), mode='constant')

    data_test = [data_test[i:i + window_size] for i in range(0, len(data_test) - window_size, window_size)]

    data_train = data_h1
    data_val = data_test

    x_train = np.delete(data_train, [0, 2, 3], 2)
    y_train = np.delete(data_train, [0, 1], 2)
    x_val = np.delete(data_val, [0, 2, 3], 2)
    y_val = np.delete(data_val, [0, 1], 2)
    x_test = x_val
    y_test = y_val

    print('DATA NORMALIZATION')
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    train_max = y_train.max()
    train_min = y_train.min()

    print("Mean train")
    print(train_mean)
    print("Std train")
    print(train_std)

    print("Max train")
    print(train_max)
    print("Min train")
    print(train_min)

    x_train = standardize_data(x_train, train_mean, train_std)
    x_val = standardize_data(x_val, train_mean, train_std)
    x_test = standardize_data(x_test, train_mean, train_std)

    y_train = (y_train - train_min) / (train_max - train_min)
    y_val = (y_val - train_min) / (train_max - train_min)
    y_test = (y_test - train_min) / (train_max - train_min)

    x_train = tf.convert_to_tensor(x_train, dtype=tf.float64)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float64)
    x_val = tf.convert_to_tensor(x_val, dtype=tf.float64)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float64)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float64)
    y_test = y_test
    return x_train, y_train, x_val, y_val, x_test, y_test, train_max

def model_setting(approach,model_path,training,type_, x_train, y_train, x_val,y_val,window_size,output_dim,batch_size):

    if approach == 'WAVENET':
        MODEL = create_WAVENET(window_size)
    if approach == 'CNN':
        MODEL = create_CNN(window_size,output_dim)
    if approach == 'TCN':
        MODEL = create_TCN(window_size,output_dim)
    if approach == 'CRNN':
        drop = 0.1
        kernel = 5
        num_layers = 3
        gru_units = 64
        lr = 0.002

        MODEL = CRNN_construction(window_size, lr=lr, classes=output_dim, drop_out=drop, kernel=kernel,
                                  num_layers=num_layers, gru_units=gru_units)
    if approach == 'BERT':
        MODEL = create_BERT(num_layers=2, d_model=128, num_heads=4, dff=512, window_size=window_size,num_appliances=2, rate=0.1)

    if approach == 'LSTM':
        MODEL = create_LSTM(window_size, output_dim)
    if training:
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mae', mode='min', patience=10,
                                                      restore_best_weights=True)
        history_ = MODEL.fit(x=x_train, y=y_train, shuffle=True, epochs=1000, batch_size=batch_size,
                             validation_data=(np.array(x_val), np.array(y_val)), callbacks=[early_stop], verbose=1)
        MODEL.save_weights('models/' + approach + '_model_' + type_ + '.weights.h5')
    else:
        MODEL.load_weights(model_path)
    return MODEL


def testing(MODEL,x_test,y_test,device,approach,train_max):
    y_pred = MODEL.predict(x=x_test)

    y_test = y_test.reshape(y_test.shape[0] * y_test.shape[1], y_test.shape[2]) * train_max
    y_pred = y_pred.reshape(y_pred.shape[0] * y_pred.shape[1], y_pred.shape[2]) * train_max

    y_pred[y_pred < 0] = 0
    np.save('outputs/'+approach + '_'+ device +'_predictions.npy',y_pred)
    np.save('outputs/'+device+'ground_truth.npy',y_test)
    print("EVALUATION METRICS:")
    error = mae(y_test, y_pred)
    error_sae = compute_sae(y_test, y_pred)


    print("Mean absolute error : " + str(error))
    print("SAE:" + str(error_sae))


    plt.plot(y_test[:4000, 0])
    plt.plot(y_pred[:4000, 0])
    plt.ylabel('Active Power')
    plt.xlabel('Samples')
    plt.legend(['Ground-truth', 'Predictions'])
    plt.title(approach + '_' + device + '_I')
    plt.show()


    plt.plot(y_test[:4000, 1])
    plt.plot(y_pred[:4000, 1])
    plt.ylabel('Active Power')
    plt.xlabel('Samples')
    plt.legend(['Ground-truth', 'Predictions'])
    plt.title(approach + '_' + device + '_II')
    plt.show()
