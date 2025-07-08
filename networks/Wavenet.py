import time
#import input_generator  # Assumi che questo modulo sia compatibile o modificato per i tuoi dati
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras import metrics
from tensorflow.keras.activations import *
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import numpy as np
import datetime, os, pickle
# from sacred import Experiment
# import os
# from sacred.observers import FileStorageObserver



def base_config():

    val_spl = 0.1  # split ration for validation, note that currently, validation is used for testing as well.

    agg_ind = [
        0]  # Location of aggregate measurement in source file (now a single channel, e.g., the 0-th column if your data is just [aggregate_power])
    noise_mode = 0  # Noise mode: 0 - denoised (ensures output size is exactly len(app_inds))
    noise_scale = 10  # Weight of noise for noise mode 2 (not used with noise_mode=0)

    app_inds = [0, 1]  # Which appliances to disaggregate (MUST BE 2 FOR YOUR (None, window_size, 2) output)

    # Network Parameters
    nb_filters = [512, 256, 256, 128, 128, 256, 256, 256,
                  512]  # Number of nodes per layer, use list for varying layer width.
    depth = 9  # Number of dilated convolutions per stack
    stacks = 1  # Number of dilated convolution stacks
    residual = False  # Whether network is residual or not (requires uniform number of nodes per layer, for now)
    use_bias = True  # Whether or not network uses bias
    activation = ReLU()  # activation - use function handle
    callbacks = [LearningRateScheduler(scheduler, verbose=1)]  # Callbacks for training
    dropout = 0.1  # Dropout value
    mask = True  # Whether to use masked output

    # Training Parameters:
    n_epochs = 300  # Number of training epochs
    batch_size = 50  # Number of samples per batch, each sample will have sample_size timesteps
    sample_size = 512  # YOUR "dimensione finestra"
    savepath = '../data/comparison'  # Folder to save models during/after training.
    save_flag = False # Flag to save best model at each iteration of cross validation
    shuffle = True  # Shuffle samples every epoch
    verbose = 2  # Printing verbositiy, because of sacred, use only 0 or 2
    res_l2 = 0.  # l2 penalty weight
    use_receptive_field_only = True  # Whether or not to ignore the samples without full input receptive field
    loss = tf.keras.losses.MeanSquaredError() # Loss, use function handle (Mean Absolute Error, common for NILM)
    all_metrics = [estimated_accuracy]  # Metrics, use function handle
    optimizer = {}  # Placeholder for optimizer
    cross_validate = True  # Cross validation parameters
    splice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Number of splices for cross-validation
    use_metric_per_app = False  # Whether to display metrics for each appliances separately


def adam():
    optimizer = {
        'optimizer': 'adam',
        'params':
            {'lr': 0.001,
             'beta_1': 0.9,
             'beta_2': 0.999,
             'decay': 0.,
             'amsgrad': False,
             'epsilon': 1e-8}

    }



def sgd():
    optimizer = {
        'optimizer': 'sgd',
        'params': {
            'lr': 0.01,
            'momentum': 0.9,
            'decay': 0.,
            'nesterov': True,
            'epsilon': None
        }
    }


def estimated_accuracy(y_true, y_pred):
    ' NILM metric as described in paper'
    return 1 - K.sum(K.abs(y_pred - y_true)) / (K.sum(y_true) + K.epsilon()) / 2


def scheduler(epoch, curr_lr):
    ' Learning rate scheduler '
    if epoch < 50:
        pass
    elif (epoch + 1) % 10 == 0:
        curr_lr = curr_lr * 0.98
    return (curr_lr)


"""
def get_meter_max(input_path, data_source):
    ' Find the "meter maximum value" - the closest power of 2 to the maximum aggregate measurement.'
    ' This is like saying we set the bit-depth of our measurements so that the maximum power still fits'
    # MODIFICA: Questa funzione ora deve leggere il TUO formato di dati.
    # Se il tuo 'data_source' è un file pickle con solo il segnale aggregato
    # nella prima colonna, allora andrà bene. Altrimenti, potresti doverla adattare.
    # Per l'obiettivo (None, finestra, 1), ci aspettiamo un solo canale di input.

    # Esempio semplificato: se il tuo file 'my_nilm_data.dat' contiene solo il segnale aggregato:
    try:
        with open(input_path + '/' + data_source, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Warning: Data file '{data_source}' not found at '{input_path}'. Returning default meter_max.")
        return [2 ** np.ceil(np.log2(2000))]  # Ritorna un valore di default sensato per la potenza (es. 2000W)

    # Assumendo che 'data' sia un numpy array o simile e che la prima colonna
    # sia la potenza aggregata.
    # Se data.shape[-1] è il numero di colonne e la potenza aggregata è la prima:
    if data.ndim == 2:  # Se il formato è (timesteps, num_channels)
        # Assumiamo che la colonna di input sia la prima (indice 0)
        meter_max_val = 2 ** np.ceil(np.log2(data[:, 0].max()))
    elif data.ndim == 1:  # Se il formato è (timesteps,)
        meter_max_val = 2 ** np.ceil(np.log2(data.max()))
    else:
        raise ValueError("Unsupported data format in get_meter_max. Expected 1D or 2D array.")

    return [meter_max_val]  # Ritorna una lista con un singolo valore
    """

def make_class_specific_loss(loss, cl_ind, name):
    'Wrap any loss or meteric so that it gives the loss only for one output dimension, meaning one sub-meter'

    def wrapper(y_true, y_pred):
        y_true = K.expand_dims(y_true[:, :, cl_ind], axis=2)
        y_pred = K.expand_dims(y_pred[:, :, cl_ind], axis=2)

        return loss(y_true, y_pred)

    wrapper.__name__ = loss.__name__ + name

    return wrapper



def get_class_weights(noise_mode, app_inds, noise_scale):
    'Give class weights so that if we are in noise mode 2  (see netdata.py for details)'
    ' we still get balanced class weights. Otherwise - give equal weight to each class.'
    ' Can be altered for unbalanced dataset to allow better training'

    if noise_mode == 2:  # Se noise_mode è 2, app_inds + 1 canale per il rumore
        cw = np.ones((1, len(app_inds)))
        cw = np.append(cw, 1 / noise_scale)
        cw = cw / cw.sum()
        return cw
    else:  # Per noise_mode=0 o 1, il numero di output è len(app_inds)
        return np.ones((1, len(app_inds))) / len(app_inds)



def extract_and_duplicate(tensor, reps=1, batch_size=0, sample_size=0):
    'Copy input once for every class for multiplication.'
    'Consider reimplementing to save memeory'

    # MODIFICA: Assicurati che il reshaping sia corretto per l'input (batch_size, sample_size, 1)
    # tensor è già (batch_size, sample_size, 1) se agg_ind = [0]
    if tensor.shape[-1] != 1:
        raise ValueError(f"Expected input tensor to have last dim of 1, but got {tensor.shape[-1]}. Check agg_ind.")

    # `tensor` è (batch_size, sample_size, 1)
    # `reps` sarà `len(app_inds)+noise_mode//2`, che per noi è 2
    if reps > 1:
        tensor = Concatenate()([tensor for _ in range(reps)])
    return tensor


# Funzioni non usate in questo script, ma presenti nel tuo originale, le conservo.
# Se `kl`, `abs_error`, `target_power`, `adj_estimated_accuracy`, `adj_abs_error`
# non sono definiti altrove, dovrai definirli o rimuoverli.
def kl(y_true, y_pred):
    # Dummy implementation if not defined elsewhere
    return K.mean(K.square(y_true - y_pred))


def abs_error(y_true, y_pred):
    # Dummy implementation if not defined elsewhere
    return K.mean(K.abs(y_true - y_pred))


def target_power(y_true, y_pred):
    # Dummy implementation if not defined elsewhere
    return K.mean(y_true)


def adj_estimated_accuracy(y_true, y_pred):
    # Dummy implementation if not defined elsewhere
    return estimated_accuracy(y_true, y_pred)


def adj_abs_error(y_true, y_pred):
    # Dummy implementation if not defined elsewhere
    return abs_error(y_true, y_pred)


def save_network_copy(out_path, in_path):
    'save a waveNILM copy'
    # Assicurati che queste custom_objects siano definite o non usate se non necessarie
    custom_objects_dict = {
        'kl': kl,
        'estimated_accuracy': estimated_accuracy,
        'abs_error': abs_error,
        'target_power': target_power,
        'adj_estimated_accuracy': adj_estimated_accuracy,
        'adj_abs_error': adj_abs_error
    }
    inmodel = load_model(in_path, custom_objects=custom_objects_dict)
    w_in = inmodel.get_weights()
    inmodel.summary()
    model = create_net()
    model.compile('adam', 'mae')
    model.summary()
    try:
        model.set_weights(w_in)
    except:
        raise BaseException('Input model and output model don''t have same shape')

    model.save(out_path)



def create_net(depth, sample_size, app_inds, nb_filters, use_bias, res_l2, residual, stacks, activation,
               noise_mode, dropout):
    # Create WaveNILM network

    meter_max = 1  # Ora ritorna [un_singolo_valore]

    # If constant amount of convolution kernels - create a list
    if len(nb_filters) == 1:
        nb_filters = np.ones(depth, dtype='int') * nb_filters[0]

    # Input layer: (batch_size, sample_size, 1)
    inpt = Input(shape=(sample_size,1))  # len(meter_max) sarà 1

    # Initial Feature mixing layer
    out = Conv1D(nb_filters[0], 1, padding='same', use_bias=use_bias, kernel_regularizer=l2(res_l2))(inpt)

    skip_connections = [out]

    # Create main wavenet structure
    for j in range(stacks):
        for i in range(depth):
            # "Signal" output
            signal_out = Conv1D(nb_filters[i], 2, dilation_rate=2 ** i, padding='causal',
                                use_bias=use_bias, kernel_regularizer=l2(res_l2))(out)
            signal_out = activation(signal_out)

            # "Gate" output
            gate_out = Conv1D(nb_filters[i], 2, dilation_rate=2 ** i, padding='causal',
                              use_bias=use_bias, kernel_regularizer=l2(res_l2))(out)
            gate_out = sigmoid(gate_out)

            # Multiply signal by gate to get gated output
            gated = Multiply()([signal_out, gate_out])

            # Create residual if desired, note that currently this can only be supported for entire network at once
            # Consider changing residual to 2 lists - split and mearge, and check for each layer individually
            if residual:
                # Making copies of previous layer nodes if the number of filters  doesn't match
                prev_ind = max(i - 1, 0)
                if not nb_filters[i] == nb_filters[prev_ind]:
                    # ATTENZIONE: Questa logica di reshaping `nb_filter[i]/nb_filter[prev_ind]`
                    # assume che `nb_filter[i]` sia un multiplo di `nb_filter[prev_ind]`.
                    # Assicurati che `nb_filters` sia progettato di conseguenza.
                    # Se `nb_filters` non è uniforme o non multiplo, questa riga potrebbe dare problemi.
                    # Per il tuo caso d'uso, se `residual=False`, non è un problema.
                    out = Lambda(extract_and_duplicate,
                                 arguments={'reps': nb_filters[i] / nb_filters[prev_ind], 'batch_size': batch_size,
                                            'sample_size': sample_size})(out)  # Corretto nb_filter -> nb_filters

                # Creating residual
                out = Add()([out, gated])
            else:
                out = gated

            # Droupout for regularization
            if dropout != 0:
                out = Dropout(dropout)(out)
            skip_connections.append(out)

    out = Concatenate()(skip_connections)

    # Masked output final layer
    mask = False
    if mask:
        # Create copies of desired input property (power, current, etc.) for multiplication with mask
        # reps sarà len(app_inds) + 0 (dato che noise_mode=0), quindi 2
        pre_mask = Lambda(extract_and_duplicate,
                          arguments={'reps': len(app_inds) + noise_mode // 2, 'batch_size': batch_size,
                                     'sample_size': sample_size})(inpt)
        # Create mask
        mask = TimeDistributed(Dense(len(app_inds) + noise_mode // 2, activation='tanh'))(out)
        # Multiply with mask
        out = Multiply()([pre_mask, mask])

    ## Optional residual mask instead of multiplicative mask
    # out = Add()([pre_mask,mask])
    # out = LeakyReLU(alpha = 0.1)(out)

    # Standard output final layer
    else:  # Se mask è False, usa una Dense lineare
        out = TimeDistributed(Dense(len(app_inds) + noise_mode // 2, activation='linear'))(out)
        out = LeakyReLU(alpha=0.1)(out)

    model = Model(inpt, out)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mae'])
    model.summary()

    return model



def make_optimizer(optimizer, params):
    # Optimizer setup using sacred prefix
    if optimizer == 'sgd':
        optim = SGD(**params)
    elif optimizer == 'adam':
        optim = Adam(**params)
    else:
        raise ValueError('Invalid config for optimizer.optimizer: ' + optimizer)
    return optim



def compile_model(model, use_receptive_field_only, loss, all_metrics, noise_mode, use_metric_per_app, app_inds):
    # Fix cost and metrics according to skip_out_of_receptive_field and compile model

    optim = make_optimizer()

    # Skipping any inputs that may contain zero padded inputs for loss calculation (and performance evaluation)
    if use_receptive_field_only:
        loss = skip_out_of_receptive_field(loss)
        all_metrics = [skip_out_of_receptive_field(m) for m in all_metrics]

    # creating specific copy of each metric for each appliance
    if use_metric_per_app and len(app_inds) > 1:
        ln = len(all_metrics)
        for i in range(len(app_inds)):
            name = '_for_appliance_%d' % app_inds[i]
            for j in range(ln):
                all_metrics.append(make_class_specific_loss(all_metrics[j], i, name))

    model.compile(loss=loss, metrics=all_metrics, optimizer=optim)



def reset_weights(model, lr):

    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            if isinstance(layer.kernel_initializer, tf.compat.v1.initializers.Initializer):
                layer.kernel.initializer.run(session=session)
            else:
                layer.kernel.initializer.__call__(shape=layer.kernel.shape, dtype=layer.kernel.dtype)
        if hasattr(layer, 'bias_initializer'):
            if isinstance(layer.bias_initializer, tf.compat.v1.initializers.Initializer):
                layer.bias.initializer.run(session=session)
            else:
                layer.bias.initializer.__call__(shape=layer.bias.shape, dtype=layer.bias.dtype)

    K.set_value(model.optimizer.lr, lr)



def data_splice(splice_ind, effective_sample_size, data_len, val_spl, sample_size, batch_size):

    val_len = val_spl * ((data_len - sample_size) // effective_sample_size)

    val_len = val_len // batch_size * batch_size
    val_start = int(np.floor(splice_ind * val_len))
    val_end = int(np.floor(splice_ind * val_len + val_len))
    val_ind = np.arange(val_start, val_end, dtype='int')
    trn_ind = np.delete(np.arange((data_len - sample_size) // effective_sample_size, dtype='int'), val_ind)

    return trn_ind, val_ind



def compute_receptive_field(depth, stacks):
    # Calculate the receptive field for the given WaveNet parameters.
    # For kernel_size=2 and dilation rates 2^0, 2^1, ..., 2^(depth-1)
    # Receptive field = sum_{i=0}^{depth-1} (kernel_size - 1) * 2^i + 1 (for the initial 1x1 conv)
    # = sum_{i=0}^{depth-1} 1 * 2^i + 1
    # = (2^depth - 1) + 1 = 2^depth
    # If there are multiple stacks, the total receptive field is approximately stacks * 2^depth
    # This formula (stacks*2**depth) assumes the receptive field grows linearly with stacks,
    # which is true if the output of one stack feeds into the next, and each stack has its own internal dilations.
    return (stacks * 2 ** depth)



def skip_out_of_receptive_field(func):

    receptive_field = compute_receptive_field()

    def wrapper(y_true, y_pred):

        if receptive_field > y_true.shape[1]:

            print(
                f"Warning: Receptive field ({receptive_field}) larger than sample_size ({y_true.shape[1]}). Clipping to sample_size.")
            start_idx = 0
        else:
            start_idx = receptive_field - 1

        y_true = y_true[:, start_idx:, :]
        y_pred = y_pred[:, start_idx:, :]
        return func(y_true, y_pred)

    wrapper.__name__ = func.__name__

    return wrapper
