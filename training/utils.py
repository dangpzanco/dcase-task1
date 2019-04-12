# Standard libraries
import os
import pathlib
import pickle
from datetime import datetime

# Scientific stack
import numpy as np
import numpy.random as rnd
import pandas as pd
import sklearn.metrics as skmetrics

# Chunked data
import dask
import zarr

# Enable multiprocessing support for Zarr
from numcodecs import blosc
blosc.use_threads = False

# Temporary fix for HDF5 multiprocessing error 11 
# Source: https://github.com/keras-team/keras/issues/11101#issuecomment-459350086
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Pretty progress bar
import tqdm
import keras_tqdm

# Tensorflow config
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
# config = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=6)
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))

# Keras
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D
from keras import regularizers

# Custom modules
import dataset_generator as dgen


# Source: https://stackoverflow.com/a/48393723/2801287
class TrainValTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = str(pathlib.Path(log_dir) / 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = str(pathlib.Path(log_dir) / 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


class MetadataModelCheckPoint(keras.callbacks.ModelCheckpoint):
    """docstring for MetadataModelCheckPoint"""
    def __init__(self, timenow, custom_dict=None, **kwargs):
        super(MetadataModelCheckPoint, self).__init__(**kwargs)

        self.custom_dict = custom_dict or {}

        # Set metadata filename
        self.meta_path = pathlib.Path(self.filepath).with_suffix('.partial.pkl')

        if self.meta_path.exists():
            # Get saved metadata
            with open(str(self.meta_path), 'rb') as handle:
                meta = pickle.load(handle)

            # Set logs_keys
            self.logs_keys = list(meta['logs'].keys())
        else:
            # Metadata dictonary
            meta = {'logs': None,
                    'custom_dict': self.custom_dict,
                    'timestamp': timenow,
                    'epochs': 0,
                    'elapsed_time': 0}

            # Save metadata
            with open(str(self.meta_path), 'wb') as output:
                pickle.dump(meta, output)

    def on_train_begin(self, logs=None):
        self.last_epoch_time = datetime.now()

    # def on_epoch_begin(self, epoch, logs=None):
    #     print(dir(self))
    #     print(self.params)
    #     print(type(self.validation_data))
    
    def on_epoch_end(self, epoch, logs=None):

        # Get saved metadata
        with open(str(self.meta_path), 'rb') as handle:
            meta = pickle.load(handle)

        # Update metadata
        if meta['logs'] is None:
            meta['logs'] = {}
            self.logs_keys = list(logs.keys())
            for key in self.logs_keys:
                meta['logs'][key] = [logs[key]]
        else:
            for key in self.logs_keys:
                meta['logs'][key].append(logs[key])
        meta['epochs'] += 1
        meta['elapsed_time'] += (datetime.now() - self.last_epoch_time).total_seconds()

        # Check for a possible bug
        assert len(meta['logs'][key]) == meta['epochs']
        
        # Save metadata
        with open(str(self.meta_path), 'wb') as output:
            pickle.dump(meta, output)

        # Set the epoch timestamp
        self.last_epoch_time = datetime.now()

        # Pass the logs to `ModelCheckpoint.on_epoch_end`
        super(MetadataModelCheckPoint, self).on_epoch_end(epoch, logs)


def get_partial_models(folder):

    folder = pathlib.Path(folder)
    meta_list = []

    for item in folder.glob('*.h5'):
        # Check if final metadata file exists (meaning the model finished training)
        if not item.with_suffix('.pkl').exists():    
            # Get saved metadata
            with open(str(item.with_suffix('.partial.pkl')), 'rb') as handle:
                meta = pickle.load(handle)
            meta_list.append([item, meta])

    return meta_list


def finish_training(model_path, partial_meta):
    # Load Model
    model = keras.models.load_model(str(model_path))

    # Set `train_model_generator()` input parameters
    params = partial_meta['custom_dict']
    params['initial_epoch'] = partial_meta['epochs']
    params['timenow'] = partial_meta['timestamp']
    params['model'] = model

    train_model_generator(**params)


def baseline_dcase2018(data_shape, normalization=True, dropout=True, dropout_dense=True):
    
    # Input Layer
    model_inputs = Input(shape=data_shape)

    # Conv Layer #1
    x = Conv2D(filters=32, kernel_size=(7,7), padding='same', activation='linear')(model_inputs)
    if normalization:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(5,5))(x)
    if dropout:
        x = Dropout(0.3)(x)

    # Conv Layer #2
    x = Conv2D(filters=32, kernel_size=(7,7), padding='same', activation='linear')(x)
    if normalization:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4,100))(x)
    if dropout:
        x = Dropout(0.3)(x)

    # MLP Layer
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    if dropout:
        x = Dropout(0.3)(x)

    # Output Layer
    predictions = Dense(10, activation='softmax', activity_regularizer=None)(x)

    # Build model from the input and output objects
    model = Model(inputs=model_inputs, outputs=predictions)

    return model


def best_model(data_shape, normalization=True, dropout=True, dropout_dense=True, 
    dropout_rate=0.3, last_pool=16, dense_size=100, activation='softmax'):
    
    # Input Layer
    model_inputs = Input(shape=data_shape)

    # Conv Layer #1
    x = Conv2D(filters=32, kernel_size=(1,7), padding='same', activation='linear')(model_inputs)
    if normalization:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(1,5))(x)
    if dropout:
        x = Dropout(dropout_rate)(x)

    # Conv Layer #2
    x = Conv2D(filters=64, kernel_size=(1,7), padding='same', activation='linear')(x)
    if normalization:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(1,5))(x)
    if dropout:
        x = Dropout(dropout_rate)(x)

    # Conv Layer #3
    x = Conv2D(filters=32, kernel_size=(1,7), padding='same', activation='linear')(x)
    if normalization:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(1,last_pool))(x)
    if dropout:
        x = Dropout(dropout_rate)(x)

    # MLP Layer
    x = Flatten()(x)
    x = Dense(dense_size, activation='relu')(x)
    if dropout and dropout_dense:
        x = Dropout(dropout_rate)(x)

    # Output Layer
    predictions = Dense(10, activation=activation, activity_regularizer=None, name='predictions')(x)

    # Build model from the input and output objects
    model = Model(inputs=model_inputs, outputs=predictions)

    return model


def best_model_old(data_shape, normalization=True, dropout=True, dropout_dense=True, 
    dropout_rate=0.3, last_pool=16, dense_size=100):

    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(1,7),
    padding='same', input_shape=data_shape, activation='linear'))
    if normalization:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,5)))
    if dropout:
        model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(filters=64, kernel_size=(1,7), activation='linear', padding='same'))
    if normalization:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,5)))
    if dropout:
        model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(filters=32, kernel_size=(1,7), activation='linear', padding='same'))
    if normalization:
        model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,last_pool)))
    if dropout:
        model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(dense_size, activation='relu'))
    if dropout and dropout_dense:
        model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation='softmax', name='predictions'))

    return model


def train_model_generator(training_generator, validation_generator, model, model_name, use_multiprocessing=True,
    workers=6, epochs=100, epoch_patience=20, monitor='val_acc', logdir='logs', model_dir='saved_models', 
    timenow=None, initial_epoch=0):
    """
    Train a Keras model.

    Parameters
    ----------
    training_generator : dataset_generator.DataGenerator
        Training set generator.

    validation_generator : dataset_generator.DataGenerator
        Validation (test) set generator.

    model : Keras Sequential Model
        Pre-compiled Keras model object.

    model_name : string
        Model filename.

    use_multiprocessing : bool
        Check to use multi processing for the generator.
        Default value True

    workers : int
        Number of threads for the generator.
        Default value 6

    epochs : int
        Maximum number of epoch.
        Default value 100

    epoch_patience : int
        Early stopping: "number of epochs with no improvement after which training will be stopped".
        Default value 20

    logdir : string
        Tensorboard log directory.
        Default value 'logs'

    model_dir : string
        Trained model HDF5 directory.
        Default value 'saved_models'

    """

    # Save input parameters (used on MetadataModelCheckPoint)
    params = dict(training_generator=training_generator, 
        validation_generator=validation_generator, 
        model_name=model_name, 
        use_multiprocessing=use_multiprocessing, 
        workers=workers, 
        epochs=epochs, 
        epoch_patience=epoch_patience, 
        monitor=monitor, 
        logdir=logdir, 
        model_dir=model_dir)

    # Get timestamp and append to model name
    if timenow is None:
        timenow = datetime.now()
        timestamp = timenow.isoformat().replace(':','_')
        model_name += timestamp
        model._name = model_name
    else:
        timestamp = timenow.isoformat().replace(':','_')
        if timestamp not in model_name:
            model_name += timestamp
            model._name = model_name

    # Get model path
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = pathlib.Path(model_dir) / pathlib.Path(model_name + '.h5')

    # Setup callbacks
    earlyStopping = keras.callbacks.EarlyStopping(monitor=monitor, patience=epoch_patience, mode='auto', restore_best_weights=True)
    tfBoard = TrainValTensorBoard(log_dir=str(pathlib.Path(logdir) / model_name), write_graph=False)
    # checkpoint = keras.callbacks.ModelCheckpoint(filepath=str(model_path), monitor=monitor, save_best_only=True)
    checkpoint = MetadataModelCheckPoint(filepath=str(model_path), timenow=timenow,
        monitor=monitor, save_best_only=True, custom_dict=params)

    # Get a nice and pretty progress bar
    try:
        get_ipython # check if inside an IPython/Jupyter shell
        prettyProgressBar = keras_tqdm.TQDMNotebookCallback(leave_inner=False, leave_outer=True)
    except:
        prettyProgressBar = keras_tqdm.TQDMCallback(leave_inner=False, leave_outer=True)
        print('Running outside Jupyter.')

    # Setup the Callback list
    callbackList = [prettyProgressBar, tfBoard, earlyStopping, checkpoint]

    # Train model and output training history
    print(f'Starting the model training [{model_name}].')

    training_history = model.fit_generator(generator=training_generator,
                    epochs=epochs,
                    use_multiprocessing=use_multiprocessing,
                    workers=workers,
                    validation_data=validation_generator,
                    shuffle=True,
                    callbacks=callbackList, 
                    initial_epoch=initial_epoch, verbose=0)

    # Get history dictionary
    history = training_history.history

    # Load best model checkpoint
    # model = keras.models.load_model(str(model_path.absolute()))
    print(f'Saved the trained model (with the best {monitor}) at [{model_path}].')
    
    # Evaluate trained model
    report, confusion, scores = classification_report(model, validation_generator, verbose=0)
    print('Validation loss:', scores[0])
    print('Validation accuracy:', scores[1])

    # Open partial metadata
    patial_meta_path = model_path.with_suffix('.partial.pkl')
    with open(str(patial_meta_path), 'rb') as handle:
        patial_meta = pickle.load(handle)

    # Elapsed time
    # elapsed_time = (datetime.now() - timenow).total_seconds()
    elapsed_time = patial_meta['elapsed_time']
    print(f'Trained and evaluated in {elapsed_time} seconds.')

    # Save normalization data
    norm_data = training_generator.norm_data

    # Metadata dictonary
    meta = {'training_history': history,
            'timestamp': timenow.isoformat(),
            'batch_size': training_generator.batch_size,
            'epochs': len(history['loss']),
            'elapsed_time': elapsed_time,
            'epoch_patience': epoch_patience,
            'report': report,
            'confusion': confusion,
            'scores': scores,
            'norm_data': norm_data,
            'feature_metadata': training_generator.metadata}

    # Save metadata
    with open(str(pathlib.Path(model_dir) / pathlib.Path(model_name + '.pkl')), 'wb') as output:
        pickle.dump(meta, output)

    return model, history, scores


def classification_report(model, test_generator, verbose=0):

    y_true = test_generator.labels[test_generator.dataset_indexes]
    scene_labels = test_generator.metadata['scene_labels']

    y_pred = np.argmax(model.predict_generator(test_generator), axis=-1).astype('uint8')
    num_classes = test_generator.num_classes

    keras.utils.to_categorical(y_true, num_classes)
    report = skmetrics.classification_report(y_true, y_pred, target_names=scene_labels)
    confusion = skmetrics.confusion_matrix(y_true, y_pred)
    scores = model.evaluate_generator(test_generator)

    if verbose:
        print(report)

    return report, confusion, scores


def plot_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, cmap='YlGnBu'):
    """Plots a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    Modified from `shaypal5's gist`.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
    cmap: str
        Colormap for the heatmap (see `Colormaps in Matplotlib`). Defaults to YlGnBu.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure

    References
    ----------
    .. _shaypal5's gist:
       https://gist.github.com/shaypal5/94c53d765083101efc0240d776a23823
    
    .. _Colormaps in Matplotlib:
       https://matplotlib.org/tutorials/colors/colormaps.html
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap=cmap) 
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label', fontsize=fontsize, fontweight='bold')
    plt.xlabel('Predicted label', fontsize=fontsize, fontweight='bold')
    return fig


def evaluate_model_generator(evaluation_generator, model_path):

    model_path = pathlib.Path(model_path)
    model = keras.models.load_model(str(model_path.absolute()))

    report, confusion, scores = classification_report(model, evaluation_generator, verbose=1)

