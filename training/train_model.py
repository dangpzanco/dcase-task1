import pathlib
import dataset_generator as dgen
import utils

import keras
import keras.backend as K
import tensorflow as tf

import pandas as pd
import numpy as np

import zarr

# Dataset definitions
version = '2019'
features_folder = pathlib.Path(f'../saved_features{version}')
feature_type = 'mel_spectrogram'
audio_preprocess = 'mid'
n_feats = 100
dataset_name = f'numfeats{n_feats}'
zarr_group = f'{feature_type}/{audio_preprocess}/{dataset_name}'

print(f'Fetching features from {features_folder}')

# Checkpoint and Tensorboard folders
model_dir = pathlib.Path('models_mixup')
logdir = model_dir / 'logs'

# Learning hyperparameters
batch_size = 128
mixup_alpha = 0.2
num_epochs = 600
patience = 100


fold_num = 0

for fold_num in range(5):


    # Generator parameters
    train_params = dict(zarr_root=features_folder, 
                        zarr_group=zarr_group, 
                        fold_num=fold_num, 
                        batch_size=batch_size)

    test_params = dict(zarr_root=features_folder, 
                        zarr_group=zarr_group, 
                        fold_num=fold_num, 
                        batch_size=batch_size)

    # Generators
    training_generator = dgen.DataGenerator(set_type='train', use_mixup=True, mixup_alpha=mixup_alpha, **train_params)
    validation_generator = dgen.DataGenerator(set_type='test', use_mixup=False, **test_params)

    # Design model
    K.reset_uids()
    data_shape = training_generator.dim
    print('data_shape:', data_shape)

    model = utils.best_model(data_shape)

    opt = keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    # Train model
    model_name = f'monaural_version{version}_{audio_preprocess}{n_feats}_mixup_{mixup_alpha}_fold{fold_num}'
    utils.train_model_generator(training_generator, validation_generator, model, model_name, 
        use_multiprocessing=True, workers=2, epochs=num_epochs, epoch_patience=patience, 
        monitor='val_loss', logdir=logdir, model_dir=model_dir)

# TPU            | GPU
# 56s 4ms/sample | 74s 5ms/sample
