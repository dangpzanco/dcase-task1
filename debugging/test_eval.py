import pathlib
import dataset_generator as dgen
import utils

import keras
import keras.backend as K
import tensorflow as tf

import pandas as pd
import numpy as np

from pprint import pprint
import tqdm

import pickle

import sklearn.metrics as skmetrics

# Dataset definitions
version = '2019c'
features_folder = pathlib.Path(f'../saved_features{version}')
feature_type = 'mel_spectrogram'
audio_preprocess = 'mid'
n_feats = 100
dataset_name = f'numfeats{n_feats}'
zarr_group = f'{feature_type}/{audio_preprocess}/{dataset_name}'

print(f'Fetching features from {features_folder}')

batch_size = 10

gen_params = dict(zarr_root=features_folder, 
                    zarr_group=zarr_group, 
                    fold_num=0, 
                    batch_size=batch_size, shuffle=False)

# Generators
dataset_generator = dgen.DataGenerator(set_type='eval', use_mixup=False, **gen_params)
# dataset_generator = dgen.DataGenerator(set_type='test', use_mixup=False, **gen_params)
# dataset_generator = dgen.DataGenerator(set_type='train', use_mixup=False, **gen_params)


# model_path = '/home/zanco/Desktop/models_mixup/monaural_version2019_mid100_mixup_0.2_fold02019-03-25T13_38_06.804083.h5'
# model_path = '/home/zanco/Desktop/models_mixup/monaural_version2019_mid100_mixup_0.2_fold12019-03-25T16_53_03.815760.h5'
# model_path = pathlib.Path(model_path)

# models_path = pathlib.Path('/home/zanco/drive/models/models_mixup/')
# model_path = '/home/zanco/drive/models/models_mixup/monaural_version2019_mid100_mixup_0.2_fold02019-03-28T19_27_41.541811.h5'
# model_path = '/home/zanco/drive/models/models_mixup/monaural_version2019_mid100_mixup_0.2_fold12019-03-28T21_27_25.730494.h5'

model_path = 'models_unkown/monaural_version2019c_mid100_mixup_0.2_fold02019-04-02T01_51_06.850876.h5'
# model_path = 'models_mixup/monaural_version2019_mid100_mixup_0.2_fold02019-03-28T19_27_41.541811.h5'

meta_path = pathlib.Path(model_path).with_suffix('.pkl')
partial_path = pathlib.Path(model_path).with_suffix('.partial.pkl')

# with open(partial_path, 'rb') as handler:
#     meta = pickle.load(handler)
# pprint(meta)


# exit(0)

model = keras.models.load_model(model_path)
model.summary()

# pred_file = pathlib.Path('predictions_fold0.npy')
pred_file = pathlib.Path('predictions_eval.npy')
# pred_file = pathlib.Path('predictions_train.npy')

# pred_file = pathlib.Path('predictions_mixup_fold0.npy')
# pred_file = pathlib.Path('predictions_mixup_eval.npy')
# pred_file = pathlib.Path('predictions_mixup_train.npy')

if not pred_file.exists():
    predictions = model.predict_generator(dataset_generator, verbose=1, workers=8, use_multiprocessing=False)
    np.save(pred_file, predictions)
else:
    predictions = np.load(pred_file)



print(predictions.shape)


y_true = dataset_generator.labels[dataset_generator.dataset_indexes].copy()
known_index = y_true > -1
unknown_index = y_true == -1

y_true_known = y_true[known_index]
y_true_unknown = y_true[unknown_index]


thresh = 0.5
num_points = 100
thresholds = np.linspace(0.01, 0.99, num_points)
accuracy = np.empty([num_points, 3])
for i, thresh in enumerate(thresholds):

    known_positions = (predictions > thresh).sum(axis=-1)
    known_positions[known_positions > 0] = 1
    known_positions[known_positions == 0] = -1
    y_pred = known_positions.copy()
    y_pred[known_positions == 1] = predictions[known_positions == 1].argmax(axis=-1)

    y_pred_known = y_pred[known_index]
    y_pred_unknown = y_pred[unknown_index]

    # y_pred_known = predictions[known_index].argmax(axis=-1)
    # y_pred_unknown = -1 * ((predictions[unknown_index] > thresh).sum(axis=-1) == 0)
    
    accuracy[i, 0] = skmetrics.accuracy_score(y_true_known, y_pred_known)
    accuracy[i, 1] = skmetrics.accuracy_score(y_true_unknown, y_pred_unknown)
    accuracy[i, 2] = 0.5*(accuracy[i, 0] + accuracy[i, 1])

# pprint(list(y_pred_known))

import matplotlib.pyplot as plt

# plt.plot(thresholds, accuracy)

# plt.show()


fig, ax = plt.subplots()
ax.plot(thresholds, accuracy)

def annotate_xy(x, y, ax=None):
    # xmax = x[np.argmax(y)]
    # ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(x, y)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->",connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(x, y), xytext=(0.94,0.96), **kw)

annotate_xy(thresholds[np.argmax(accuracy[:,2])], accuracy[:,2].max())

# ax.set_ylim(-0.3,1.5)
plt.legend(['Accuracy (Known Classes)', 'Accuracy (Unknown Classes)', 'Accuracy (Average)'])
plt.grid()
plt.show()


exit(0)

# pprint(list(known_positions))
pprint(list(y_true))
pprint(list(y_pred))


accuracy = skmetrics.accuracy_score(y_true, y_pred)

report = skmetrics.classification_report(y_true, y_pred)
confusion = skmetrics.confusion_matrix(y_true, y_pred)

print(report)
print(confusion)
print(f'Accuracy: {accuracy * 100}%')


# y_pred = 

# validation fold0 [1.0419330989846487, 0.6106870285232299]
# test model_fold0 [0.9239499481503824, 0.6550335607501255]

# training   fold1 [0.5752689281080454, 0.8553938175882390]
# validation fold1 [0.4313417017914867, 0.9203925782097595]
# test       fold1 [0.9508740292185219, 0.6621284787744026]


