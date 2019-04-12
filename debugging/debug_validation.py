# Standard libraries
import pathlib
import glob
import platform
import pickle
from datetime import datetime
from pprint import pprint

# Scientific stack
import numpy as np
import numpy.random as rnd
import pandas as pd

# Chunked data
import zarr

# Audio processing
import dcase_util as du

# Pretty progress bar
import tqdm

import preprocessing as prep

n_feats = 100
dataset_name = f'numfeats{n_feats}'

# db_path = '/media/zanco/DADOS/zanco/datasets/TUT-urban-acoustic-scenes-2018-development/'
db_path = '/media/zanco/DADOS/zanco/datasets/TAU-urban-acoustic-scenes-2019-development/'

# db_path = 'E:/datasets/TUT-urban-acoustic-scenes-2018-development/'
# db_path = 'E:/datasets/TAU-urban-acoustic-scenes-2019-development/'

# version = '2018'
version = '2019'


preprocessor = prep.DataPreprocessing(db_path=db_path,
                                      version=version,
                                      n_feats=n_feats, 
                                      dataset_name=dataset_name,
                                      dataset_folder=f'../saved_features{version}',
                                      audio_preprocess='mid',
                                      feature_type='mel_spectrogram')
# preprocessor.process(overwrite=False)
fold_meta, fold_split = preprocessor.generate_fold_meta(overwrite=False)

train_ids = fold_meta['identifier'][fold_split[0][0]]
valid_ids = fold_meta['identifier'][fold_split[0][1]]

c = list(set(train_ids) & set(valid_ids))

print(len(c))


seed = 0
n_splits = 5
# Get consistent results (same folds every time)
rand_state = rnd.get_state() # get current PRNG state
rnd.seed(seed)


# Get training and evaluation example indexes
train_ind = np.where(preprocessor.db_meta['example_type'].values == 'train')[0]
eval_ind = np.where(preprocessor.db_meta['example_type'].values == 'test')[0]

# Split based on labels and identifiers
from sklearn.model_selection import GroupKFold
splitter = GroupKFold(n_splits=n_splits)
X = np.empty([train_ind.size,1])
y = preprocessor.db_meta['scene_label'][train_ind]
ids = preprocessor.db_meta['identifier'][train_ind]
temp_fold_split = list(splitter.split(X=X,y=y,groups=ids))

# Fix indexing
fold_split = [[train_ind[x[0]], train_ind[x[1]]] for x in temp_fold_split]





from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
np.random.seed(1338)
cmap_data = plt.cm.Paired
cmap_group = plt.cm.prism
cmap_cv = plt.cm.coolwarm
n_splits = 5


# Generate the class/group data
_, label_index = np.unique(preprocessor.db_meta['scene_label'][train_ind].values, return_inverse=True)
y = label_index.astype('i1')

_, id_index = np.unique(preprocessor.db_meta['identifier'][train_ind].values, return_inverse=True)
groups = id_index.astype(int)

def visualize_groups(classes, groups):
    # Visualize dataset groups
    fig, ax = plt.subplots()
    plot = ax.scatter(range(len(groups)),  [.5] * len(groups), c=groups, marker='_',
               lw=50, cmap=cmap_group)
    ax.scatter(range(len(groups)),  [3.5] * len(groups), c=classes, marker='_',
               lw=50, cmap=cmap_data)
    ax.set(ylim=[-1, 5], yticks=[.5, 3.5],
           yticklabels=['Data\ngroup', 'Data\nclass'], xlabel="Sample index")
    fig.colorbar(plot)


visualize_groups(y, groups)


def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        plot = ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    fig.colorbar(plot)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_group)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


fig, ax = plt.subplots()
# cv = KFold(n_splits)
plot_cv_indices(splitter, X, y, groups, ax, n_splits)

plt.show()
exit(0)


