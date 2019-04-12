# Standard Libraries
import pathlib
import pickle

# Scientific stack
import numpy as np
import keras

# Chunked data
import dask
import dask.array as da
import zarr

# generate_fold_meta()
from sklearn.model_selection import GroupKFold
import numpy.random as rnd
import pandas as pd

# Pretty progress bar
import tqdm

# References: 
# https://github.com/keras-team/keras/issues/1638
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly



class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self, zarr_root, zarr_group, fold_num, set_type='train', use_mixup=True, mixup_alpha=0.2,
        batch_size=32, shuffle=True, conv2d=True, transpose=False, **kwargs):
        'Initialization'

        # Set mixup hyper-parameter
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # Zarr dataset handling
        self.zarr_root = zarr.open_group(str(zarr_root), mode='r')
        self.zarr_group = zarr_group
        self.zarr_dataset = self.zarr_root[self.zarr_group]
        
        # Set cross-validation fold metadata
        self.fold_num = fold_num
        self.zarr_folds = self.zarr_dataset['folds']
        self.zarr_fold = self.zarr_folds[f'fold{self.fold_num}']

        # Get metadata
        self.metadata = self.zarr_dataset.attrs.asdict()

        # Get index for the subset (train, test or eval)
        self.set_type = set_type
        if set_type == 'eval':
            fold_meta = self.get_fold_meta()
            self.dataset_indexes = np.where(fold_meta['fold0'].values == 'eval')[0]
        else:
            self.dataset_indexes = self.zarr_fold[set_type][:]

        # Set features dimensions
        self.set_dim(transpose, conv2d)

        # Normalization data
        self.norm_data = {}
        self.norm_data['mean'] = self.zarr_fold['norm_data']['mean'][:]
        self.norm_data['std'] = self.zarr_fold['norm_data']['std'][:]

        # Labels and features
        self.labels = self.zarr_dataset['y'][:]
        self.features = self.zarr_dataset['X']

        # Set batch size and if we want to shuffle the examples
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Set number of classes
        if 'unknown' in self.metadata['scene_labels']:
            self.has_unkown = True
            self.num_classes = len(self.metadata['scene_labels']) - 1
            self.to_categorical = self.unkown_to_categorical
        else:
            self.has_unkown = False
            self.num_classes = len(self.metadata['scene_labels'])
            self.to_categorical = keras.utils.to_categorical
        self.on_epoch_end()

        self.len = len(self)
        self.shape = self.features.shape
        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.dataset_indexes) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        if (index+1)*self.batch_size > self.batch_indexes.size - 1:
            indexes = self.batch_indexes[index*self.batch_size:]
        else:
            indexes = self.batch_indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        batch_data = self.__data_generation(indexes)

        return batch_data


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.batch_indexes = self.dataset_indexes.copy()
        if self.shuffle == True:
            np.random.shuffle(self.batch_indexes)


    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        # Fetch features and normalize them
        Xnorm = da.from_zarr(self.features)[indexes,]
        Xnorm = (Xnorm - self.norm_data['mean']) / self.norm_data['std']

        # Transpose if needed
        if self.transpose:
            Xnorm = da.transpose(Xnorm, axes=[0,2,1])

        # Generate data
        X = da.reshape(Xnorm, [len(indexes), *self.dim]).compute()
        y = self.labels[indexes].copy()

        # Return X,y pairs now (without mixup)
        if not self.use_mixup or (self.set_type != 'train'):
            return X, self.to_categorical(y, num_classes=self.num_classes)

        # Mixup
        mixed_x, y_a, y_b, lamb = self.mixup_batch(X, y, alpha=self.mixup_alpha)
        batch_data_in = mixed_x # X_mixup
        y_a = self.to_categorical(y_a, num_classes=self.num_classes)
        y_b = self.to_categorical(y_b, num_classes=self.num_classes)
        batch_data_out = lamb * y_a + (1 - lamb) * y_b # y_mixup
        
        return batch_data_in, batch_data_out


    def mixup_batch(self, X, y, alpha=1.0):
        # Based on `mixup: Beyond Empirical Risk Minimization`
        # https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py#L119

        if alpha > 0:
            lamb = np.random.beta(alpha, alpha)
        else:
            lamb = 1

        batch_size = y.shape[0]
        index = np.random.permutation(batch_size)

        mixed_x = lamb * X + (1 - lamb) * X[index,]
        y_a, y_b = y, y[index,]

        return mixed_x, y_a, y_b, lamb


    def set_dim(self, transpose, conv2d):
        
        # Set features dimensions
        self.dim = self.metadata['feature_shape']

        # Set transposed output
        self.transpose = transpose
        if self.transpose:
            self.dim = (*self.dim[::-1],)

        # Set Conv2D input
        self.conv2d = conv2d
        if self.conv2d:
            self.dim = (*self.dim, 1)


    def get_fold_meta(self):

        # Convert JSON to pandas.Dataframe (needs index sorting...)
        fold_meta = pd.read_json(self.zarr_folds.attrs.asdict()['fold_meta'], orient='column')
        fold_meta = fold_meta.sort_index()

        return fold_meta

    def unkown_to_categorical(self, y, num_classes=None, dtype='float32'):
        # Source: https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py#L9

        y = np.array(y, dtype='int')
        input_shape = y.shape
        
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])

        y = y.ravel()

        if not num_classes:
            num_classes = np.max(y) + 1        
        n = y.shape[0]

        categorical = np.zeros((n, num_classes), dtype=dtype)

        # Unkown categories are negative (e.g. -1)
        # If unkown, keep zero
        ind = y >= 0
        categorical[ind, y[ind]] = 1


        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)

        return categorical


class FeatureNormalizer():

    def __init__(self, zarr_root, zarr_group, fold_num, conv2d=True, transpose=False, **kwargs):
        'Initialization'

        # Zarr dataset handling
        zarr_root = zarr.open_group(str(zarr_root), mode='r')

        # Get cross-validation fold metadata
        zarr_fold = zarr_root[f'{zarr_group}/folds/fold{fold_num}']
        
        # Get metadata
        self.metadata = zarr_root[zarr_group].attrs.asdict()
        self.scene_labels = self.metadata['scene_labels']

        # Normalization data
        self.norm_data = {}
        self.norm_data['mean'] = zarr_fold['norm_data']['mean'][:]
        self.norm_data['std'] = zarr_fold['norm_data']['std'][:]

        # Set features dimensions
        self.set_dim(transpose, conv2d)

    def process(self, features):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        num_samples = features.shape[0]

        # Fetch features and normalize them
        Xnorm = (features - self.norm_data['mean']) / self.norm_data['std']

        # Transpose if needed
        if self.transpose:
            Xnorm = np.transpose(Xnorm, axes=[0,2,1])

        # Generate data
        X = np.reshape(Xnorm, [num_samples, *self.dim])

        return X

    def set_dim(self, transpose, conv2d):
        
        # Set features dimensions
        self.dim = self.metadata['feature_shape']

        # Set transposed output
        self.transpose = transpose
        if self.transpose:
            self.dim = (*self.dim[::-1],)

        # Set Conv2D input
        self.conv2d = conv2d
        if self.conv2d:
            self.dim = (*self.dim, 1)

