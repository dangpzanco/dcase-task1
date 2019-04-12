
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
import dask.array as da
from dask.diagnostics import ProgressBar as DaskProgressBar

# Audio processing
import dcase_util as du

# Pretty progress bar
import tqdm


def get_dataset(db_root=None, meta_path='dataset_metadata.csv', version='2018', overwrite=False):

    # Default path
    if db_root is None:
        db_root = 'datasets'
    
    db_root = pathlib.Path(db_root)

    if version == '2018':
        db_path = db_root / 'TUT-urban-acoustic-scenes-2018-development'
    elif version in ['2019', '2019a']:
        db_path = db_root / 'TAU-urban-acoustic-scenes-2019-development'
    elif version == '2019c':
        db_path = db_root / 'TAU-urban-acoustic-scenes-2019-openset-development'

    # Download and extract dataset if it doesn't exist
    if not db_path.exists():
        
        if version == '2018':
            db = du.datasets.TUTUrbanAcousticScenes_2018_DevelopmentSet(data_path=db_path, included_content_types=['all'])
        elif version in ['2019', '2019a']:
            db = du.datasets.TAUUrbanAcousticScenes_2019_DevelopmentSet(data_path=db_path, included_content_types=['all'])
        elif version == '2019c':
            db = du.datasets.TAUUrbanAcousticScenes_2019_Openset_DevelopmentSet(data_path=db_path, included_content_types=['all'])

        db.initialize()
        db.show()

    meta_path = db_path / pathlib.Path(meta_path)

    # Works for both DCASE2018 Task 1A and DCASE2019 Task 1A/C
    if version == '2018':
        ext = 'txt'
        names = ['filename', 'scene_label']
    elif version in ['2019', '2019a', '2019c']:
        ext = 'csv'
        names = None
    else:
        print(f'Version "{version}" not supported [only "2018", "2019", "2019a" and "2019c"].')
        exit(-1)

    # Open CSV metadata
    metadata = pd.read_csv(str(db_path / 'meta.csv'), sep='\t')
    train_meta = pd.read_csv(str(db_path / f'evaluation_setup/fold1_train.{ext}'), sep='\t', names=names)
    test_meta = pd.read_csv(str(db_path / f'evaluation_setup/fold1_evaluate.{ext}'), sep='\t', names=names)
    num_samples = train_meta.shape[0] + test_meta.shape[0]
    metadata = metadata.iloc[:num_samples,]

    # Find all the training examples in the complete metadata
    example_type = metadata['filename'].isin(train_meta['filename'])

    # Split the dataset files between training and testing
    example_type[example_type == True] = 'train'
    example_type[example_type == False] = 'test'

    # Rearrange columns
    cols = metadata.columns.tolist()
    cols.insert(1,'example_type')

    # Insert the column for example types (train and test)
    metadata['example_type'] = example_type.values
    metadata = metadata[cols]

    # Remove useless column (for Task 1A)
    metadata = metadata.drop('source_label', axis=1)

    # Keep filename simpler
    metadata['filename'] = metadata['filename'].str.replace('audio/','').str.replace('.wav','')

    # Save metadata to CSV
    if not meta_path.exists() or overwrite:
        metadata.to_csv(meta_path, index=False, sep='\t')

    return db_path, metadata


class DataPreprocessing():
    """Constructor

        Parameters
        ----------
        fs : int
            Sampling rate of the incoming signal.

        win_length_samples : int
            Window length in samples.
            Default value None

        hop_length_samples : int
            Hop length in samples.
            Default value None

        win_length_seconds : float
            Window length in seconds.
            Default value 0.04

        hop_length_seconds : float
            Hop length in seconds.
            Default value 0.02

        n_feats : int
            Number of features per window.
            Default value 50

        n_fft : int
            Length of the FFT window.
            Default value 2048

        window_type : str
            Window function type.
            Default value 'hann'

        feature_type : str
            Type of feature to compute, possible values include:
            'mel_spectrogram', 'mfcc_static' and 'mfcc_delta'.
            Default value 'mel_spectrogram'

        audio_preprocess : str
            Stereo to mono conversion mode, possible values include:
            'mid': L + R
            'side': L - R
            'left': L
            'right': R
            Default value 'mid'

        min_max : bool
            If True, Normalize each example by its maximum amplitude.
            Default value False

        dataset_name : str
            Give a name to the output dataset files.
            Default value 'default'

    """
    def __init__(self, db_root=None, version='2018', fs=None, win_length_seconds=40e-3, hop_length_seconds=20e-3, n_feats=100, n_fft=2048, window_type='hann', 
                 feature_type='mel_spectrogram', audio_preprocess='mid', minmax=False, dataset_name='default', dataset_folder='saved_features', **kwargs):
        super().__init__()

        self.__dict__.update(kwargs)
        
        if fs is None:
            if version in ['2018', '2018a', '2019', '2019a']:
                fs = 48000
            elif version in ['2019c']:
                fs = 44100
        
        self.fs = fs
        self.version = version

        self.audio_preprocess = audio_preprocess
        self.minmax = minmax
        
        self.dataset_name = dataset_name
        self.features_folder = pathlib.Path(dataset_folder)

        self.db_path, self.db_meta = get_dataset(db_root=db_root, version=version, overwrite=True)

        # Set feature metadata
        self.feature_metadata = dict(fs=fs, win_length_seconds=win_length_seconds, hop_length_seconds=hop_length_seconds, 
                n_feats=n_feats, spectrogram_type='magnitude', n_fft=n_fft, window_type=window_type, audio_preprocess=audio_preprocess, 
                minmax=minmax, db_meta=self.db_meta.to_json(), **kwargs)
        
        # Feature extractor input arguments
        args = dict(fs=fs, win_length_seconds=win_length_seconds, hop_length_seconds=hop_length_seconds, 
                n_mels=n_feats, n_mfcc=n_feats, n_dim=n_feats, spectrogram_type='magnitude', n_fft=n_fft, window_type=window_type, **kwargs)
        self.set_extractor(feature_type, args)


    def set_extractor(self, feature_type, args):
        self.feature_type = feature_type

        # Create feature extractor
        if feature_type == 'spectrogram':
            self.extractor = du.features.SpectralFeatureExtractor(**args)
        elif feature_type == 'mel_spectrogram':
            self.extractor = du.features.MelExtractor(**args)
        elif feature_type == 'mfcc_static':
            args['n_mels'] = int(n_fft/4)
            self.extractor = du.features.MfccStaticExtractor(**args)
        elif feature_type == 'mfcc_delta':
            args['n_mels'] = int(n_fft/4)
            self.extractor = du.features.MfccDeltaExtractor(**args)
        else:
            message = f'set_extractor(): Unknown feature type [{feature_type}]'
            raise ValueError(message)


    def process(self, use_folds=True, overwrite=False):

        # Create folder to hold the dataset
        dataset_folder = self.features_folder / f'{self.feature_type}/{self.audio_preprocess}/{self.dataset_name}'
        dataset_folder.mkdir(parents=True, exist_ok=True)

        # Zarr group
        root, data_group = self.get_group(mode='a')

        # Get shapes
        audio_length, feature_shape = self.get_shapes()

        # Dataset size
        num_samples = self.db_meta.shape[0]

        # Set X, y zarr arrays
        if overwrite:
            X = root.create_dataset(f'{data_group}/X', shape=(num_samples,*feature_shape), dtype='float32', chunks=(1,None,None), overwrite=overwrite)
            y = root.create_dataset(f'{data_group}/y', shape=(num_samples,), dtype='int', chunks=None, overwrite=overwrite)
        else:
            X = root.require_dataset(f'{data_group}/X', shape=(num_samples,*feature_shape), dtype='float32', chunks=(1,None,None))
            y = root.require_dataset(f'{data_group}/y', shape=(num_samples,), dtype='int', chunks=None)

        # Stop processing if already done AND we don't want to overwrite the dataset
        if (X.nchunks == X.nchunks_initialized) and not overwrite:
            return

        # Get label data
        scene_labels, label_index = np.unique(self.db_meta['scene_label'], return_inverse=True)
        scene_labels = list(scene_labels)
        labels = label_index.astype('i1')

        # This only works if "unkown" is the last element of scene_labels (possible bug)
        if self.version == '2019c':
            unknown_index = scene_labels.index('unknown')
            labels[labels == unknown_index] = -1

        # Save label data
        y[:] = labels

        # Get metadata
        meta = self.feature_metadata
        meta.update({'scene_labels': scene_labels,
            'num_samples': num_samples,
            'feature_shape': feature_shape})

        # Save metadata
        g = root.require_group(data_group)
        g.attrs.update(**meta)

        # Print some nice information
        self.print_summary(feature_shape, num_samples)

        # Make a nice progress bar
        try:
            get_ipython # check if inside an IPython/Jupyter shell
            prettyProgressBar = tqdm.tqdm_notebook(self.db_meta['filename'])
        except:
            prettyProgressBar = tqdm.tqdm(self.db_meta['filename'])

        # Process the train dataset
        for i, item in enumerate(prettyProgressBar):

            # Open audio file and convert to mono
            audio_data = self.audio_mixing(item, self.audio_preprocess, self.minmax)

            # Fix audio duration
            audio_data = self.fix_audio_length(audio_data, audio_length)

            # Extract features for each example
            features = self.feature_extraction(audio_data)

            # Save each example of the dataset
            X[i,] = features


    def get_group(self, mode='a'):
        root = zarr.open_group(str(self.features_folder), mode=mode)
        data_group = f'{self.feature_type}/{self.audio_preprocess}/{self.dataset_name}'
        return root, data_group


    def get_shapes(self):
        audio_path = str(self.db_path / 'audio' / (self.db_meta['filename'][0] + '.wav'))

        # Open sample file
        audio_container = du.containers.AudioContainer().load(
            filename=audio_path,
            mono=True,
            fs=self.fs
        )

        # Get audio duration in samples
        audio_length = audio_container.data.size

        # Get feature vector shape
        feature_shape = self.extractor.extract(audio_container).shape

        return audio_length, feature_shape


    def fix_audio_length(self, audio_data, length):

        current_size = audio_data.size

        if current_size == length:
            return audio_data
        elif current_size < length:
            new_audio_data = np.empty(length)
            new_audio_data[:current_size] = audio_data.copy()
            new_audio_data[current_size:] = 0
            return new_audio_data
        else:
            new_audio_data = audio_data[:length].copy()
            return new_audio_data


    def audio_mixing(self, audio_filename, audio_preprocess, minmax):

        # Get audio file path
        audio_path = str(self.db_path / 'audio' / (audio_filename + '.wav'))

        # Put audio in a DCASE Util container
        audio_container = du.containers.AudioContainer().load(
            filename=str(audio_path),
            mono=False,
            fs=self.fs
        )
        audio_data = audio_container.data

        if self.version is not '2019c':
            # Audio mixing
            if audio_preprocess == 'left':
                audio_data = audio_data[0,:]
            elif audio_preprocess == 'right':
                audio_data = audio_data[1,:]
            elif audio_preprocess == 'mid':
                audio_data = audio_data[0,:] + audio_data[1,:]
            elif audio_preprocess == 'side':
                audio_data = audio_data[0,:] - audio_data[1,:]
            else:
                message = f'audio_mixing(): Unknown audio_preprocess type [{audio_preprocess}]'
                raise ValueError(message)

        # Optional normalization per audio snippet
        if minmax:
            audio_data /= np.maximum(audio_data.max(),-audio_data.min())

        return audio_data


    def feature_extraction(self, audio_data):
        return self.extractor.extract(audio_data)


    def print_summary(self, feature_shape, num_samples):
        print('feature_shape:', feature_shape)
        print('num_samples:', num_samples)
        print('feature_metadata:')
        pprint(self.feature_metadata.keys())


    def generate_fold_meta(self, n_splits=5, seed=0, overwrite=False, check_balance=True, verbose=True):

        # Get consistent results (same folds every time)
        rand_state = rnd.get_state() # get current PRNG state
        rnd.seed(seed)

        # Get training and evaluation example indexes
        train_ind = np.where(self.db_meta['example_type'].values == 'train')[0]
        eval_ind = np.where(self.db_meta['example_type'].values == 'test')[0]

        # Split based on labels and identifiers
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=n_splits)
        X = np.empty([train_ind.size,1])
        y = self.db_meta['scene_label'][train_ind]
        ids = self.db_meta['identifier'][train_ind]
        temp_fold_split = list(splitter.split(X=X,y=y,groups=ids))

        # Fix indexing
        fold_split = [[train_ind[x[0]], train_ind[x[1]]] for x in temp_fold_split]

        # Create crossvalidation fold metadata
        fold_meta = self.db_meta
        fold_meta['example_type'][eval_ind] = 'eval'
        example_type = fold_meta['example_type']

        # Set metadata for each fold
        for i in range(n_splits):
            example_type[fold_split[i][0]] = 'train'
            example_type[fold_split[i][1]] = 'test'
            fold_meta[f'fold{i}'] = example_type

        # Drop unused column
        fold_meta = fold_meta.drop(columns='example_type')

        # Acess dataset Zarr group
        root, data_group_path = self.get_group(mode='r+')
        data_group = root.require_group(data_group_path)
        folds_group = data_group.require_group('folds', overwrite=overwrite)

        # Save fold indexes and normalization data for each validation fold
        for i in range(n_splits):
            fold_group = folds_group.require_group(f'fold{i}', overwrite=overwrite)

            train_indexes = fold_group.create_dataset('train', shape=(fold_split[i][0].size,), dtype='int64', chunks=None, overwrite=True)
            train_indexes[:] = fold_split[i][0]

            test_indexes = fold_group.create_dataset('test', shape=(fold_split[i][1].size,), dtype='int64', chunks=None, overwrite=True)
            test_indexes[:] = fold_split[i][1]

            with DaskProgressBar():
                self.set_norm_factors(data_group, fold_group, overwrite=overwrite)

        # Save metadata
        meta = {'num_folds': n_splits, 'fold_meta': fold_meta.to_json()}
        folds_group.attrs.update(**meta)
        
        # Avoid messing with the PRNG (set the state back)
        rnd.set_state(rand_state)

        if verbose:
            print(fold_meta.head(n=20).to_string())

        # Check folds class balance and identifier decorrelation
        if check_balance:
            self.check_fold_balance(fold_meta, n_splits, verbose=verbose)

        return fold_meta, fold_split


    def set_norm_factors(self, data_group, fold_group, overwrite=False):

        # Get Zarr arrays
        train_indexes = fold_group['train'][:]
        X = da.from_zarr(data_group['X'])
        norm_shape = X.shape[1:]

        # Create normalization data Zarr arrays
        norm_group = fold_group.require_group('norm_data')
        norm_group.require_dataset('s1', shape=norm_shape, dtype='float32', chunks=None)
        norm_group.require_dataset('s2', shape=norm_shape, dtype='float32', chunks=None)
        norm_group.require_dataset('mean', shape=norm_shape, dtype='float32', chunks=None)
        norm_group.require_dataset('std', shape=norm_shape, dtype='float32', chunks=None)

        # Stop processing if already done AND we don't want to overwrite the dataset
        if (norm_group['s1'].nchunks == norm_group['s1'].nchunks_initialized) and not overwrite:
            return

        # Compute normalization factors
        fold_num = pathlib.PurePath(fold_group.name).name[-1]
        print(f'Computing the normalization factors for the cross-validation fold #{fold_num}.\nThis may take some time...')

        # Compute sum and squared sum
        s1 = X[train_indexes,].sum(axis=0)
        s2 = (X[train_indexes,] ** 2).sum(axis=0)
        S = da.stack([s1, s2], axis=0).compute()
        s1 = S[0,]
        s2 = S[1,]
        n = train_indexes.size

        # Fill Zarr arrays with the normalization factors
        norm_group['s1'][:] = s1
        norm_group['s2'][:] = s2
        norm_group['mean'][:] = s1 / n
        norm_group['std'][:] = np.sqrt((n * s2 - (s1 * s1)) / (n * (n - 1)))


    def check_fold_balance(self, fold_meta, n_splits, verbose=True):
        # Get possible class labels
        labels = list(np.unique(fold_meta['scene_label']))

        # Get train and test indices
        train_ind = []
        valid_ind = []
        for i in range(n_splits):
            train_ind.append(fold_meta[f'fold{i}'].values == 'train')
            valid_ind.append(fold_meta[f'fold{i}'].values == 'test')

            train_ids = fold_meta['identifier'][train_ind[i]]
            valid_ids = fold_meta['identifier'][valid_ind[i]]

            c = list(set(train_ids) & set(valid_ids))

            if len(c) > 0:
                print(f"""Warning: Fold #{i} has {len(c)} overlaping groups between training and validation sets.\n
                          This can result in overfitting and poor performance.""")

            # print(list(np.where(valid_ind[i])[0]))

        # Find class occurences
        class_occurence = {}
        for label in labels:
            class_occurence[label] = [None] * n_splits
            for i in range(n_splits):
                class_occurence[label][i] = np.sum(fold_meta['scene_label'][train_ind[i]] == label)/train_ind[i].sum()

        # Put class occurences in a DataFrame
        df = pd.DataFrame(class_occurence)

        # Reference: 'a general measure of data-set imbalance' - Shannon Entropy
        # https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
        balance = -(df.values * np.log(df.values)).sum(axis=1) / np.log(len(labels))

        df['num_samples'] = np.sum(train_ind, axis=1)
        df['mean_balance'] = balance
        
        if verbose:
            print(df.to_string())
            print(f"total_balance: {df['mean_balance'].mean()}")

        return df

    

class FeatureExtractor():

    def __init__(self, fs=48000, win_length_seconds=40e-3, hop_length_seconds=20e-3, n_feats=100, n_fft=2048, window_type='hann', 
                 feature_type='mel_spectrogram', audio_preprocess='mid', minmax=False, audio_duration=10.0, **kwargs):
        super().__init__()

        self.__dict__.update(kwargs)
        
        self.fs = fs

        self.audio_preprocess = audio_preprocess
        self.minmax = minmax

        self.audio_length = int(audio_duration * self.fs)
        
        # Feature extractor input arguments
        args = dict(fs=fs, win_length_seconds=win_length_seconds, hop_length_seconds=hop_length_seconds, 
                n_mels=n_feats, n_mfcc=n_feats, n_dim=n_feats, spectrogram_type='magnitude', n_fft=n_fft, window_type=window_type, **kwargs)
        self.set_extractor(feature_type, args)

    def set_extractor(self, feature_type, args):
        self.feature_type = feature_type

        # Create feature extractor
        if feature_type == 'spectrogram':
            self.extractor = du.features.SpectralFeatureExtractor(**args)
        elif feature_type == 'mel_spectrogram':
            self.extractor = du.features.MelExtractor(**args)
        elif feature_type == 'mfcc_static':
            args['n_mels'] = int(n_fft/4)
            self.extractor = du.features.MfccStaticExtractor(**args)
        elif feature_type == 'mfcc_delta':
            args['n_mels'] = int(n_fft/4)
            self.extractor = du.features.MfccDeltaExtractor(**args)
        else:
            message = f'set_extractor(): Unknown feature type [{feature_type}]'
            raise ValueError(message)

    def process(self, audio_handle):

        feat_shape = self.get_shape()
        
        if type(audio_handle) is list:

            num_files = len(audio_handle)

            features = np.empty([num_files, *feat_shape])

            for i, file in enumerate(range(num_files)):
                features[i,] = self._single_process(audio_handle)

        else:
            features = np.expand_dims(self._single_process(audio_handle), axis=0)

        return features

    def _single_process(self, audio_handle):

        if type(audio_handle) not in [np.array, du.containers.AudioContainer]:
            # Open audio file
            audio_container = du.containers.AudioContainer().load(
                filename=str(audio_handle),
                mono=False,
                fs=self.fs
            )
            audio_data = audio_container.data

        # Convert audio to mono (left, right, mid or side)
        audio_data = self.audio_mixing(audio_data, self.audio_preprocess, self.minmax)

        # Fix audio duration
        audio_data = self.fix_audio_length(audio_data, self.audio_length)

        # Extract features from the audio data
        features = self.feature_extraction(audio_data)

        return features



    def fix_audio_length(self, audio_data, length):

        current_size = audio_data.size

        if current_size == length:
            return audio_data
        elif current_size < length:
            new_audio_data = np.empty(length)
            new_audio_data[:current_size] = audio_data.copy()
            new_audio_data[current_size:] = 0
            return new_audio_data
        else:
            new_audio_data = audio_data[:length].copy()
            return new_audio_data

    def audio_mixing(self, audio_data, audio_preprocess, minmax):

        if len(audio_data.shape) == 2:
            # Audio mixing
            if audio_preprocess == 'left':
                audio_data = audio_data[0,:]
            elif audio_preprocess == 'right':
                audio_data = audio_data[1,:]
            elif audio_preprocess == 'mid':
                audio_data = audio_data[0,:] + audio_data[1,:]
            elif audio_preprocess == 'side':
                audio_data = audio_data[0,:] - audio_data[1,:]
            else:
                message = f'audio_mixing(): Unknown audio_preprocess type [{audio_preprocess}]'
                raise ValueError(message)

        # Optional normalization per audio snippet
        if minmax:
            audio_data /= np.maximum(audio_data.max(),-audio_data.min())

        return audio_data

    def feature_extraction(self, audio_data):
        return self.extractor.extract(audio_data)

    def get_shape(self):

        # Temporary array
        a = np.empty(self.audio_length)

        # Get feature vector shape
        feature_shape = self.extractor.extract(a).shape

        return feature_shape


